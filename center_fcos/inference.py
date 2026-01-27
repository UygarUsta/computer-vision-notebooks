import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
import glob
from tqdm import tqdm

# =============================================================================
# 1. Config & Classes
# =============================================================================
class Config:
    NUM_CLASSES = 2 # Ensure this matches your trained model
    INPUT_SIZE = 512
    OUTPUT_STRIDE = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    VIS_THRESH = 0.15 # Threshold to filter low confidence boxes

COCO_CLASSES = [
    'apple','apple'
]

# =============================================================================
# 2. Model Definition (Must match training exactly)
# =============================================================================
class ResNetBackbone(nn.Module):
    def __init__(self, layers=18):
        super(ResNetBackbone, self).__init__()
        self.backbone = models.resnet18(pretrained=False) # No need for pretrained weights here, we load ours
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4: return 4, 1, 0
        elif deconv_kernel == 3: return 3, 1, 1
        elif deconv_kernel == 2: return 2, 0, 0

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        in_planes = 512
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_planes, planes, kernel, 2, padding, output_padding, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        return self.deconv_layers(x)

class CenterNetHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super(CenterNetHead, self).__init__()
        self.hm_head = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, num_classes, 1))
        self.ltrb_head = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes * 4, 1))
        self.reg_head = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 2, 1))
    
    def forward(self, x):
        hm = torch.sigmoid(self.hm_head(x))
        ltrb = torch.exp(self.ltrb_head(x))
        reg = self.reg_head(x)
        return hm, ltrb, reg

class CenterNet(nn.Module):
    def __init__(self):
        super(CenterNet, self).__init__()
        self.backbone = ResNetBackbone(layers=18)
        self.head = CenterNetHead(in_channels=256, num_classes=Config.NUM_CLASSES)
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

# =============================================================================
# 3. Helpers (Pre/Post Processing)
# =============================================================================
def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32)):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def ctdet_decode(hm, ltrb, reg, K=100):
    batch, cat, height, width = hm.size()
    hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hm == hm_pool).float()
    hm = hm * keep
    
    topk_scores, topk_inds = torch.topk(hm.view(batch, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    
    topk_clses = (torch.topk(hm.view(batch, -1), K)[1] // (height * width)).int().float()
    
    reg = _tranpose_and_gather_feat(reg, topk_inds)
    ltrb = _tranpose_and_gather_feat(ltrb, topk_inds) # [B, K, 320]
    
    topk_xs = topk_xs.view(batch, K, 1) + reg[:, :, 0:1]
    topk_ys = topk_ys.view(batch, K, 1) + reg[:, :, 1:2]
    
    bboxes = []
    for b in range(batch):
        b_box = []
        for k in range(K):
            cls_id = int(topk_clses[b, k])
            base_idx = cls_id * 4
            l = ltrb[b, k, base_idx]
            t = ltrb[b, k, base_idx+1]
            r = ltrb[b, k, base_idx+2]
            bb = ltrb[b, k, base_idx+3]
            
            x1 = topk_xs[b, k, 0] - l
            y1 = topk_ys[b, k, 0] - t
            x2 = topk_xs[b, k, 0] + r
            y2 = topk_ys[b, k, 0] + bb
            b_box.append([x1, y1, x2, y2])
        bboxes.append(torch.tensor(b_box).to(hm.device))
    return torch.stack(bboxes), topk_scores, topk_clses

# =============================================================================
# 4. Main Inference Function
# =============================================================================
def run_inference(image_path, model, output_dir):
    filename = os.path.basename(image_path)
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Could not read {image_path}")
        return

    # 1. Preprocess
    height, width = img_orig.shape[:2]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    
    trans_input = get_affine_transform(c, s, 0, [Config.INPUT_SIZE, Config.INPUT_SIZE])
    inp = cv2.warpAffine(img_orig, trans_input, (Config.INPUT_SIZE, Config.INPUT_SIZE), flags=cv2.INTER_LINEAR)
    
    # Normalize
    inp = (inp.astype(np.float32) / 255. - Config.MEAN) / Config.STD
    inp = inp.transpose(2, 0, 1)
    inp = torch.from_numpy(inp).unsqueeze(0).to(Config.DEVICE).float()

    # 2. Forward
    with torch.no_grad():
        hm, ltrb, reg = model(inp)
        # Decode
        bboxes, scores, clses = ctdet_decode(hm, ltrb, reg, K=100)

    # 3. Post-process (Rescale to original image)
    # The output is in "output resolution" (128x128). 
    # We need to map it back to Input Size (512) then Original Size.
    
    # First, scale up by stride to match Input Size (512x512)
    bboxes = bboxes * Config.OUTPUT_STRIDE 
    
    # Now invert affine transform to get back to original image
    # We can do this mathematically:
    # box = [x1, y1, x2, y2]
    # We can transform (x1,y1) and (x2,y2) using the inverse affine matrix
    
    # Easier way in CenterNet logic: Transform the center point and wh? 
    # No, let's just transform the 4 coordinates.
    trans_inv = cv2.invertAffineTransform(trans_input)
    
    bboxes_np = bboxes[0].cpu().numpy()
    scores_np = scores[0].cpu().numpy()
    clses_np = clses[0].cpu().numpy()
    
    for i in range(len(bboxes_np)):
        if scores_np[i] < Config.VIS_THRESH: continue
        
        x1, y1, x2, y2 = bboxes_np[i]
        
        # Transform points back
        pt1 = np.dot(trans_inv, np.array([x1, y1, 1]))
        pt2 = np.dot(trans_inv, np.array([x2, y2, 1]))
        
        # Draw
        x_min, y_min = int(pt1[0]), int(pt1[1])
        x_max, y_max = int(pt2[0]), int(pt2[1])
        
        cls_id = int(clses_np[i])
        if cls_id < len(COCO_CLASSES):
            label = f"{COCO_CLASSES[cls_id]} {scores_np[i]:.2f}"
        else:
            label = f"Class {cls_id} {scores_np[i]:.2f}"
            
        cv2.rectangle(img_orig, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img_orig, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    cv2.imwrite(os.path.join(output_dir, filename), img_orig)

if __name__ == '__main__':
    # --- USER SETTINGS ---
    MODEL_PATH = "centernet_ltrb_fcos.pth" 
    INPUT_DIR = "/home/uygarusta/Oriented-Centernet/center_fcos/Apple-Vision-3/test"       # Put your test images here
    OUTPUT_DIR = "output_vis"  # Results will be saved here
    # ---------------------
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = CenterNet().to(Config.DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Tip: If you trained on fewer classes, change Config.NUM_CLASSES in the script.")
        exit()
        
    model.eval()
    
    images = glob.glob(os.path.join(INPUT_DIR, "*.*"))
    print(f"Found {len(images)} images.")
    
    for img_path in tqdm(images):
        run_inference(img_path, model, OUTPUT_DIR)
        
    print(f"Done! Check {OUTPUT_DIR}")
