import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
from tqdm import tqdm

# =============================================================================
# 1. Configuration & Constants
# =============================================================================
class Config:
    NUM_CLASSES = 2
    INPUT_SIZE = 512
    OUTPUT_STRIDE = 4
    OUTPUT_SIZE = INPUT_SIZE // OUTPUT_STRIDE
    BATCH_SIZE = 8  # Adjust based on VRAM
    LR = 1.25e-4
    NUM_EPOCHS = 30 # Reduced for demo; usually 140+
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VIS_THRESH = 0.3
    # Mean/Std for ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

# =============================================================================
# 2. Utilities (Gaussian, IoU Loss, Encoding)
# =============================================================================

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def get_iou_loss(pred_ltrb, target_ltrb, weight=None):
    """
    IoU Loss for FCOS-style LTRB.
    pred: [N, 4] (l, t, r, b)
    target: [N, 4] (l, t, r, b)
    """
    pred_left = pred_ltrb[:, 0]
    pred_top = pred_ltrb[:, 1]
    pred_right = pred_ltrb[:, 2]
    pred_bottom = pred_ltrb[:, 3]

    target_left = target_ltrb[:, 0]
    target_top = target_ltrb[:, 1]
    target_right = target_ltrb[:, 2]
    target_bottom = target_ltrb[:, 3]

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

    # Clamp to zero to avoid issues with bad predictions
    w_intersect = w_intersect.clamp(min=0)
    h_intersect = h_intersect.clamp(min=0)
    
    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect + 1e-7

    iou = area_intersect / area_union
    loss = -torch.log(iou + 1e-7)

    if weight is not None:
        loss = loss * weight

    return loss.sum()

# =============================================================================
# 3. Dataset (COCO with FCOS+CenterNet Targets)
# =============================================================================
class COCOCenterNetDataset(Dataset):
    def __init__(self, root_dir, set_name='train', transform=None):
        self.coco = COCO(os.path.join(root_dir,set_name,f'_annotations.coco.json'))
        self.img_dir = os.path.join(root_dir, set_name)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.num_classes = Config.NUM_CLASSES
        self.output_size = Config.OUTPUT_SIZE
        self.input_size = Config.INPUT_SIZE
        
        # Class mapping (COCO ids are not contiguous 0-79)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_cls_ind = {cat['id']: i for i, cat in enumerate(cats)}
        self.cls_ind_to_cat_id = {i: cat['id'] for i, cat in enumerate(cats)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs([img_id])[0]
        
        path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        height, width = img.shape[:2]
        
        # Basic Preprocessing: Resize and Pad to square
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        
        # Affine transformation to input size
        trans_input = get_affine_transform(c, s, 0, [self.input_size, self.input_size])
        inp = cv2.warpAffine(img, trans_input, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR)
        
        # Normalize
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - Config.MEAN) / Config.STD
        inp = inp.transpose(2, 0, 1) # C, H, W

        # Prepare Targets
        hm = np.zeros((self.num_classes, self.output_size, self.output_size), dtype=np.float32)
        # 4 channels * num_classes for Per-Class LTRB
        ltrb = np.zeros((self.num_classes * 4, self.output_size, self.output_size), dtype=np.float32)
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        ind = np.zeros((128), dtype=np.int64) # Max 128 objs
        reg_mask = np.zeros((128), dtype=np.uint8)
        
        # We need to map which class the object at 'ind' belongs to for loss calculation
        # But since we use per-class output maps, we can use the map directly
        
        # Transformation for output map
        trans_output = get_affine_transform(c, s, 0, [self.output_size, self.output_size])

        draw_ct = 0
        for ann in anns:
            if ann.get('iscrowd', 0): continue
            cls_id = self.cat_id_to_cls_ind[ann['category_id']]
            
            bbox = ann['bbox'] # x, y, w, h
            if bbox[2] <= 0 or bbox[3] <= 0: continue
            
            # Transform bbox to output resolution
            rect = np.array([[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]]], dtype=np.float32)
            rect = affine_transform_pts(rect, trans_output)
            
            # Constraints
            rect[:, 0] = np.clip(rect[:, 0], 0, self.output_size - 1)
            rect[:, 1] = np.clip(rect[:, 1], 0, self.output_size - 1)
            
            h, w = rect[1, 1] - rect[0, 1], rect[1, 0] - rect[0, 0]
            
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                
                # Center calculation
                ct = np.array([(rect[0, 0] + rect[1, 0]) / 2, (rect[0, 1] + rect[1, 1]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                # Gaussian Heatmap
                draw_gaussian(hm[cls_id], ct_int, radius)
                
                # Regression Targets (FCOS Style LTRB)
                # l = x - x1, t = y - y1, r = x2 - x, b = y2 - y
                # Computed at the center point ct_int
                x1, y1 = rect[0, 0], rect[0, 1]
                x2, y2 = rect[1, 0], rect[1, 1]
                
                l = ct_int[0] - x1
                t = ct_int[1] - y1
                r = x2 - ct_int[0]
                b = y2 - ct_int[1]
                
                # Store in specific class channels [4*cls:4*(cls+1)]
                # Channel order: l, t, r, b
                base_idx = cls_id * 4
                ltrb[base_idx + 0, ct_int[1], ct_int[0]] = l
                ltrb[base_idx + 1, ct_int[1], ct_int[0]] = t
                ltrb[base_idx + 2, ct_int[1], ct_int[0]] = r
                ltrb[base_idx + 3, ct_int[1], ct_int[0]] = b
                
                # Local Offset (discretization error)
                reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]
                
                ind[draw_ct] = ct_int[1] * self.output_size + ct_int[0]
                reg_mask[draw_ct] = 1
                draw_ct += 1
                if draw_ct >= 128: break
        
        return {
            'input': torch.from_numpy(inp),
            'hm': torch.from_numpy(hm),
            'ltrb': torch.from_numpy(ltrb),
            'reg': torch.from_numpy(reg),
            'ind': torch.from_numpy(ind),
            'reg_mask': torch.from_numpy(reg_mask),
            'img_id': img_id
        }

# Helper functions for affine transforms
def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32)):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def affine_transform_pts(pts, trans):
    # pts: N x 2
    new_pts = np.hstack([pts, np.ones((pts.shape[0], 1))])
    new_pts = np.dot(trans, new_pts.T).T
    return new_pts

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

# =============================================================================
# 4. Model Architecture (ResNet + Deconv + LTRB Head)
# =============================================================================

class ResNetBackbone(nn.Module):
    def __init__(self, layers=18):
        super(ResNetBackbone, self).__init__()
        if layers == 18:
            self.backbone = models.resnet18(pretrained=True)
            self.channels = [64, 128, 256, 512]
        # Remove Average Pool and FC
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Deconvolution layers (3 layers to go from stride 32 to stride 4)
        # ResNet stride is 32 at layer4. We need stride 4.
        # 32 -> 16 -> 8 -> 4 (3 deconvs)
        self.deconv_layers = self._make_deconv_layer(3, [256, 256, 256], [4, 4, 4])
    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        in_planes = 512 # ResNet18 output channels
        
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_layers(x)
        return x

class CenterNetHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super(CenterNetHead, self).__init__()
        
        # Heatmap Head
        self.hm_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        self.hm_head[-1].bias.data.fill_(-2.19) # Initialize bias for focal loss stability
        
        # LTRB Head (Per Class)
        # Output dim: 4 * Num_Classes
        self.ltrb_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes * 4, 1) 
        )
        
        # Offset Head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1)
        )
    
    def forward(self, x):
        hm = torch.sigmoid(self.hm_head(x))
        # Use Exp to ensure positive distances for LTRB, common in FCOS
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
# 5. Loss Function
# =============================================================================

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class COCODenseCenterNetDataset(Dataset):
    def __init__(self, root_dir, set_name='train', transform=None):
        self.coco = COCO(os.path.join(root_dir, set_name, f'_annotations.coco.json'))
        self.img_dir = os.path.join(root_dir, set_name)
        self.ids = list(self.coco.imgs.keys())
        #self.num_classes = Config.NUM_CLASSES
        self.output_size = Config.OUTPUT_SIZE
        self.input_size = Config.INPUT_SIZE
        
        # Coordinate grid for vectorized target generation
        # Shape: [H, W] -> x and y grids
        y_range = np.arange(self.output_size, dtype=np.float32)
        x_range = np.arange(self.output_size, dtype=np.float32)
        self.y_grid, self.x_grid = np.meshgrid(y_range, x_range, indexing='ij')

        catIds = self.coco.getCatIds()
        
        cats = self.coco.loadCats(catIds)
        
        # Re-map the found categories to contiguous 0...N-1 indices
        # If you only loaded 'person', this map will be {1: 0}
        self.cat_id_to_cls_ind = {cat['id']: i for i, cat in enumerate(cats)}
        self.cls_ind_to_cat_id = {i: cat['id'] for i, cat in enumerate(cats)}
        
        # Update config num_classes dynamically to be safe
        self.num_classes = len(cats)

        print("Coco Categories:",cats)
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs([img_id])[0]
        
        path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # --- Preprocessing (Same as before) ---
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        trans_input = get_affine_transform(c, s, 0, [self.input_size, self.input_size])
        inp = cv2.warpAffine(img, trans_input, (self.input_size, self.input_size), flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255. - Config.MEAN) / Config.STD
        inp = inp.transpose(2, 0, 1)

        # --- Target Initialization ---
        # Heatmap (Sparse - still peaks only)
        hm = np.zeros((self.num_classes, self.output_size, self.output_size), dtype=np.float32)
        
        # LTRB (Dense - filled for entire box)
        # We need an "area_map" to resolve ambiguities (assign pixel to smallest box)
        ltrb = np.zeros((self.num_classes * 4, self.output_size, self.output_size), dtype=np.float32)
        area_buffer = np.full((self.output_size, self.output_size), float('inf'), dtype=np.float32)
        
        # Regression Mask (Tracks which pixels have a valid box)
        # We separate masks per class to ensure we don't punish "Car" channels for "Person" pixels
        reg_mask = np.zeros((self.num_classes, self.output_size, self.output_size), dtype=np.float32)
        
        # Offset (Sparse - typically only needed at the center for precise peak picking)
        # However, some variants make this dense too. We will keep it sparse (CenterNet style)
        # to ensure the peak location is precise.
        reg = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        ind = np.zeros((128), dtype=np.int64)
        offset_mask = np.zeros((128), dtype=np.uint8)

        trans_output = get_affine_transform(c, s, 0, [self.output_size, self.output_size])
        
        draw_ct = 0
        
        # Sort anns by area (descending) so smaller boxes overwrite larger ones naturally
        # though we use area_buffer check to be robust.
        anns = sorted(anns, key=lambda x: x['bbox'][2]*x['bbox'][3], reverse=True)

        for ann in anns:
            if ann.get('iscrowd', 0): continue
            cls_id = self.cat_id_to_cls_ind[ann['category_id']]
            bbox = ann['bbox']
            if bbox[2] <= 0 or bbox[3] <= 0: continue
            
            # Transform bbox
            rect = np.array([[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]]], dtype=np.float32)
            rect = affine_transform_pts(rect, trans_output)
            
            # Clip to map boundaries
            rect[:, 0] = np.clip(rect[:, 0], 0, self.output_size - 1)
            rect[:, 1] = np.clip(rect[:, 1], 0, self.output_size - 1)
            
            x1, y1 = rect[0, 0], rect[0, 1]
            x2, y2 = rect[1, 0], rect[1, 1]
            h, w = y2 - y1, x2 - x1
            
            if h > 0 and w > 0:
                area = h * w
                
                # --- 1. Heatmap (Standard CenterNet) ---
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                
                # --- 2. Dense LTRB Generation (FCOS Style) ---
                # Define the box region in integers
                ix1, iy1 = int(x1), int(y1)
                ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
                
                # Clip to output size (safety)
                ix1 = max(0, ix1); iy1 = max(0, iy1)
                ix2 = min(self.output_size, ix2); iy2 = min(self.output_size, iy2)

                # Select the grid patch corresponding to this box
                # grid_x shape: [Box_H, Box_W]
                grid_x = self.x_grid[iy1:iy2, ix1:ix2]
                grid_y = self.y_grid[iy1:iy2, ix1:ix2]
                
                # Calculate distances for the whole patch
                # l = x - x1, t = y - y1, r = x2 - x, b = y2 - y
                l_map = grid_x - x1
                t_map = grid_y - y1
                r_map = x2 - grid_x
                b_map = y2 - grid_y
                
                # FCOS constraint: targets must be positive
                # Due to sub-pixel coordinates, edge pixels might be slightly negative, clamp them
                l_map = np.maximum(l_map, 0.05)
                t_map = np.maximum(t_map, 0.05)
                r_map = np.maximum(r_map, 0.05)
                b_map = np.maximum(b_map, 0.05)
                
                # Check Ambiguity: Only update pixels where this box is smaller than existing ones
                current_area_patch = area_buffer[iy1:iy2, ix1:ix2]
                update_mask = area < current_area_patch
                
                # Update Area Buffer
                area_buffer[iy1:iy2, ix1:ix2][update_mask] = area
                
                # Update LTRB Maps (Per Class Channels)
                base_idx = cls_id * 4
                
                # We need to update [iy1:iy2, ix1:ix2] but only where update_mask is True
                # Numpy advanced indexing is tricky with slices, so we iterate channel-wise or mask
                
                # Helper to update a slice with a mask
                def update_slice(full_map, patch, y_slice, x_slice, mask):
                    current_slice = full_map[y_slice, x_slice]
                    current_slice[mask] = patch[mask]
                    full_map[y_slice, x_slice] = current_slice

                update_slice(ltrb[base_idx + 0], l_map, slice(iy1, iy2), slice(ix1, ix2), update_mask)
                update_slice(ltrb[base_idx + 1], t_map, slice(iy1, iy2), slice(ix1, ix2), update_mask)
                update_slice(ltrb[base_idx + 2], r_map, slice(iy1, iy2), slice(ix1, ix2), update_mask)
                update_slice(ltrb[base_idx + 3], b_map, slice(iy1, iy2), slice(ix1, ix2), update_mask)
                
                # Update Regression Mask (Per Class)
                update_slice(reg_mask[cls_id], np.ones_like(l_map), slice(iy1, iy2), slice(ix1, ix2), update_mask)
                
                # --- 3. Offset (Sparse - Peak Only) ---
                # We still only care about sub-pixel offset at the PEAK for inference decoding
                reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]
                
                ind[draw_ct] = ct_int[1] * self.output_size + ct_int[0]
                offset_mask[draw_ct] = 1
                draw_ct += 1
                if draw_ct >= 128: break
        
        return {
            'input': torch.from_numpy(inp),
            'hm': torch.from_numpy(hm),
            'ltrb': torch.from_numpy(ltrb),
            'reg': torch.from_numpy(reg),
            'ind': torch.from_numpy(ind),
            'offset_mask': torch.from_numpy(offset_mask),
            'reg_mask': torch.from_numpy(reg_mask), # Dense mask [Num_Classes, H, W]
            'img_id': img_id
        }


class DenseCenterNetLoss(nn.Module):
    def __init__(self):
        super(DenseCenterNetLoss, self).__init__()
        
    def forward(self, outputs, batch):
        hm_pred, ltrb_pred, reg_pred = outputs
        
        hm_true = batch['hm'].to(Config.DEVICE)
        ltrb_true = batch['ltrb'].to(Config.DEVICE) # [B, 320, H, W]
        reg_true = batch['reg'].to(Config.DEVICE)
        reg_mask_dense = batch['reg_mask'].to(Config.DEVICE) # [B, 80, H, W]
        
        # --- FIX: Force cast 'ind' to .long() (int64) ---
        ind = batch['ind'].to(Config.DEVICE).long() 
        
        offset_mask = batch['offset_mask'].to(Config.DEVICE)
        
        # --- 1. Heatmap Loss (Focal) ---
        pos_inds = hm_true.eq(1).float()
        neg_inds = hm_true.lt(1).float()
        neg_weights = torch.pow(1 - hm_true, 4)
        
        pos_loss = torch.log(hm_pred + 1e-6) * torch.pow(1 - hm_pred, 2) * pos_inds
        neg_loss = torch.log(1 - hm_pred + 1e-6) * torch.pow(hm_pred, 2) * neg_weights * neg_inds
        
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss_hm = -neg_loss.sum()
        else:
            loss_hm = -(pos_loss.sum() + neg_loss.sum()) / num_pos
            
        # --- 2. Offset Loss (Sparse L1) ---
        # We only care about offset at the exact peak to recover the float center
        reg_pred_gather = _tranpose_and_gather_feat(reg_pred, ind)
        reg_true_gather = _tranpose_and_gather_feat(reg_true, ind)
        offset_mask_ex = offset_mask.unsqueeze(2).expand_as(reg_pred_gather).float()
        
        loss_reg = F.l1_loss(reg_pred_gather * offset_mask_ex, 
                             reg_true_gather * offset_mask_ex, reduction='sum')
        loss_reg = loss_reg / (offset_mask.sum() + 1e-4)
        
        # --- 3. Dense LTRB IoU Loss (FCOS Style) ---
        # ltrb_pred: [B, 320, H, W]
        # ltrb_true: [B, 320, H, W]
        # reg_mask_dense: [B, 80, H, W] (Which class is active where)
        
        # We need to broadcast the mask to 320 channels (4 per class)
        # reg_mask_dense [B, 80, H, W] -> repeat interleave -> [B, 320, H, W]
        reg_mask_expanded = reg_mask_dense.repeat_interleave(4, dim=1)
        
        # Select only valid pixels to save computation and ensure correct mean
        # flatten valid pixels.
        
        valid_mask = reg_mask_expanded > 0
        if valid_mask.sum() > 0:
            pred_valid = ltrb_pred[valid_mask].view(-1, 4)
            true_valid = ltrb_true[valid_mask].view(-1, 4)
            
            # GIoU or IoU Loss
            loss_ltrb = get_iou_loss(pred_valid, true_valid)
            # Normalize by number of positive PIXELS (not objects), same as FCOS
            loss_ltrb = loss_ltrb / (valid_mask.sum() / 4 + 1e-4) 
        else:
            loss_ltrb = torch.tensor(0.0).to(Config.DEVICE)

        total_loss = loss_hm + 1.0 * loss_reg + 5.0 * loss_ltrb
        return total_loss, loss_hm, loss_ltrb, loss_reg
    
# =============================================================================
# 6. Inference Decoder
# =============================================================================
def ctdet_decode(hm, ltrb, reg, K=100):
    batch, cat, height, width = hm.size()
    
    # 1. MaxPool NMS on heatmap
    hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hm == hm_pool).float()
    hm = hm * keep
    
    # 2. Select top K scores
    topk_scores, topk_inds = torch.topk(hm.view(batch, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    
    # Get classes
    topk_clses = (torch.topk(hm.view(batch, -1), K)[1] // (height * width)).int().float()
    
    # 3. Retrieve Offset and LTRB
    # ltrb is [B, 320, H, W]
    reg = _tranpose_and_gather_feat(reg, topk_inds)
    ltrb = _tranpose_and_gather_feat(ltrb, topk_inds) # [B, K, 320]
    
    topk_xs = topk_xs.view(batch, K, 1) + reg[:, :, 0:1]
    topk_ys = topk_ys.view(batch, K, 1) + reg[:, :, 1:2]
    
    # 4. Extract specific class LTRB
    # ltrb shape is [B, K, 320]. We need to pluck the 4 channels based on topk_clses
    # topk_clses is [B, K]
    
    bboxes = []
    for b in range(batch):
        b_box = []
        for k in range(K):
            cls_id = int(topk_clses[b, k])
            base_idx = cls_id * 4
            # l, t, r, b
            l = ltrb[b, k, base_idx]
            t = ltrb[b, k, base_idx+1]
            r = ltrb[b, k, base_idx+2]
            bb = ltrb[b, k, base_idx+3]
            
            ct_x = topk_xs[b, k, 0]
            ct_y = topk_ys[b, k, 0]
            
            x1 = ct_x - l
            y1 = ct_y - t
            x2 = ct_x + r
            y2 = ct_y + bb
            
            b_box.append([x1, y1, x2, y2])
        bboxes.append(torch.tensor(b_box).to(hm.device))
        
    bboxes = torch.stack(bboxes) # [B, K, 4]
    
    return bboxes, topk_scores, topk_clses

# =============================================================================
# 7. Training & Evaluation Engine
# =============================================================================
def train_one_epoch(model, optimizer, loader, loss_fn):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        for k in batch:
            if k != 'img_id':
                batch[k] = batch[k].to(Config.DEVICE).float()
        
        outputs = model(batch['input'])
        loss, l_hm, l_ltrb, l_reg = loss_fn(outputs, batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'l_hm': l_hm.item(), 'l_ltrb': l_ltrb.item()})
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_coco(model, loader, coco_gt):
    model.eval()
    results = []
    
    print("Generating predictions for Eval...")
    for batch in tqdm(loader):
        inputs = batch['input'].to(Config.DEVICE).float()
        img_ids = batch['img_id']
        
        # 1. Forward
        hm, ltrb, reg = model(inputs)
        
        # 2. Decode (Returns [B, K, 4] in x1,y1,x2,y2 format relative to output size)
        bboxes, scores, clses = ctdet_decode(hm, ltrb, reg, K=100)
        
        # 3. Rescale to Input Size (512x512)
        bboxes = bboxes * Config.OUTPUT_STRIDE
        
        # 4. Map back to Original Image Size
        # We need the original image dimensions to scale correctly. 
        # The DataLoader logic resized the *longest side* to 512.
        
        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        clses = clses.cpu().numpy()
        
        for b in range(inputs.size(0)):
            img_id = int(img_ids[b])
            img_info = coco_gt.loadImgs(img_id)[0]
            orig_h, orig_w = img_info['height'], img_info['width']
            
            # Re-calculate the affine transform used in preprocessing
            # to invert it perfectly
            c = np.array([orig_w / 2., orig_h / 2.], dtype=np.float32)
            s = max(orig_h, orig_w) * 1.0
            
            # Transform from Input Size (512x512) -> Original Size
            trans_input = get_affine_transform(c, s, 0, [Config.INPUT_SIZE, Config.INPUT_SIZE])
            trans_inv = cv2.invertAffineTransform(trans_input)
            
            for k in range(100):
                if scores[b, k] < 0.05: continue # Skip noise
                
                # Get the class ID
                cls_idx = int(clses[b, k])
                if cls_idx not in loader.dataset.cls_ind_to_cat_id: continue
                cat_id = loader.dataset.cls_ind_to_cat_id[cls_idx]
                
                # Raw Box (x1, y1, x2, y2) in 512x512 scale
                raw_box = bboxes[b, k]
                
                # Transform points
                pt1 = np.dot(trans_inv, np.array([raw_box[0], raw_box[1], 1]))
                pt2 = np.dot(trans_inv, np.array([raw_box[2], raw_box[3], 1]))
                
                x1, y1 = pt1[0], pt1[1]
                x2, y2 = pt2[0], pt2[1]
                
                # Convert to [x, y, w, h]
                w_box = x2 - x1
                h_box = y2 - y1
                
                # Validation: Skip invalid boxes
                if w_box <= 0 or h_box <= 0: continue
                
                # COCO Format: [x_min, y_min, width, height]
                coco_box = [
                    float(x1), 
                    float(y1), 
                    float(w_box), 
                    float(h_box)
                ]
                
                results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": coco_box,
                    "score": float(scores[b, k])
                })
                
    if not results:
        print("No detections found for eval.")
        return
        
    print(f"Evaluating on {len(results)} detections...")
    
    # Save to JSON
    import json
    res_file = "results.json"
    with open(res_file, "w") as f:
        json.dump(results, f)
        
    try:
        coco_dt = coco_gt.loadRes(res_file)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"COCO Eval Error: {e}")
    finally:
        if os.path.exists(res_file):
            os.remove(res_file)

# =============================================================================
# 8. Main Execution
# =============================================================================

def main():
    # PATHS - Update these
    COCO_ROOT = '/home/uygarusta/Oriented-Centernet/center_fcos/Apple-Vision-3/' 
    
    # Check if COCO exists, else mock for syntax check
    if not os.path.exists(COCO_ROOT):
        print(f"Warning: COCO root {COCO_ROOT} not found. Code is ready but needs data.")
        return

    # Datasets
    train_ds = COCODenseCenterNetDataset(COCO_ROOT, 'train')
    val_ds = COCODenseCenterNetDataset(COCO_ROOT, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = CenterNet().to(Config.DEVICE)
    loss_fn = DenseCenterNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print(f"Starting training on {Config.DEVICE}...")
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        train_loss = train_one_epoch(model, optimizer, train_loader, loss_fn)
        scheduler.step()
        print(f"Epoch Loss: {train_loss:.4f}")
        
        # Eval every 5 epochs
        if (epoch + 1) % 3 == 0:
            evaluate_coco(model, val_loader, val_ds.coco)
            torch.save(model.state_dict(), "centernet_ltrb_fcos.pth")
            
    # Final Eval
    evaluate_coco(model, val_loader, val_ds.coco)
    
    torch.save(model.state_dict(), "centernet_ltrb_fcos.pth")
    # Save
    

if __name__ == '__main__':
    main()
