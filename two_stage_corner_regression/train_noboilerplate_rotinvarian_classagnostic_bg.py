import cv2
import numpy as np
import os
import json
import math
import random
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

# Modeli dışarıdan alıyoruz (Kendi dosyanıza göre yolu ayarlayın)
from cornerpool_deneme_offset_rotinvariant import get_pose_net

assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 1. KONFİGÜRASYON VE HİPERPARAMETRELER
# ==========================================
DATASET_PATH = "/home/uygarusta/datasets/gbsv2/*" #"/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
MODEL_SAVE_FOLDER = "modelWeights_rotinvariant_classagnostic"
os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)

neg_bg_folder = "/home/uygarusta/datasets/gbsv2_background/diger/"
INPUT_WIDTH = 512
INPUT_HEIGHT = 512
MODEL_SCALE = 4
BATCH_SIZE = 4
WORKERS = 4
EPOCHS = 100
IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. YARDIMCI FONKSİYONLAR (GEOMETRİ & SINIFLAR)
# ==========================================
def get_unique_classes(dataset_path):
    """JSON dosyalarını tarayıp eşsiz sınıfları otomatik olarak çıkarır."""
    unique_classes = set()
    json_files = glob.glob(os.path.join(dataset_path, "*.json"))
    for j_file in json_files:
        with open(j_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label:
                    if label == "ruhsata": label = "ruhsat" # Düzeltme kuralı
                    if label != "alan":
                        label = "doc"
                    unique_classes.add(label)
    return sorted(list(unique_classes))

CLASSES = get_unique_classes(DATASET_PATH)
print(f"Otomatik Algılanan Sınıflar ({len(CLASSES)} adet): {CLASSES}")

with open("classes.txt", "w") as file:
    # Joins items with a newline character
    file.write("\n".join(CLASSES))

def calculate_center_from_points(points):
    points = np.array(points, dtype=np.float32)
    return np.mean(points[:, 0]), np.mean(points[:, 1])

def poly_iou(pts_a, pts_b):
    """Shapely kullanarak iki dörtgen arası IoU hesaplar."""
    try:
        pa, pb = Polygon(pts_a), Polygon(pts_b)
        if not pa.is_valid or not pb.is_valid: return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

# ==========================================
# 3. CENTERNET ISI HARİTASI (HEATMAP) ÜRETİMİ
# ==========================================
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1; b1 = (height + width); c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2 = 4; b2 = 2 * (height + width); c2 = (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3 = 4 * min_overlap; b3 = -2 * min_overlap * (height + width); c3 = (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)

def draw_offset(offset, x, y):
    H, W = offset.shape[1], offset.shape[2]
    clipped_x, clipped_y = np.clip(x, 0, W - 1 - 0.001), np.clip(y, 0, H - 1 - 0.001)
    offset[0, int(clipped_y), int(clipped_x)] = clipped_x - int(clipped_x)
    offset[1, int(clipped_y), int(clipped_x)] = clipped_y - int(clipped_y)
    return offset

def apply_dense_offset(offset_map, mask_map, float_pt, int_pt, start_idx, fm_w, fm_h, radius=2):
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = int_pt[0] + dx, int_pt[1] + dy
            if 0 <= px < fm_w and 0 <= py < fm_h:
                offset_map[start_idx, py, px] = float_pt[0] - px
                offset_map[start_idx+1, py, px] = float_pt[1] - py
                mask_map[start_idx:start_idx+2, py, px] = 1.0


def apply_dense_regr(regr, mask, scaled_corners, scaled_cx, scaled_cy, ct_int_x, ct_int_y, fm_w, fm_h, radius=1):
    """
    Fill regr and mask in a (2*radius+1)^2 neighbourhood around the center.
    Each pixel (px, py) gets the offset to each corner relative to *that pixel*,
    not relative to the float center.
    """
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px = ct_int_x + dx
            py = ct_int_y + dy
            if not (0 <= px < fm_w and 0 <= py < fm_h):
                continue
            for i, (cx_corner, cy_corner) in enumerate(scaled_corners):
                regr[i*2,     py, px] = cx_corner - px   # not - scaled_cx
                regr[i*2 + 1, py, px] = cy_corner - py
            mask[py, px] = 1

def make_hm_offset_regr_angle(target):
    fm_h, fm_w = INPUT_HEIGHT // MODEL_SCALE, INPUT_WIDTH // MODEL_SCALE
    hm = np.zeros([len(CLASSES), fm_h, fm_w])
    hm_corners = np.zeros([4, fm_h, fm_w]) 
    offset = np.zeros([2, fm_h, fm_w])
    corner_offset = np.zeros([len(CLASSES) * 8, fm_h, fm_w]) 
    regr = np.zeros([4 * 2, fm_h, fm_w])
    mask  = np.zeros((fm_h, fm_w), dtype=np.float32)
    corner_offset_mask = np.zeros([len(CLASSES) * 8, fm_h, fm_w], dtype=np.float32)

    if len(target) == 0:
        return hm, offset, regr, mask, hm_corners, corner_offset, corner_offset_mask

    for obj_idx, i in enumerate(target):
        cx, cy, tl, tr, br, bl, cls = i
        
        obj_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        obj_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        
        scaled_cx, scaled_cy = cx / MODEL_SCALE, cy / MODEL_SCALE
        scaled_tl, scaled_tr = tl / MODEL_SCALE, tr / MODEL_SCALE
        scaled_br, scaled_bl = br / MODEL_SCALE, bl / MODEL_SCALE
        
        ct_int_x, ct_int_y = int(np.clip(scaled_cx, 0, fm_w - 0.001)), int(np.clip(scaled_cy, 0, fm_h - 0.001))
        ct_int = (ct_int_x, ct_int_y)
        
        radius = max(0, int(gaussian_radius((obj_height / MODEL_SCALE, obj_width / MODEL_SCALE))))
        hm[cls] = draw_gaussian(hm[cls], ct_int, radius)
        
        tl_c, tr_c = (int(scaled_tl[0]), int(scaled_tl[1])), (int(scaled_tr[0]), int(scaled_tr[1]))
        br_c, bl_c = (int(scaled_br[0]), int(scaled_br[1])), (int(scaled_bl[0]), int(scaled_bl[1]))
        
        corner_radius = max(2, int(radius / 3.5))
        
        hm_corners[0] = draw_gaussian(hm_corners[0], tl_c, corner_radius)
        hm_corners[1] = draw_gaussian(hm_corners[1], tr_c, corner_radius)
        hm_corners[2] = draw_gaussian(hm_corners[2], br_c, corner_radius)
        hm_corners[3] = draw_gaussian(hm_corners[3], bl_c, corner_radius)

        offset_radius = max(1, min(3, int(corner_radius * 0.5)))
        

        cls_base = cls * 8
        apply_dense_offset(corner_offset, corner_offset_mask, scaled_tl, tl_c, cls_base + 0, fm_w, fm_h, radius=offset_radius) #was fixed 3 for all corner offsets, experimenting with dynamic radius
        apply_dense_offset(corner_offset, corner_offset_mask, scaled_tr, tr_c, cls_base + 2, fm_w, fm_h, radius=offset_radius)
        apply_dense_offset(corner_offset, corner_offset_mask, scaled_br, br_c, cls_base + 4, fm_w, fm_h, radius=offset_radius)
        apply_dense_offset(corner_offset, corner_offset_mask, scaled_bl, bl_c, cls_base + 6, fm_w, fm_h, radius=offset_radius)

        offset = draw_offset(offset, scaled_cx, scaled_cy)

        # regr[0, ct_int_y, ct_int_x] = scaled_tl[0] - scaled_cx
        # regr[1, ct_int_y, ct_int_x] = scaled_tl[1] - scaled_cy
        # regr[2, ct_int_y, ct_int_x] = scaled_tr[0] - scaled_cx
        # regr[3, ct_int_y, ct_int_x] = scaled_tr[1] - scaled_cy
        # regr[4, ct_int_y, ct_int_x] = scaled_br[0] - scaled_cx
        # regr[5, ct_int_y, ct_int_x] = scaled_br[1] - scaled_cy
        # regr[6, ct_int_y, ct_int_x] = scaled_bl[0] - scaled_cx
        # regr[7, ct_int_y, ct_int_x] = scaled_bl[1] - scaled_cy
        
        # mask[ct_int_y, ct_int_x] = 1
        scaled_corners = [
            (scaled_tl[0], scaled_tl[1]),
            (scaled_tr[0], scaled_tr[1]),
            (scaled_br[0], scaled_br[1]),
            (scaled_bl[0], scaled_bl[1]),
        ]
        apply_dense_regr(regr, mask, scaled_corners, scaled_cx, scaled_cy,
                 ct_int_x, ct_int_y, fm_w, fm_h, radius=1)

    return hm, offset, regr, mask, hm_corners, corner_offset, corner_offset_mask

# ==========================================
# 4. VERİ YÜKLEME VE ARTIRMA (AUGMENTATION)
# ==========================================
def process_json_raw(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    annotations = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label == "ruhsata": label = "ruhsat"
        if label != "alan":
            label = "doc"
        if label in CLASSES:
            points = np.array(shape["points"], dtype=np.float32)
            if len(points) == 4:
                cx, cy = calculate_center_from_points(points)
                tl, tr, br, bl = points[3], points[2], points[1], points[0]
                annotations.append([cx, cy, tl, tr, br, bl, CLASSES.index(label)])
    return annotations


def create_mosaic_image(images, all_annotations, target_size=(512, 512)):
    """
    Creates a mosaic image by combining four input images and their annotations.

    Args:
        images (list): A list of four images (as numpy arrays).
        all_annotations (list): A list of four corresponding annotation lists.
        target_size (tuple): The final (width, height) of the mosaic image.

    Returns:
        tuple: (mosaic_image, mosaic_annotations)
    """
    output_w, output_h = target_size
    mosaic_img = np.full((output_h, output_w, 3), 114, dtype=np.uint8) # Use a neutral gray background
    mosaic_annotations = []

    # Define the center point for splitting the canvas
    center_x, center_y = output_w // 2, output_h // 2
    quadrant_size = (center_x, center_y)

    # Positions for pasting the four images: top-left, top-right, bottom-left, bottom-right
    paste_positions = [(0, 0), (center_x, 0), (0, center_y), (center_x, center_y)]

    for i in range(4):
        img, annotations = images[i], all_annotations[i]
        orig_h, orig_w = img.shape[:2]

        # Resize image to fit into its quadrant
        resized_img = cv2.resize(img, quadrant_size, interpolation=cv2.INTER_AREA)

        # Get the pasting coordinates for this quadrant
        paste_x, paste_y = paste_positions[i]
        mosaic_img[paste_y:paste_y + center_y, paste_x:paste_x + center_x] = resized_img

        # Calculate scaling factors for annotation coordinates
        scale_x = quadrant_size[0] / orig_w
        scale_y = quadrant_size[1] / orig_h

        # Transform all annotation points for the current image
        for anno in annotations:
            # Unpack the original annotation
            orig_cx, orig_cy, tl, tr, br, bl, class_id = anno
            
            # Scale and translate each corner point to its new position on the mosaic canvas
            new_tl = np.array([tl[0] * scale_x + paste_x, tl[1] * scale_y + paste_y])
            new_tr = np.array([tr[0] * scale_x + paste_x, tr[1] * scale_y + paste_y])
            new_br = np.array([br[0] * scale_x + paste_x, br[1] * scale_y + paste_y])
            new_bl = np.array([bl[0] * scale_x + paste_x, bl[1] * scale_y + paste_y])
            
            # Recalculate the center point from the newly transformed corners
            new_cx = np.mean([new_tl[0], new_tr[0], new_br[0], new_bl[0]])
            new_cy = np.mean([new_tl[1], new_tr[1], new_br[1], new_bl[1]])
            
            mosaic_annotations.append([new_cx, new_cy, new_tl, new_tr, new_br, new_bl, class_id])
            
    return mosaic_img, mosaic_annotations



def rotate_and_fit_to_size(image, annotations, target_size=(512, 512), angle_range=(-45, 45), fixed_angles=None,border_value=(0, 0, 0)):
    """
    Rotates an image, then resizes and pads it to fit a fixed target size,
    updating annotations accordingly.

    Args:
        image (numpy.ndarray): The input image.
        annotations (list): A list of annotations.
        target_size (tuple): The final (width, height) of the output image.
        angle_range (tuple): A tuple (min_angle, max_angle) for random rotation.
        border_value (tuple): RGB color for padding. Black by default.

    Returns:
        tuple: (final_image, final_annotations)
    """
    # === 1. Perform initial rotation to prevent cropping ===
    (h, w) = image.shape[:2]
    center_image = (w // 2, h // 2)
    if fixed_angles is not None:
        angle = random.choice(fixed_angles)
    else:
        angle = random.uniform(angle_range[0], angle_range[1])
    #angle = random.uniform(angle_range[0], angle_range[1])
    
    M = cv2.getRotationMatrix2D(center_image, angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center_image[0]
    M[1, 2] += (new_h / 2) - center_image[1]

    # Get the intermediate rotated image and annotations
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=border_value)
    
    # Transform annotations to the intermediate coordinate system
    temp_annotations = []
    for anno in annotations:
        # Original annotation: [cx, cy, tl, tr, br, bl, class_id]
        points_to_transform = np.array(anno[2:6], dtype=np.float32)
        transformed_points = cv2.transform(points_to_transform.reshape(-1, 1, 2), M).reshape(-1, 2)
        temp_annotations.append(transformed_points)

    # === 2. Resize and pad to fit the target size ===
    target_w, target_h = target_size

    # Calculate scaling factor to fit inside target_size
    scale = min(target_w / new_w, target_h / new_h)
    resized_w, resized_h = int(new_w * scale), int(new_h * scale)

    if resized_w > 0 and resized_h > 0:
        resized_image = cv2.resize(rotated_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    else:
        # Avoid errors if the image becomes too small
        return None, None

    # Create final canvas and calculate padding
    final_image = np.full((target_h, target_w, 3), border_value, dtype=np.uint8)
    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    
    # Paste the resized image onto the center of the final canvas
    final_image[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized_image

    # === 3. Transform annotations to the final 512x512 canvas ===
    final_annotations = []
    for corners in temp_annotations:
        # Apply the same scaling and translation (padding) to the annotation points
        final_corners = (corners * scale) + np.array([pad_x, pad_y])

        # Recalculate the center point on the final canvas
        final_cx = np.mean(final_corners[:, 0])
        final_cy = np.mean(final_corners[:, 1])

        # Find the original class_id (assuming annotation order is preserved)
        original_anno = annotations[len(final_annotations)]
        class_id = original_anno[6]
        
        final_annotations.append([
            final_cx, final_cy,
            final_corners[0], final_corners[1],
            final_corners[2], final_corners[3],
            class_id
        ])
        
    return final_image, final_annotations

def _adjust_brightness_contrast(image, bc_range=(0.7, 1.3)):
    """
    Adjusts image brightness and contrast.
    """
    alpha = random.uniform(bc_range[0], bc_range[1]) # contrast control
    beta = random.uniform(-20, 20) # brightness control
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def _add_gaussian_noise(image, noise_std_dev=10):
    """
    Adds Gaussian noise to the image.
    """
    noise = np.random.normal(0, noise_std_dev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise) # Add noise, clipping automatically
    return noisy_image

def _adjust_hue(image, hue_range=(-18, 18)):
    """
    Adjusts the hue of the image.
    Hue range should be in degrees, typically -180 to 180.
    OpenCV's HSV hue range is 0-179 for 8-bit images.
    So, 18 degrees corresponds to 180 / 18 = 10 units in OpenCV's hue.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Convert hue_range from degrees to OpenCV's 0-179 scale for 8-bit images
    # A change of 18 degrees is 1 unit in OpenCV hue (180/18 = 10 -> 18/10 = 1.8 unit, but for 180 deg range, it is 1 deg = 1 unit for 360 deg range, so 0.5 for 180 deg range)
    # The hue in OpenCV is 0-179. A full 360 degree circle is mapped to 0-179.
    # So, a change of 'X' degrees in a 360-degree circle is 'X/2' in OpenCV's 0-179 range.
    # Therefore, hue_range=(-18, 18) in degrees means an actual shift range of (-9, 9) in OpenCV's hue units.
    hue_shift_cv = random.uniform(hue_range[0] / 2, hue_range[1] / 2) # Divide by 2 for OpenCV's hue scale

    h_float = h.astype(np.float32)
    h_float = (h_float + hue_shift_cv) % 180 # Hue wraps around 0-179
    h_float = np.clip(h_float, 0, 179) # Ensure values are within valid range

    h = h_float.astype(np.uint8)
    
    hsv_adjusted = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB)



def _apply_perspective_transform(image, annotations, perspective_magnitude=0.05):
    """
    Applies a random perspective transform to the image and updates its corner point annotations.

    Args:
        image (numpy.ndarray): The input image (e.g., from cv2.imread).
        annotations (list): A list of annotations.
        perspective_magnitude (float): Controls the intensity of the perspective distortion.
                                      Typically a value between 0 and 1.
    Returns:
        tuple: (transformed_image, transformed_annotations)
    """
    h, w = image.shape[:2]

    # Define original corners of the image
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # Define random offsets for destination corners
    # These offsets determine the perspective distortion
    offset_x = w * perspective_magnitude
    offset_y = h * perspective_magnitude

    # Top-left, Top-right, Bottom-right, Bottom-left
    dst_pts = np.float32([
        [random.uniform(0, offset_x), random.uniform(0, offset_y)],
        [w - 1 - random.uniform(0, offset_x), random.uniform(0, offset_y)],
        [w - 1 - random.uniform(0, offset_x), h - 1 - random.uniform(0, offset_y)],
        [random.uniform(0, offset_x), h - 1 - random.uniform(0, offset_y)]
    ])

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Perform the perspective transform on the image
    transformed_image = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    transformed_annotations = []
    for anno in annotations:
        cx, cy, tl, tr, br, bl, class_id = anno

        points_to_transform = np.array([tl, tr, br, bl], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(points_to_transform.reshape(-1, 1, 2), M).reshape(-1, 2)

        # Recalculate center based on new bounding box of transformed points
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])

        new_cx = (min_x + max_x) / 2.0
        new_cy = (min_y + max_y) / 2.0

        transformed_annotations.append([
            new_cx,
            new_cy,
            transformed_corners[0], # Top-left
            transformed_corners[1], # Top-right
            transformed_corners[2], # Bottom-right
            transformed_corners[3], # Bottom-left
            class_id
        ])
    return transformed_image, transformed_annotations


def _random_resized_crop(image, annotations, target_size, scale_range=(0.4, 1.0), ratio_range=(3/4, 4/3)):
    """
    Görüntüden rastgele bir bölgeyi kırpar, hedef boyuta yeniden boyutlandırır.
    Eğer köşe noktaları dışarıda kalırsa, onları görüntünün kenarına 'sınırlar' (clamp).
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 10 Deneme hakkı (uygun bir crop bulmak için)
    for _ in range(10): 
        area = h * w
        target_area = random.uniform(*scale_range) * area
        log_ratio = (math.log(ratio_range[0]), math.log(ratio_range[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < crop_w <= w and 0 < crop_h <= h:
            # Sol üst köşe koordinatlarını rastgele seç
            x_start = random.randint(0, w - crop_w)
            y_start = random.randint(0, h - crop_h)

            # 1. Görüntüyü Kırp
            cropped_image = image[y_start:y_start+crop_h, x_start:x_start+crop_w]
            
            # 2. Hedef boyuta (örn: 512x512) boyutlandır (Zoom etkisi)
            final_image = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            scale_x = target_w / crop_w
            scale_y = target_h / crop_h
            
            new_annotations = []
            for anno in annotations:
                # Orijinal annotation'dan verileri al (cx, cy burada kullanılmayacak, yeniden hesaplanacak)
                _, _, tl, tr, br, bl, class_id = anno
                
                # Köşeleri bir array'e topla
                points = np.array([tl, tr, br, bl], dtype=np.float32)
                
                # 3. Noktaları önce crop koordinatlarına taşı, sonra scale et
                points[:, 0] = (points[:, 0] - x_start) * scale_x
                points[:, 1] = (points[:, 1] - y_start) * scale_y
                
                # 4. CLAMP İŞLEMİ (İsteğin burası)
                # Noktaları 0 ile target_w/h arasına sıkıştır.
                # Eğer nokta -50 ise 0 olur. Eğer 600 ise 512 olur.
                points[:, 0] = np.clip(points[:, 0], 0, target_w - 1)
                points[:, 1] = np.clip(points[:, 1], 0, target_h - 1)
                
                # Yeni köşe noktaları
                new_tl, new_tr, new_br, new_bl = points[0], points[1], points[2], points[3]

                # 5. Yeni Merkez Hesaplama
                # Köşeler sıkıştırıldığı için merkez değişti. Yeni "görünür" merkez:
                # CenterNet için kritik: Merkez mutlaka görüntü içinde olmalı.
                new_cx = np.mean(points[:, 0])
                new_cy = np.mean(points[:, 1])
                
                # 6. Kontrol: Nesne tamamen yok oldu mu?
                # Eğer bütün noktalar aynı kenara yapıştıysa nesne görünmüyordur.
                # Basitçe bounding box genişlik ve yüksekliğine bakabiliriz.
                min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
                min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
                
                obj_w = max_x - min_x
                obj_h = max_y - min_y
                
                # Eğer nesne çok çok küçük kaldıysa (örn: 5 pikselden az) onu eğitime katmayalım, gürültü yapar.
                if obj_w < 5 or obj_h < 5:
                    continue

                new_annotations.append([new_cx, new_cy, new_tl, new_tr, new_br, new_bl, class_id])
            
            # Eğer annotation listesi boşalmadıysa veya en azından görüntü geçerliyse döndür
            return final_image, new_annotations

    # Fallback: Crop yapılamazsa resize et ve dön
    resized_img = cv2.resize(image, (target_w, target_h))
    sx, sy = target_w/w, target_h/h
    res_annos = []
    for anno in annotations:
         cx, cy, tl, tr, br, bl, cid = anno
         # Resize durumunda da clamp yapmaya gerek yok ama garanti olsun diye scale ediyoruz
         res_annos.append([cx*sx, cy*sy, tl*sx, tr*sx, br*sx, bl*sx, cid])
         
    return resized_img, res_annos

def sort_four_points(points):
    """
    Sorts a list of four 2D points into the order:
    [top-left, top-right, bottom-right, bottom-left].

    Args:
        points (list of lists or np.ndarray): A list of 4 points,
                                               e.g., [[x1, y1], [x2, y2], ...].

    Returns:
        np.ndarray: A 4x2 NumPy array of sorted points.
    """
    # Convert to a NumPy array for easier calculations
    points = np.array(points, dtype=np.float32)
    
    # Initialize the array for sorted points
    rect = np.zeros((4, 2), dtype=np.float32)

    # 1. Sum the coordinates:
    # The top-left point will have the smallest sum (x + y)
    # The bottom-right point will have the largest sum (x + y)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)] # top-left
    rect[2] = points[np.argmax(s)] # bottom-right

    # 2. Difference the coordinates:
    # The top-right point will have the smallest difference (x - y)
    # The bottom-left point will have the largest difference (x - y)
    diff = np.diff(points, axis=1) # Calculates x - y
    rect[1] = points[np.argmin(diff)] # top-right
    rect[3] = points[np.argmax(diff)] # bottom-left

    return rect


def calculate_center_from_points(points):
    """
    Calculates the center point (centroid) from a list of 2D points.

    Args:
        points (list of lists or np.ndarray): A list of points,
                                               e.g., [[x1, y1], [x2, y2], ...].

    Returns:
        tuple: A tuple (center_x, center_y).
    """
    points = np.array(points, dtype=np.float32)
    
    # Calculate the mean of all x-coordinates and all y-coordinates
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    
    return (center_x, center_y)


def _horizontal_flip(image, annotations):
    """
    Applies horizontal flip to the image and updates annotations.
    """
    flipped_image = cv2.flip(image, 1) # 1 for horizontal flip
    w = image.shape[1]
    flipped_annotations = []
    for anno in annotations:
        cx, cy, tl, tr, br, bl, class_id = anno
        
        # Flip x-coordinates of center and corners
        new_cx = w - cx - 1
        new_tl = np.array([w - tl[0] - 1, tl[1]])
        new_tr = np.array([w - tr[0] - 1, tr[1]])
        new_br = np.array([w - br[0] - 1, br[1]])
        new_bl = np.array([w - bl[0] - 1, bl[1]])

        # For horizontal flip, the new order of sorted points (TL, TR, BR, BL) will be:
        # TR_original becomes new TL
        # TL_original becomes new TR
        # BL_original becomes new BR
        # BR_original becomes new BL
        # However, it's safer to re-sort them explicitly after flipping if your system relies on sorted points.
        # Given sort_four_points is available, let's use it for robustness.
        
        # Create an array of flipped raw corner points, then sort them.
        raw_flipped_points = np.array([new_tl, new_tr, new_br, new_bl], dtype=np.float32)
        sorted_flipped_points = sort_four_points(raw_flipped_points)

        flipped_annotations.append([
            new_cx,
            cy, # Y-coordinate remains the same for horizontal flip
            sorted_flipped_points[0], # Top-left (after re-sorting)
            sorted_flipped_points[1], # Top-right (after re-sorting)
            sorted_flipped_points[2], # Bottom-right (after re-sorting)
            sorted_flipped_points[3], # Bottom-left (after re-sorting)
            class_id
        ])
    return flipped_image, flipped_annotations

class Normalize(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.norm = transforms.Normalize(self.mean, self.std)
    def __call__(self, image):
        image = image.astype(np.float32) / 255
        image -= self.mean
        image /= self.std
        return image
    

class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_id, transform=None, mosaic_prob=0.5, rotation_prob=0.4, brightness_contrast_prob=0.3, gaussian_noise_prob=0.3, hue_prob=0.3, perspective_prob=0.2, horizontal_flip_prob=0,
                 crop_prob=0.5,
                 neg_bg_folder=neg_bg_folder, neg_bg_ratio=0.1):
        self.img_id = img_id
        self.transform = transform
        self.normalize = Normalize()
        
        # --- Augmentation Probabilities ---
        self.mosaic_prob = mosaic_prob
        self.rotation_prob = rotation_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.hue_prob = hue_prob
        self.perspective_prob = perspective_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.crop_prob = crop_prob # Yeni
        self.neg_bg_ratio = neg_bg_ratio
        self.neg_bg_files = []
        if self.transform and neg_bg_folder and os.path.exists(neg_bg_folder):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.JPEG']:
                self.neg_bg_files.extend(glob.glob(os.path.join(neg_bg_folder, ext)))
            print(f"Loaded {len(self.neg_bg_files)} negative background images to counter false detections.")
        
    def __len__(self):
        return len(self.img_id)

    def _load_image_and_annotations_old(self, idx):
        """Helper function to load a single image and its annotations."""
        json_file_path = self.img_id[idx]
        with open(json_file_path, "r") as f:
            jsonfile = json.load(f)

        image_path = os.path.join(os.path.dirname(json_file_path), jsonfile["imagePath"])
        

        
        # Handle potential variations in image file extensions
        if not os.path.isfile(image_path):
            base, _ = os.path.splitext(image_path)
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
                if os.path.isfile(base + ext):
                    image_path = base + ext
                    break

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_w, img_h = img.shape[1], img.shape[0]

        json_file = os.path.splitext(image_path)[0] + ".json"
        target = []
        if os.path.isfile(json_file):
            target = process_json_raw(json_file)
            
        return img, target, img_w, img_h

    def _load_image_and_annotations(self, idx):
        """Helper function to load a single image and its annotations safely."""
        json_file_path = self.img_id[idx]
        
        base_path = os.path.splitext(json_file_path)[0]
        image_path = None
        
        # 1. Check for the matching image with common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
            if os.path.isfile(base_path + ext):
                image_path = base_path + ext
                break
                
        # 2. Fallback to reading the JSON's internal path if base path fails
        if image_path is None:
            try:
                with open(json_file_path, "r") as f:
                    jsonfile = json.load(f)
                fallback_path = os.path.join(os.path.dirname(json_file_path), jsonfile.get("imagePath", ""))
                if os.path.isfile(fallback_path):
                    image_path = fallback_path
            except Exception:
                pass # If JSON is unreadable, let it fall through to the safety net below

        # --- THE SAFETY NET: Catch missing/corrupted files and replace on the fly ---
        if image_path is None or not os.path.isfile(image_path):
            print(f"\n[Warning] Missing image for: {json_file_path} - Skipping and picking random substitute.")
            new_idx = random.randint(0, len(self.img_id) - 1)
            return self._load_image_and_annotations(new_idx)

        img = cv2.imread(image_path)
        if img is None:
            print(f"\n[Warning] Unreadable or corrupted image: {image_path} - Skipping and picking random substitute.")
            new_idx = random.randint(0, len(self.img_id) - 1)
            return self._load_image_and_annotations(new_idx)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_w, img_h = img.shape[1], img.shape[0]

        # 3. Catch corrupted JSONs so they don't crash training either
        try:
            target = process_json_raw(json_file_path) 
        except Exception as e:
            print(f"\n[Warning] Broken JSON annotation: {json_file_path} (Error: {e}) - Skipping.")
            new_idx = random.randint(0, len(self.img_id) - 1)
            return self._load_image_and_annotations(new_idx)
            
        return img, target, img_w, img_h

    def __getitem__(self, idx):
        # 1. Check if we should serve a negative background sample
        if self.transform and self.neg_bg_files and random.random() < self.neg_bg_ratio:
            bg_path = random.choice(self.neg_bg_files)
            img = cv2.imread(bg_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                target = []  # No annotations -> pure negative sample!
                
                # Apply strictly pixel-level augmentations (No geometric warps needed)
                if random.random() < self.brightness_contrast_prob:
                    img = _adjust_brightness_contrast(img)
                if random.random() < self.gaussian_noise_prob:
                    img = _add_gaussian_noise(img)
                if random.random() < self.hue_prob:
                    img = _adjust_hue(img)
                    
                # Skip to final processing block
                img = self.normalize(img)
                img = img.transpose([2, 0, 1])
                hm, offset, regr, mask, hm_corners, corner_offset, corner_offset_mask = make_hm_offset_regr_angle(target)
                return img, hm, offset, regr, mask, hm_corners, corner_offset, corner_offset_mask, target
        # --- Mosaic Augmentation ---
        if self.transform and random.random() < self.mosaic_prob:
            # 1. Load the primary image and annotations
            img1, target1, _, _ = self._load_image_and_annotations(idx)
            
            # 2. Get 3 more random samples from the dataset
            other_indices = [random.randint(0, len(self.img_id) - 1) for _ in range(3)]
            other_images = [self._load_image_and_annotations(i)[0] for i in other_indices]
            other_targets = [self._load_image_and_annotations(i)[1] for i in other_indices]
            
            # 3. Combine them into a single mosaic image and annotation list
            all_images = [img1] + other_images
            all_targets = [target1] + other_targets
            img, target = create_mosaic_image(all_images, all_targets, target_size=(INPUT_WIDTH, INPUT_HEIGHT))

        else:
            # --- Standard Loading ---
            img, target, img_w, img_h = self._load_image_and_annotations(idx)
            
            # --- YENİ EKLENEN KISIM: Random Crop ---
            # Mosaic yapılmadıysa crop yapma ihtimalini değerlendir.
            # Crop yaparsak zaten resize içinde olduğu için tekrar resize yapmaya gerek yok.
            # if self.transform and random.random() < self.crop_prob:
            #     pass
                # img, target = _random_resized_crop(img, target, 
                #                                    target_size=(INPUT_WIDTH, INPUT_HEIGHT),
                #                                    scale_range=(0.4, 1.0)) # Görüntünün %40'ına kadar inebilir (Zoom in)
            #else:
                # Crop yapılmadıysa normal resize
            img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
            scale_x = INPUT_WIDTH / img_w
            scale_y = INPUT_HEIGHT / img_h
            for anno in target:
                anno[0] *= scale_x
                anno[1] *= scale_y
                for i in range(2, 6):
                    anno[i][0] *= scale_x
                    anno[i][1] *= scale_y

        # --- Other Data Augmentations (applied after mosaic or resize) ---
        if self.transform:
            if random.random() < self.rotation_prob:
                img, target = rotate_and_fit_to_size(img, target, target_size=(INPUT_WIDTH, INPUT_HEIGHT), angle_range=(-270, 270))
            if random.random() < self.brightness_contrast_prob:
                img = _adjust_brightness_contrast(img)
            if random.random() < self.gaussian_noise_prob:
                img = _add_gaussian_noise(img)
            if random.random() < self.hue_prob:
                img = _adjust_hue(img)
            if len(target) > 0 and random.random() < self.perspective_prob:
                img, target = _apply_perspective_transform(img, target, perspective_magnitude=0.07)
            # if len(target) > 0 and random.random() < self.horizontal_flip_prob:
            #     img, target = _horizontal_flip(img, target)

        # Final processing
        img = self.normalize(img)
        img = img.transpose([2, 0, 1])
        hm, offset, regr, mask, hm_corners, corner_offset,corner_offset_mask = make_hm_offset_regr_angle(target)
        
        return img, hm, offset, regr, mask, hm_corners, corner_offset, corner_offset_mask, target



dataset_folder = glob.glob(f"{DATASET_PATH}/*.json")
train_id, test_id = train_test_split(dataset_folder, test_size=0.05, random_state=777)

with open("test_id.txt", "w") as file:
    # Joins items with a newline character
    file.write("\n".join(test_id))

traindataset = DetectionDataset(train_id, transform=True)
valdataset = DetectionDataset(test_id, transform=False)

def collate_fn(batch):
    """Hem Train hem Val DataLoader için target dahil Collate Function"""
    imgs, hms, offsets, regrs, masks, hm_corners_list, corner_offsets, corner_offset_masks, targets = zip(*batch)
    return torch.stack([torch.from_numpy(i) for i in imgs]), \
           torch.stack([torch.from_numpy(i) for i in hms]), \
           torch.stack([torch.from_numpy(i) for i in offsets]), \
           torch.stack([torch.from_numpy(i) for i in regrs]), \
           torch.stack([torch.from_numpy(i) for i in masks]), \
           torch.stack([torch.from_numpy(i) for i in hm_corners_list]), \
           torch.stack([torch.from_numpy(i) for i in corner_offsets]), \
           torch.stack([torch.from_numpy(i) for i in corner_offset_masks]), \
           targets

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(valdataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, collate_fn=collate_fn)

# ==========================================
# 5. KAYIP (LOSS) FONKSİYONLARI
# ==========================================
def focal_loss(pred_mask, gt, gamma=2):
    gt = gt.unsqueeze(1).float()
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    pos_loss = torch.log(pred_mask) * torch.pow(1 - pred_mask, 2) * pos_inds
    neg_loss = torch.log(1 - pred_mask) * torch.pow(pred_mask, 2) * neg_weights * neg_inds
    num_pos = pos_inds.float().sum()
    if num_pos == 0: return -neg_loss.sum(), 0.0, neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos, pos_loss.sum(), neg_loss.sum()

def wing_loss(pred, target, mask, w=10.0, epsilon=2.0):
    diff = target - pred
    abs_diff = torch.abs(diff)
    c = w - w * math.log(1 + w / epsilon)
    losses = torch.where(abs_diff < w, w * torch.log(1 + abs_diff / epsilon), abs_diff - c)
    losses = losses * mask
    num_pos = mask.float().sum()
    return losses.sum() / (num_pos + 1e-4)

def global_loss(hm_pred, hm_gt, off_pred, off_gt, wh_coarse_pred, wh_fine_pred, wh_gt, corner_off_pred, corner_off_gt, mask, hm_corners_pred, hm_corners_gt, corner_offset_mask):
    pred_mask = torch.clamp(torch.sigmoid(hm_pred), 1e-4, 1 - 1e-4).unsqueeze(1).float()
    pred_mask_corners = torch.clamp(torch.sigmoid(hm_corners_pred), 1e-4, 1 - 1e-4).unsqueeze(1).float()

    foc_loss, pos_loss, neg_loss = focal_loss(pred_mask, hm_gt)
    foc_loss_corners, _, _ = focal_loss(pred_mask_corners, hm_corners_gt)

    mask_ = mask.clone()
    mask_2 = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2).permute(0, 3, 1, 2)
    mask_8 = torch.unsqueeze(mask_, -1).repeat(1, 1, 1, 8).permute(0, 3, 1, 2)
    
    num = mask_2.float().sum()
    off_loss = nn.functional.smooth_l1_loss(off_pred * mask_2, off_gt * mask_2, reduction='sum') / (num + 1e-4)
    
    wh_loss_coarse = 0.1 * nn.functional.smooth_l1_loss(wh_coarse_pred * mask_8, wh_gt * mask_8, reduction='sum') / (mask_8.float().sum() + 1e-4)
    wh_loss_fine = 0.1 * wing_loss(wh_fine_pred * mask_8, wh_gt * mask_8, mask_8)
    wh_loss = wh_loss_coarse + wh_loss_fine

    if corner_offset_mask.sum() > 0:
        corner_off_loss = 0.1 * wing_loss(corner_off_pred * corner_offset_mask, corner_off_gt * corner_offset_mask, corner_offset_mask, w=3, epsilon=0.5)
    else:
        corner_off_loss = torch.tensor(0.0, device=DEVICE)

    return foc_loss, off_loss, wh_loss, foc_loss_corners, corner_off_loss

# ==========================================
# 6. INFERENCE VE DEĞERLENDİRME (EVALUATION)
# ==========================================
def pred2box(hm, offset, regr, dense_corner_offset, thresh=0.5):
    pred = hm > thresh
    pred_center = np.asarray(np.where(hm > thresh)).T
    pred_r = regr[:, pred].T
    boxes, scores = [], hm[pred]
    
    H, W = hm.shape 
    for (center, b) in zip(pred_center, pred_r):
        offset_xy = offset[:, center[0], center[1]]
        cx_fm, cy_fm = center[1] + offset_xy[0], center[0] + offset_xy[1]
        
        arr_list = []
        for i in range(4):
            corner_x_fm, corner_y_fm = cx_fm + b[i*2], cy_fm + b[i*2 + 1]
            grid_x, grid_y = int(np.clip(round(corner_x_fm), 0, W - 1)), int(np.clip(round(corner_y_fm), 0, H - 1))
            dense_dx, dense_dy = dense_corner_offset[i*2, grid_y, grid_x], dense_corner_offset[i*2 + 1, grid_y, grid_x]
            
            final_corner_x = (grid_x + dense_dx) * MODEL_SCALE
            final_corner_y = (grid_y + dense_dy) * MODEL_SCALE
            arr_list.extend([final_corner_x, final_corner_y])
        boxes.append(np.array(arr_list))
    return np.asarray(boxes), scores

def evaluate_validation(model, val_loader, device, iou_threshold=0.5):
    """
    Hem Validation Loss hesaplar hem de tahminleri Shapely IoU ile değerlendirir.
    Bunu tek bir döngüde yaparak performansı artırır.
    """
    model.eval()
    val_loss = 0.0
    all_tp, all_fp, all_fn, total_iou, matched_count = 0, 0, 0, 0.0, 0
    
    with torch.no_grad():
        for imgs, hms, offsets, regrs, masks, hm_corners_list, corner_offsets, corner_offset_masks, targets in val_loader:
            imgs = imgs.to(device)
            hms, offsets, regrs = hms.to(device), offsets.to(device), regrs.to(device)
            masks, hm_corners_list = masks.to(device), hm_corners_list.to(device)
            corner_offsets, corner_offset_masks = corner_offsets.to(device), corner_offset_masks.to(device)

            # 1. Model Forward Pass
            preds_hm, preds_offset, preds_wh_coarse, preds_wh_fine, preds_hm_corner, preds_corner_offset = model(imgs)

            # 2. Validation Loss Hesaplama
            foc_loss, off_loss, wh_loss, foc_loss_corners, corner_off_loss = global_loss(
                preds_hm, hms, preds_offset, offsets, preds_wh_coarse, preds_wh_fine, regrs,
                preds_corner_offset, corner_offsets, masks, preds_hm_corner, hm_corners_list, corner_offset_masks
            )
            batch_loss = foc_loss + off_loss + wh_loss + foc_loss_corners + corner_off_loss
            val_loss += batch_loss.item()

            # 3. IoU ve F1 Metrikleri için Tahminleri Ayrıştırma
            for b in range(imgs.size(0)):
                hm_b = torch.sigmoid(preds_hm[b]).cpu().numpy()
                offset_b = preds_offset[b].cpu().numpy()
                regr_b = preds_wh_fine[b].cpu().numpy()
                dense_corner_b = preds_corner_offset[b].cpu().numpy()
                
                preds = []
                for cls_idx in range(hm_b.shape[0]):
                    boxes, scores = pred2box(hm_b[cls_idx], offset_b, regr_b, dense_corner_b, thresh=0.3)
                    for box in boxes:
                        points = [[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]]
                        preds.append({"label": CLASSES[cls_idx], "points": points})
                
                # Ground Truth'ları hazırlama (targets listesinden alınıyor)
                gts = targets[b]
                gt_dicts = [{"label": CLASSES[gt[6]], "points": [gt[2], gt[3], gt[4], gt[5]]} for gt in gts]
                
                # 4. Greedy Matching (IoU Hesaplama)
                iou_matrix = np.zeros((len(preds), len(gt_dicts)))
                for pi, pred in enumerate(preds):
                    for gi, gt in enumerate(gt_dicts):
                        if pred["label"] == gt["label"]:
                            iou_matrix[pi, gi] = poly_iou(pred["points"], gt["points"])
                
                matched_pred, matched_gt = set(), set()
                while iou_matrix.size > 0:
                    max_iou = iou_matrix.max()
                    if max_iou < iou_threshold: break
                    pi, gi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                    
                    all_tp += 1
                    total_iou += max_iou
                    matched_count += 1
                    matched_pred.add(pi); matched_gt.add(gi)
                    iou_matrix[pi, :] = -1; iou_matrix[:, gi] = -1

                all_fp += len(preds) - len(matched_pred)
                all_fn += len(gt_dicts) - len(matched_gt)

    # Metrik Sonuçları
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / matched_count if matched_count > 0 else 0.0
    avg_val_loss = val_loss / len(val_loader)
    
    return avg_val_loss, f1, mean_iou

# ==========================================
# 7. EĞİTİM DÖNGÜSÜ
# ==========================================
model = get_pose_net(85, {"hm": len(CLASSES), "offset": 2, "wh": 8, "corners": 4, "corner_offset": len(CLASSES) * 8})
model.to(DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

# scheduler = optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=1e-3,
#     steps_per_epoch=len(train_loader),
#     epochs=EPOCHS,
#     pct_start=0.1,        # warm up for 10% of training
#     anneal_strategy='cos'
# )


optimizer = optim.Adam([
    {
        'params': model.base.parameters(),
        'lr': 1e-4,
        'name': 'backbone'
    },
    {
        'params': list(model.denseBlocksUp.parameters()) +
                  list(model.transUpBlocks.parameters()) +
                  list(model.conv1x1_up.parameters()) +
                  list(model.last_blk.parameters()) +
                  list(model.last_proj.parameters()),
        'lr': 1e-4,
        'name': 'decoder'
    },
    {
        'params': list(model.hm.parameters()) +
                  list(model.offset.parameters()),
        'lr': 1e-4,
        'name': 'hm_offset_heads'
    },
    {
        'params': list(model.wh.parameters()),
        'lr': 1e-4,
        'name': 'wh_coarse_head'
    },
    {
        'params': model.wh_refinement.parameters(),
        'lr': 1e-4,
        'name': 'wh_refinement'
    },
    {
        'params': list(model.corner_heads.parameters()) +
                  list(model.corner_offset.parameters()),
        'lr': 1e-4,
        'name': 'corner_heads'
    },
],
    weight_decay=0
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[
        3e-4,   # backbone      — pretrained, keep conservative
        5e-4,   # decoder       — trained from scratch but feeds everything
        8e-4,   # hm/offset     — simple heads, can move faster
        8e-4,   # wh_coarse     — regression head
        5e-4,   # wh_refinement — new module, needs stable coarse signal first
        5e-4,   # corner_heads  — new RotInvCornerHead, also fresh
    ],
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    anneal_strategy='cos'
)

best_f1 = 0.0
best_miou = 0.0
best_loss = np.inf
train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    t = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for imgs, hms, offsets, regrs, masks, hm_corners_list, corner_offsets, corner_offset_masks, _ in t:
        imgs = imgs.to(DEVICE)
        hms, offsets, regrs = hms.to(DEVICE), offsets.to(DEVICE), regrs.to(DEVICE)
        masks, hm_corners_list = masks.to(DEVICE), hm_corners_list.to(DEVICE)
        corner_offsets, corner_offset_masks = corner_offsets.to(DEVICE), corner_offset_masks.to(DEVICE)
        
        optimizer.zero_grad()
        preds_hm, preds_offset, preds_wh_coarse, preds_wh_fine, preds_hm_corner, preds_corner_offset = model(imgs)
        
        foc_loss, off_loss, wh_loss, foc_loss_corners, corner_off_loss = global_loss(
            preds_hm, hms, preds_offset, offsets, preds_wh_coarse, preds_wh_fine, regrs,
            preds_corner_offset, corner_offsets, masks, preds_hm_corner, hm_corners_list, corner_offset_masks
        )
        
        loss = foc_loss + off_loss + wh_loss + foc_loss_corners + corner_off_loss
        
        if torch.isnan(loss):
            print("\nUyarı: NaN loss saptandı, bu batch atlanıyor.")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        t.set_description(f"Epoch {epoch+1}/{EPOCHS} (Train Loss: {train_loss / (t.n + 1):.3f})")

    avg_train_loss = train_loss / len(train_loader)
    train_loss_history.append(avg_train_loss)

    # ----------------------------------------------------
    # Her Epoch sonunda Validation ve IoU Değerlendirmesi
    # ----------------------------------------------------
    print(f"\nEpoch {epoch+1} değerlendiriliyor...")
    val_loss, f1_score, miou = evaluate_validation(model, val_loader, DEVICE, IOU_THRESHOLD)
    val_loss_history.append(val_loss)
    
    print(f"-> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"-> F1 Score: {f1_score:.4f} | mIoU: {miou:.4f}")
    
    # En iyi F1 Skoruna sahip modeli kaydet (İstersen val_loss tabanlı da kaydedebilirsin)
    if  miou >= best_miou:
        best_miou = miou
        save_path = os.path.join(MODEL_SAVE_FOLDER, f"best_centernet_{INPUT_WIDTH}_miou_{best_miou:.3f}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"YENİ EN İYİ MODEL KAYDEDİLDİ (mIoU): {save_path}\n")
    if  val_loss < best_loss:
        best_loss = val_loss
        save_path = os.path.join(MODEL_SAVE_FOLDER, f"best_centernet_{INPUT_WIDTH}_val_loss_{best_loss:.3f}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"YENİ EN İYİ MODEL KAYDEDİLDİ (Val Loss): {save_path}\n")
    else:
        print("\n")

# Eğitim bitince loss grafiğini göster
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()