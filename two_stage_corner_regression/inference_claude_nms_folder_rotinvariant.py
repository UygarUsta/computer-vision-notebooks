import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torchvision
from torchvision import transforms
from functools import partial
from glob import glob
import json
import torch.nn.functional as F
assert torch.cuda.is_available(), "CUDA not available."

from cornerpool_deneme_offset_rotinvariant import get_pose_net

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
INPUT_WIDTH   = 512
INPUT_HEIGHT  = 512
MODEL_SCALE   = 4
THRESHOLD     = 0.3
NMS_IOU_THR   = 0.45      # box-level polygon IoU suppression threshold
N_REFINE      = 2         # how many iterative passes for Stage 2 (set 1 = original behaviour)
DISPLAY_W     = 1024       # each panel width in the side-by-side window
DISPLAY_H     = 720

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "modelWeights/best_centernet_512_val_loss_1.694.pth"

if not os.path.isfile("classes.txt"):
    CLASSES = ['T', 'azami_yuk', 'kullanim_amaci', 'menfaat', 'net_agirlik', 'plaka', 'qr', 'romork_azami_yuk', 'ruhsat', 'seri_no', 'tc']
else:
    with open("classes.txt","r") as f:
        CLASSES = f.readlines()

# Per-class colours for the final bounding box outline (BGR)
CLASS_COLORS = [
    (255, 128,   0),   # dogum_tarih   – orange
    (  0, 255, 255),   # ehliyet_arka  – yellow
    (255,   0, 255),   # ehliyet_on    – magenta
    (  0, 255,   0),   # kisitlama     – green
    (255,   0,   0),   # kose          – blue
    (  0, 128, 255),   # qr_ehliyet    – light-blue
    (128,   0, 255),   # tc            – purple
]

# ── Stage visualisation colours ──────────────────────────────────────────────
#
#  LEFT panel  (original, single-pass)
#    • Stage-1 coarse corner  : RED    filled circle
#    • Grid snap point        : BLUE   filled square
#    • Gravity arrow (1 pass) : WHITE  arrowed line
#    • Final corner           : PINK   filled circle
#    • Bounding box           : class colour
#
#  RIGHT panel (iterative, N_REFINE passes)
#    • Stage-1 coarse corner  : RED    filled circle   (same as left)
#    • Grid snap 1            : BLUE   filled square
#    • Arrow pass-1           : YELLOW arrowed line
#    • Refined-1 position     : CYAN   hollow circle
#    • Grid snap 2            : GREEN  filled square
#    • Arrow pass-2           : ORANGE arrowed line
#    • Final corner           : PINK   filled circle
#    • Bounding box           : class colour (brighter / thicker)

COL_COARSE   = (  0,   0, 255)   # red    – stage-1 coarse
COL_GRID1    = (255,   0,   0)   # blue   – first grid snap
COL_ARROW1   = (255, 255, 255)   # white  – single-pass gravity arrow
COL_ARROW_P1 = (  0, 255, 255)   # yellow – pass-1 arrow (iterative)
COL_MID      = (255, 255,   0)   # cyan   – intermediate (after pass-1)
COL_GRID2    = (  0, 255,   0)   # green  – second grid snap
COL_ARROW_P2 = (  0, 165, 255)   # orange – pass-2 arrow (iterative)
COL_FINAL    = (255,   0, 255)   # pink   – final refined corner


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
model = get_pose_net(
    85,
    {"hm": len(CLASSES), "offset": 2, "wh": 8,
     "corners": 4, "corner_offset": len(CLASSES) * 8}
)
model.load_state_dict(torch.load(model_name))
model.to(DEVICE)
model.eval()


# ─────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────
class Normalize:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image):
        img = image.astype(np.float32) / 255.0
        img -= self.mean
        img /= self.std
        return img

normalize = Normalize()


# ─────────────────────────────────────────
# NMS HELPERS
# ─────────────────────────────────────────

# ── 1. Heatmap peak NMS (max-pool, replaces O(N²) select) ───────────────────
def heatmap_nms(hm: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    Fast local-maximum suppression via max-pooling.

    For every pixel that is NOT the local maximum within a (kernel×kernel)
    neighbourhood the value is zeroed out.  This is identical in spirit to the
    original `select()` but runs in O(H·W) instead of O(N²) and is the same
    approach used in CenterNet / CornerNet papers.

    Args:
        hm     : 2-D numpy array [H, W], already sigmoid-activated.
        kernel : suppression radius; must be odd.  Default 5 ≈ 2-pixel radius.

    Returns:
        hm_nms : same shape, non-maxima zeroed.
    """
    assert kernel % 2 == 1, "kernel must be odd"
    t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
    pad    = kernel // 2
    hmax   = F.max_pool2d(t, kernel_size=kernel, stride=1, padding=pad)
    keep   = (t == hmax).float()                                 # 1 where local max
    return (t * keep).squeeze().numpy()


def select(hm: np.ndarray, threshold: float) -> np.ndarray:
    """
    Drop-in replacement for the original select():
    max-pool peak NMS → threshold → return.
    """
    hm = heatmap_nms(hm, kernel=5)
    hm[hm <= threshold] = 0
    return hm


def select_4corners(hm_channel: np.ndarray, threshold: float) -> list:
    """NMS for the auxiliary corner heatmap channels (unchanged logic, kept for
    compatibility; benefits from heatmap already being peak-suppressed upstream)."""
    processed    = heatmap_nms(hm_channel.copy(), kernel=5)
    pred_centers = np.argwhere(processed > threshold)
    if len(pred_centers) == 0:
        return []

    scores = processed[pred_centers[:, 0], pred_centers[:, 1]]
    order  = np.argsort(scores)[::-1]
    pred_centers = pred_centers[order]

    selected   = []
    suppressed = np.zeros(len(pred_centers), dtype=bool)
    for i in range(len(pred_centers)):
        if suppressed[i]:
            continue
        ci = pred_centers[i]
        selected.append({'x': ci[1], 'y': ci[0], 'score': processed[ci[0], ci[1]]})
        for j in range(i + 1, len(pred_centers)):
            if not suppressed[j] and np.linalg.norm(ci - pred_centers[j]) <= 2:
                suppressed[j] = True
    return selected


# ── 2. Polygon IoU (Sutherland-Hodgman clip, exact for convex quads) ─────────
def _clip_polygon_by_halfplane(poly, p1, p2):
    """Sutherland-Hodgman clip of `poly` against the half-plane left of p1→p2."""
    def inside(p):
        return (p2[0]-p1[0])*(p[1]-p1[1]) - (p2[1]-p1[1])*(p[0]-p1[0]) >= 0

    def intersect(a, b, c, d):
        A1, B1 = b[1]-a[1], a[0]-b[0]
        C1 = A1*a[0] + B1*a[1]
        A2, B2 = d[1]-c[1], c[0]-d[0]
        C2 = A2*c[0] + B2*c[1]
        det = A1*B2 - A2*B1
        if abs(det) < 1e-10:
            return a
        return ((C1*B2 - C2*B1)/det, (A1*C2 - A2*C1)/det)

    if not poly:
        return []
    output = []
    for i in range(len(poly)):
        curr, prev = poly[i], poly[i-1]
        if inside(curr):
            if not inside(prev):
                output.append(intersect(prev, curr, p1, p2))
            output.append(curr)
        elif inside(prev):
            output.append(intersect(prev, curr, p1, p2))
    return output


def _polygon_area(pts):
    """Signed shoelace area of a polygon given as list of (x,y) tuples."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1]
            for i in range(n))
    return abs(s) * 0.5


def polygon_iou(quad_a, quad_b):
    """
    Exact IoU between two convex quadrilaterals.

    Each quad is a list/tuple of 4 (x, y) points in order (tl, tr, br, bl or
    any consistent winding).

    Returns a float in [0, 1].
    """
    poly = list(quad_a)
    clip = list(quad_b)
    n    = len(clip)
    for i in range(n):
        poly = _clip_polygon_by_halfplane(poly, clip[i], clip[(i+1) % n])
        if not poly:
            return 0.0

    inter = _polygon_area(poly)
    union = _polygon_area(quad_a) + _polygon_area(quad_b) - inter
    return inter / (union + 1e-7)


# ── 3. Box-level quad NMS ────────────────────────────────────────────────────
def quad_nms(detections: list, iou_threshold: float = 0.45) -> list:
    """
    Non-maximum suppression on a list of detection dicts (output of pred2box_*).

    Suppression is based on polygon IoU of the *final* 4-corner quadrilateral.
    Detections are sorted by score descending; any detection whose IoU with an
    already-kept detection exceeds `iou_threshold` is discarded.

    Args:
        detections    : list of dicts with keys 'score' and 'final'
                        ('final' is a list of 4 (x,y) image-space points).
        iou_threshold : suppress boxes with IoU > this value.  Default 0.45.

    Returns:
        Filtered list of detection dicts.
    """
    if not detections:
        return []

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d['score'], reverse=True)

    kept = []
    for det in dets:
        quad_new = det['final']            # list of 4 (x, y)
        suppress = False
        for kept_det in kept:
            iou = polygon_iou(quad_new, kept_det['final'])
            if iou > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(det)
    return kept


# ─────────────────────────────────────────
# PRED2BOX – SINGLE PASS (original logic)
# ─────────────────────────────────────────
def pred2box_single(hm, offset, regr, dense_corner_offset, thresh=0.5,
                    nms_iou_thr=None):
    """
    Original single-pass refinement + box-level quad NMS.

    Returns list of dicts, each containing:
        cx, cy          – image-space centre
        stage1 [4×2]    – coarse corners from wh_fine (image space)
        grid1  [4×2]    – first (only) grid snap points (image space)
        final  [4×2]    – after one dense-offset correction (image space)
    """
    if nms_iou_thr is None:
        nms_iou_thr = NMS_IOU_THR

    pred        = hm > thresh
    pred_center = np.asarray(np.where(pred)).T        # (N,2) – (y,x)
    pred_r      = regr[:, pred].T                     # (N,8)
    H, W        = hm.shape
    results     = []

    for center, b in zip(pred_center, pred_r):
        offset_xy = offset[:, center[0], center[1]]
        cx_fm = center[1] + offset_xy[0]
        cy_fm = center[0] + offset_xy[1]

        stage1, grid1, final = [], [], []

        for i in range(4):
            # ── Stage 1: coarse corner from regression head ──
            corner_x_fm = cx_fm + b[i * 2]
            corner_y_fm = cy_fm + b[i * 2 + 1]

            # ── Snap to nearest grid cell ──
            gx = int(np.clip(round(corner_x_fm), 0, W - 1))
            gy = int(np.clip(round(corner_y_fm), 0, H - 1))

            # ── Stage 2 (single pass): read dense offset ──
            dx = dense_corner_offset[i * 2,     gy, gx]
            dy = dense_corner_offset[i * 2 + 1, gy, gx]

            final_x = (gx + dx) * MODEL_SCALE
            final_y = (gy + dy) * MODEL_SCALE

            stage1.append((corner_x_fm * MODEL_SCALE, corner_y_fm * MODEL_SCALE))
            grid1.append( (gx          * MODEL_SCALE, gy           * MODEL_SCALE))
            final.append( (final_x,                   final_y))

        results.append({
            'cx':     cx_fm * MODEL_SCALE,
            'cy':     cy_fm * MODEL_SCALE,
            'score':  hm[center[0], center[1]],
            'stage1': stage1,
            'grid1':  grid1,
            'final':  final,
        })

    # ── Box-level NMS on final quadrilaterals ────────────────────────────────
    results = quad_nms(results, iou_threshold=nms_iou_thr)
    return results


# ─────────────────────────────────────────
# PRED2BOX – ITERATIVE (N_REFINE passes)
# ─────────────────────────────────────────
def pred2box_iterative(hm, offset, regr, dense_corner_offset, thresh=0.5,
                       n_refine=2, nms_iou_thr=None):
    """
    Iterative refinement + box-level quad NMS.

    Returns list of dicts, each containing:
        cx, cy          – image-space centre
        stage1 [4×2]    – coarse corner from wh_fine (image space)
        grids  [N][4×2] – grid snap at each pass (image space)
        mids   [N-1][4×2] – intermediate positions after each pass except last
        final  [4×2]    – position after all passes (image space)
    """
    if nms_iou_thr is None:
        nms_iou_thr = NMS_IOU_THR
    pred        = hm > thresh
    pred_center = np.asarray(np.where(pred)).T
    pred_r      = regr[:, pred].T
    H, W        = hm.shape
    results     = []

    for center, b in zip(pred_center, pred_r):
        offset_xy = offset[:, center[0], center[1]]
        cx_fm = center[1] + offset_xy[0]
        cy_fm = center[0] + offset_xy[1]

        stage1 = []
        grids  = [[] for _ in range(n_refine)]   # snap at each pass
        mids   = [[] for _ in range(n_refine)]    # position after each pass
        final  = []

        for i in range(4):
            # ── Stage 1: coarse corner ──
            corner_x_fm = cx_fm + b[i * 2]
            corner_y_fm = cy_fm + b[i * 2 + 1]
            stage1.append((corner_x_fm * MODEL_SCALE, corner_y_fm * MODEL_SCALE))

            # ── Iterative Stage 2 ──
            iter_x, iter_y = corner_x_fm, corner_y_fm

            for p in range(n_refine):
                gx = int(np.clip(round(iter_x), 0, W - 1))
                gy = int(np.clip(round(iter_y), 0, H - 1))

                dx = dense_corner_offset[i * 2,     gy, gx]
                dy = dense_corner_offset[i * 2 + 1, gy, gx]

                grids[p].append((gx * MODEL_SCALE, gy * MODEL_SCALE))

                iter_x = gx + dx
                iter_y = gy + dy

                mids[p].append((iter_x * MODEL_SCALE, iter_y * MODEL_SCALE))

            final.append((iter_x * MODEL_SCALE, iter_y * MODEL_SCALE))

        results.append({
            'cx':     cx_fm * MODEL_SCALE,
            'cy':     cy_fm * MODEL_SCALE,
            'score':  hm[center[0], center[1]],
            'stage1': stage1,
            'grids':  grids,
            'mids':   mids,
            'final':  final,
        })

    # ── Box-level NMS on final quadrilaterals ────────────────────────────────
    results = quad_nms(results, iou_threshold=nms_iou_thr)
    return results


# ─────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────
def _pt(coord, sfx, sfy):
    """Scale a feature-map-space point to display-image space."""
    return (int(coord[0] * sfx), int(coord[1] * sfy))


def _arrow(img, p1, p2, color, thickness=1, tip=0.25):
    """Draw arrow only when the two endpoints are meaningfully apart."""
    if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) > 2:
        cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=tip)


def _sq(img, pt, color, half=3):
    """Small filled square – used for grid snap markers."""
    x, y = pt
    cv2.rectangle(img, (x - half, y - half), (x + half, y + half), color, -1)


def draw_legend(img, is_iterative):
    """Burn a small legend into the top-left corner."""
    entries = []
    if is_iterative:
        entries = [
            (COL_COARSE,   "Stage-1 coarse"),
            (COL_GRID1,    "Grid snap 1"),
            (COL_ARROW_P1, "Pass-1 arrow"),
            (COL_MID,      "After pass 1"),
            (COL_GRID2,    "Grid snap 2"),
            (COL_ARROW_P2, "Pass-2 arrow"),
            (COL_FINAL,    "Final corner"),
        ]
        title = f"ITERATIVE  (N={N_REFINE} passes)"
    else:
        entries = [
            (COL_COARSE,  "Stage-1 coarse"),
            (COL_GRID1,   "Grid snap"),
            (COL_ARROW1,  "Gravity arrow"),
            (COL_FINAL,   "Final corner"),
        ]
        title = "ORIGINAL  (single pass)"

    x0, y0 = 12, 12
    pad = 4
    lh  = 20
    box_h = lh * (len(entries) + 1) + pad * 2
    box_w = 230

    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - pad),
                  (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(img, title, (x0, y0 + lh - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    for k, (col, label) in enumerate(entries):
        y = y0 + lh * (k + 2)
        cv2.circle(img, (x0 + 6, y - 5), 5, col, -1)
        cv2.putText(img, label, (x0 + 16, y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────
# SHOWBOX – LEFT  (single pass)
# ─────────────────────────────────────────
def showbox_single(img, hm_all, offset, regr, corner_offset_all,
                   img_w, img_h, thresh):
    """
    Draw single-pass detections onto img (in-place) and return it.

    Visual layers (back to front):
      1. Bounding box outline  – class colour
      2. Coarse corner         – red filled circle  (Stage 1)
      3. Grid snap             – blue filled square
      4. Gravity arrow         – white arrowed line
      5. Final corner          – pink filled circle
    """
    # scale from input-space (512×512) to original image space
    sfx = img_w / INPUT_WIDTH
    sfy = img_h / INPUT_HEIGHT

    for cls_idx in range(hm_all.shape[0]):
        cls_color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        cls_corner_offset = corner_offset_all[cls_idx * 8: (cls_idx + 1) * 8]

        detections = pred2box_single(
            hm_all[cls_idx], offset, regr, cls_corner_offset, thresh=thresh
        )

        for det in detections:
            corners_s1    = [_pt(p, sfx, sfy) for p in det['stage1']]
            corners_grid1 = [_pt(p, sfx, sfy) for p in det['grid1']]
            corners_final = [_pt(p, sfx, sfy) for p in det['final']]

            tl_f, tr_f, br_f, bl_f = corners_final

            # 1. Bounding box
            th = 2
            cv2.line(img, tl_f, tr_f, cls_color, th)
            cv2.line(img, tr_f, br_f, cls_color, th)
            cv2.line(img, br_f, bl_f, cls_color, th)
            cv2.line(img, bl_f, tl_f, cls_color, th)

            # class label
            cx_px = int(det['cx'] * sfx)
            cy_px = int(det['cy'] * sfy)
            cv2.putText(img, CLASSES[cls_idx], (cx_px, cy_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, cls_color, 2, cv2.LINE_AA)

            for g1, s1, fn in zip(corners_grid1, corners_s1, corners_final):
                # 2. Stage-1 coarse corner – red
                cv2.circle(img, s1, 5, COL_COARSE, -1)
                # 3. Grid snap – blue square
                _sq(img, g1, COL_GRID1, half=3)
                # 4. Gravity arrow grid→final – white
                _arrow(img, g1, fn, COL_ARROW1, thickness=1, tip=0.3)
                # 5. Final corner – pink
                cv2.circle(img, fn, 5, COL_FINAL, -1)

    draw_legend(img, is_iterative=False)
    return img


# ─────────────────────────────────────────
# SHOWBOX – RIGHT  (iterative)
# ─────────────────────────────────────────
def showbox_iterative(img, hm_all, offset, regr, corner_offset_all,
                      img_w, img_h, thresh, n_refine=2):
    """
    Draw iterative detections onto img (in-place) and return it.

    Visual layers (back to front):
      1. Bounding box outline  – class colour (thicker)
      2. Coarse corner         – red filled circle   (Stage 1)
      3. Grid snap pass-1      – blue filled square
      4. Arrow pass-1          – yellow
      5. Intermediate position – cyan hollow circle
      6. Grid snap pass-2      – green filled square  (only if N≥2)
      7. Arrow pass-2          – orange               (only if N≥2)
      8. Final corner          – pink filled circle
    """
    sfx = img_w / INPUT_WIDTH
    sfy = img_h / INPUT_HEIGHT

    for cls_idx in range(hm_all.shape[0]):
        cls_color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        cls_corner_offset = corner_offset_all[cls_idx * 8: (cls_idx + 1) * 8]

        detections = pred2box_iterative(
            hm_all[cls_idx], offset, regr, cls_corner_offset,
            thresh=thresh, n_refine=n_refine
        )

        for det in detections:
            corners_s1    = [_pt(p, sfx, sfy) for p in det['stage1']]
            corners_final = [_pt(p, sfx, sfy) for p in det['final']]

            # grids[pass][corner], mids[pass][corner]
            grids = [[_pt(det['grids'][p][i], sfx, sfy) for i in range(4)]
                     for p in range(n_refine)]
            mids  = [[_pt(det['mids'][p][i],  sfx, sfy) for i in range(4)]
                     for p in range(n_refine)]

            tl_f, tr_f, br_f, bl_f = corners_final

            # 1. Bounding box (thicker to distinguish from left panel)
            th = 3
            cv2.line(img, tl_f, tr_f, cls_color, th)
            cv2.line(img, tr_f, br_f, cls_color, th)
            cv2.line(img, br_f, bl_f, cls_color, th)
            cv2.line(img, bl_f, tl_f, cls_color, th)

            cx_px = int(det['cx'] * sfx)
            cy_px = int(det['cy'] * sfy)
            cv2.putText(img, CLASSES[cls_idx], (cx_px, cy_px - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, cls_color, 2, cv2.LINE_AA)

            for corner_i in range(4):
                s1 = corners_s1[corner_i]
                fn = corners_final[corner_i]

                # 2. Stage-1 coarse – red
                cv2.circle(img, s1, 5, COL_COARSE, -1)

                prev_pt = grids[0][corner_i]

                for p in range(n_refine):
                    g  = grids[p][corner_i]
                    m  = mids[p][corner_i]

                    if p == 0:
                        # 3. Grid snap 1 – blue square
                        _sq(img, g, COL_GRID1, half=3)
                        # 4. Arrow pass-1 – yellow
                        _arrow(img, g, m, COL_ARROW_P1, thickness=2, tip=0.25)
                        # 5. Intermediate – cyan hollow circle
                        if n_refine > 1:
                            cv2.circle(img, m, 4, COL_MID, 1)

                    elif p == 1:
                        # 6. Grid snap 2 – green square
                        _sq(img, g, COL_GRID2, half=3)
                        # 7. Arrow pass-2 – orange
                        _arrow(img, g, m, COL_ARROW_P2, thickness=2, tip=0.25)

                    prev_pt = m

                # 8. Final corner – pink
                cv2.circle(img, fn, 5, COL_FINAL, -1)

    draw_legend(img, is_iterative=True)
    return img


# ─────────────────────────────────────────
# MAIN INFERENCE LOOP
# ─────────────────────────────────────────
#image_folder = glob(r"C:\Users\uygar.usta\Desktop\datasets\ruhsat_detection_dataset\stripped_ruhsat\*")

#image_folder = ["/home/uygarusta/Oriented-Centernet/2stage_corner_regression/dataset/gbs/beyan/c6683595-c8b5-4f91-b7ec-74e78aeef11d.jpg"]

with open("test_id.txt","r") as f:
    image_folder = f.readlines()

for img_path in image_folder:
    img_path = img_path.replace("\n","")
    if img_path.endswith(".json"): img_path = img_path.replace(".json",".jpg")
    if not os.path.isfile(img_path):
        base, _ = os.path.splitext(img_path)
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
            if os.path.isfile(base + ext):
                img_path = base + ext
                break
    print("Image Path ->",img_path)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    img_w = img_bgr.shape[1]
    img_h = img_bgr.shape[0]

    # ── Preprocess ──────────────────────────────────────────────────────────
    img_resized = cv2.resize(img_bgr, (INPUT_WIDTH, INPUT_HEIGHT),
                             interpolation=cv2.INTER_AREA)
    img_norm    = normalize(img_resized).transpose([2, 0, 1])
    img_tensor  = torch.from_numpy(img_norm).to(DEVICE).float().unsqueeze(0)

    # ── Inference ───────────────────────────────────────────────────────────
    with torch.no_grad():
        hm_raw, offset_raw, wh_coarse, wh_fine, corners_raw, corner_offset_raw = \
            model(img_tensor)

    hm            = torch.sigmoid(hm_raw).cpu().numpy().squeeze(0)
    offset        = offset_raw.cpu().numpy().squeeze(0)
    wh            = wh_fine.cpu().numpy().squeeze(0)
    corners       = torch.sigmoid(corners_raw).cpu().numpy().squeeze(0)
    corner_offset = corner_offset_raw.cpu().numpy().squeeze(0)

    # ── NMS on centre heatmap ────────────────────────────────────────────────
    for c in range(hm.shape[0]):
        hm[c] = select(hm[c], THRESHOLD)

    # ── Build LEFT panel (single-pass) ──────────────────────────────────────
    left = img_bgr.copy()
    left = showbox_single(
        left, hm, offset, wh, corner_offset,
        img_w, img_h, THRESHOLD
    )

    # ── Build RIGHT panel (iterative) ───────────────────────────────────────
    right = img_bgr.copy()
    right = showbox_iterative(
        right, hm, offset, wh, corner_offset,
        img_w, img_h, THRESHOLD, n_refine=N_REFINE
    )

    # ── Resize both panels to the same display size ─────────────────────────
    left_disp  = cv2.resize(left,  (DISPLAY_W, DISPLAY_H))
    right_disp = cv2.resize(right, (DISPLAY_W, DISPLAY_H))

    # ── Draw divider and panel labels ────────────────────────────────────────
    divider = np.full((DISPLAY_H, 4, 3), 60, dtype=np.uint8)

    for panel, label in [(left_disp, "ORIGINAL (1 pass)"),
                         (right_disp, f"ITERATIVE ({N_REFINE} passes)")]:
        tw, th2 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = (DISPLAY_W - tw) // 2
        cv2.putText(panel, label, (tx, DISPLAY_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

    combined = np.concatenate([left_disp, divider, right_disp], axis=1)
    cv2.imwrite(f"inference_outputs/{os.path.basename(img_path)}",combined)
    #cv2.imshow("Stage comparison  |  ESC = quit  |  any key = next", combined)
    #key = cv2.waitKey(0)
    #if key == 27:   # ESC
    #    break

#cv2.destroyAllWindows()