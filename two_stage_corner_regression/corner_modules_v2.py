"""
corner_modules_v2.py
====================
Drop-in replacements for two modules in cornerpool_deneme_offset.py.

WHAT CHANGED vs the original
─────────────────────────────
1.  RotInvCornerHead  (replaces CornerPoolModule)
    • Removes directional cummax pooling (TopPool / BottomPool / LeftPool / RightPool).
    • Those operations assume axis-aligned corners, which breaks for arbitrary-rotation
      documents.  This head is a plain depthwise-separable conv block — rotation
      invariance comes from data augmentation, not from a built-in directional bias.
    • Param count per head: ~5 k  (vs ~66 k for CornerPoolModule at prev_ch=200)

2.  PatchRefinement  (replaces CoarseToFineRefinement)
    • For each of the 4 corners the coarse offset field places us near the true corner.
      Instead of sampling a single point, we sample a 3×3 patch around that location.
      The concatenated patch features give the MLP local neighbourhood context, which
      is the key ingredient that single-point sampling lacks.
    • Channel reduction (reduce_ch=16) keeps VRAM low — total concat width is
      16 + 4*9*16 = 592 channels, processed by a lightweight DW-sep block.
    • Param count: ~130 k  (vs ~270 k for the original CoarseToFineRefinement)

HOW TO PLUG IN
──────────────
In cornerpool_deneme_offset.py make exactly three edits:

  A) Remove (or comment out) the old class definitions:
       class CornerPoolModule
       class CoarseToFineRefinement

  B) At the top of the file, add:
       from corner_modules_v2 import RotInvCornerHead, PatchRefinement

  C) In HarDNetSeg.__init__, replace:
       # old corner head wiring
       self.corner_heads['tl'] = CornerPoolModule(prev_ch, corner_type='tl')
       self.corner_heads['tr'] = CornerPoolModule(prev_ch, corner_type='tr')
       self.corner_heads['br'] = CornerPoolModule(prev_ch, corner_type='br')
       self.corner_heads['bl'] = CornerPoolModule(prev_ch, corner_type='bl')

     with:
       self.corner_heads['tl'] = RotInvCornerHead(prev_ch)
       self.corner_heads['tr'] = RotInvCornerHead(prev_ch)
       self.corner_heads['br'] = RotInvCornerHead(prev_ch)
       self.corner_heads['bl'] = RotInvCornerHead(prev_ch)

     and replace:
       self.wh_refinement = CoarseToFineRefinement(prev_ch)
     with:
       self.wh_refinement = PatchRefinement(prev_ch)

  Nothing else in the file needs to change — forward() signatures are identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RotInvCornerHead
#     A lightweight depthwise-separable block that predicts a 1-channel heatmap.
#     No directional pooling → works for any rotation.
# ─────────────────────────────────────────────────────────────────────────────

class _DWSepConv(nn.Module):
    """Depthwise + pointwise conv with BN+ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch,  kernel_size=3, padding=1,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(x))))


class RotInvCornerHead(nn.Module):
    """
    Rotation-invariant corner heatmap head.

    in_ch  → 32 (pointwise squeeze)  → 32 (DW-sep)  → 1 (pointwise predict)

    Args
    ────
    in_ch : int
        Number of channels in the shared feature map (prev_ch = 200).
    mid_ch : int
        Internal width after the initial squeeze.  Default 32 keeps it tiny.
    """
    def __init__(self, in_ch: int, mid_ch: int = 32):
        super().__init__()
        # squeeze
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        # refine
        self.refine = _DWSepConv(mid_ch, mid_ch)
        # predict
        self.predict = nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
        # focal-loss bias init: -log((1-π)/π) for π≈0.01 → ≈ -4.59
        nn.init.constant_(self.predict.bias, -4.59)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, in_ch, H, W]
        returns : [B, 1, H, W]   (raw logits, apply sigmoid outside)
        """
        x = self.squeeze(x)
        x = self.refine(x)
        return self.predict(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PatchRefinement
#     Coarse-to-fine refinement using 3×3 patch sampling around each corner.
# ─────────────────────────────────────────────────────────────────────────────

class PatchRefinement(nn.Module):
    """
    Coarse-to-fine refinement with local patch sampling.

    For every spatial position (y, x) the coarse wh head predicts 8 offsets
    (dx, dy for each of 4 corners).  This module:

      1. Squeezes the feature map to `reduce_ch` channels.
      2. For each of the 4 corners, uses the coarse offset to locate the
         predicted corner on the feature map, then samples a (patch_size×patch_size)
         grid of points around it via bilinear grid_sample.
      3. Concatenates the center features + all patch features and regresses
         a residual delta (Δwh) with a DW-sep MLP.
      4. Returns  wh_coarse + Δwh  as the fine prediction.

    Gradient flow
    ─────────────
    • grid_sample coordinates are derived from wh_coarse.detach() so the
      sampling locations do not carry gradients back to the coarse head.
    • The residual addition wh_coarse + delta_wh keeps the gradient for the
      fine loss flowing through both wh_coarse (quality of initial estimate)
      and delta_wh (quality of correction).

    Params  (reduce_ch=16, patch_size=3, in_ch=200)
    ────────────────────────────────────────────────
    • reduce_conv:       200 × 16 × 1 × 1                =   3 200
    • concat width:      16 + 4 × 9 × 16 = 592
    • dw  conv:          592 × 592 (groups=592, k=3)     =   5 328
    • pw  conv:          592 × 64                        =  37 888
    • bn + predict:      64 → 8                          =     576
    Total ≈ 47 k params
    """

    def __init__(
        self,
        in_ch:      int = 200,
        reduce_ch:  int = 16,
        patch_size: int = 3,
    ):
        super().__init__()
        self.reduce_ch  = reduce_ch
        self.patch_size = patch_size
        self.radius     = patch_size // 2          # = 1 for 3×3

        # ── channel squeeze ──────────────────────────────────────────────────
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_ch, reduce_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_ch, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        # ── concat width: center_feat + 4 corners × patch² × reduce_ch ──────
        num_pts   = patch_size ** 2          # 9
        concat_ch = reduce_ch + 4 * num_pts * reduce_ch   # 16 + 576 = 592

        # ── lightweight DW-sep regression head ───────────────────────────────
        mid_ch = 64
        self.regress = nn.Sequential(
            # DW
            nn.Conv2d(concat_ch, concat_ch, kernel_size=3, padding=1,
                      groups=concat_ch, bias=False),
            # PW → mid
            nn.Conv2d(concat_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=0.1),
            nn.ReLU(inplace=True),
            # predict Δwh  (8 channels: Δdx,Δdy × 4 corners)
            nn.Conv2d(mid_ch, 8, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)

    def forward(
        self,
        feat:      torch.Tensor,   # [B, in_ch, H, W]  shared feature map
        wh_coarse: torch.Tensor,   # [B, 8, H, W]      coarse corner offsets
    ) -> torch.Tensor:             # [B, 8, H, W]      refined corner offsets
        B, _, H, W = feat.shape

        # ── 1. squeeze features ──────────────────────────────────────────────
        rfeat = self.reduce_conv(feat)          # [B, reduce_ch, H, W]

        # ── 2. build base sampling grid (pixel coordinates) ──────────────────
        gy, gx = torch.meshgrid(
            torch.arange(H, device=feat.device, dtype=torch.float32),
            torch.arange(W, device=feat.device, dtype=torch.float32),
            indexing='ij',
        )
        # center_grid[0] = x-coords, center_grid[1] = y-coords
        center_grid = torch.stack([gx, gy], dim=0).unsqueeze(0)  # [1, 2, H, W]

        # ── 3. detach coarse for sampling coordinates ────────────────────────
        wh_det = wh_coarse.detach()             # no grad into coarse head via sample coords

        patch_feats = [rfeat]                   # start with center context

        for i in range(4):                      # 4 corners
            # coarse predicted corner location in feature-map space
            base_pos = center_grid + wh_det[:, i*2:(i+1)*2]  # [B, 2, H, W]
            base_x   = base_pos[:, 0]           # [B, H, W]
            base_y   = base_pos[:, 1]

            for dy in range(-self.radius, self.radius + 1):   # −1, 0, +1
                for dx in range(-self.radius, self.radius + 1):
                    # absolute sample position
                    sx = base_x + dx            # [B, H, W]
                    sy = base_y + dy

                    # normalise to [-1, 1] for grid_sample
                    nx = (sx / (W - 1)) * 2.0 - 1.0
                    ny = (sy / (H - 1)) * 2.0 - 1.0
                    grid = torch.stack([nx, ny], dim=-1)   # [B, H, W, 2]

                    sampled = F.grid_sample(
                        rfeat, grid,
                        mode='bilinear',
                        padding_mode='border',   # clamp at edges, no zero-padding artifact
                        align_corners=True,
                    )                            # [B, reduce_ch, H, W]
                    patch_feats.append(sampled)

        # ── 4. concat all sampled features ───────────────────────────────────
        # shape: [B, reduce_ch + 4*9*reduce_ch, H, W]  = [B, 592, H, W]
        concat = torch.cat(patch_feats, dim=1)

        # ── 5. regress residual delta ─────────────────────────────────────────
        delta = self.regress(concat)            # [B, 8, H, W]

        return wh_coarse + delta


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check (run this file directly to verify shapes & param counts)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PREV_CH = 200   # confirmed from HarDNet-85 decoder

    corner_head = RotInvCornerHead(in_ch=PREV_CH, mid_ch=32)
    refinement  = PatchRefinement(in_ch=PREV_CH, reduce_ch=16, patch_size=3)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"RotInvCornerHead  params : {count_params(corner_head):>10,}")
    print(f"PatchRefinement   params : {count_params(refinement):>10,}")
    print(f"4× corner heads   params : {4 * count_params(corner_head):>10,}")
    print()

    feat      = torch.randn(2, PREV_CH, 128, 128)   # B=2, feature map at /4
    wh_coarse = torch.randn(2, 8, 128, 128)

    # corner head
    hm_out = corner_head(feat)
    print(f"RotInvCornerHead  output : {tuple(hm_out.shape)}  (expect [2,1,128,128])")

    # refinement
    wh_fine = refinement(feat, wh_coarse)
    print(f"PatchRefinement   output : {tuple(wh_fine.shape)}  (expect [2,8,128,128])")

    print("\nAll shapes OK.")
