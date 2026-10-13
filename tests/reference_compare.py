#!/usr/bin/env python3
# SPDX-License-Identifier: Unlicense
"""Compare i.sam2.opencl debug dumps with the PyTorch sam2 code.

Run i.sam2.opencl with the environment variable I_SAM2_OPENCL_DUMP set to
a directory; it then writes, for each crop, the 8-bit crop it segmented,
the normalized encoder input, the image embedding and the two
high-resolution feature maps, plus the final mask list (masks.csv). This
script runs the reference implementation on the same 8-bit pixels and
reports:

1. the maximum difference between the reference preprocessing
   (SAM2ImagePredictor.set_image) and the dumped encoder input,
2. the relative error of the image embedding and of the high-resolution
   features,
3. how the automatically generated masks match (box IoU pairing).

Run i.sam2.opencl with minsize=0 for the mask comparison: the reference
generator does not remove small regions (see the manual).

Requires torch and sam2 (not needed by the module itself).

Usage:
    reference_compare.py DUMP_DIR CHECKPOINT [--config CONFIG] [--points N]
        [--iou 0.8] [--stability 0.95]
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

CONFIGS = {
    "tiny": "t",
    "small": "s",
    "base_plus": "b+",
    "large": "l",
}


def config_for(checkpoint):
    """Hydra config name matching a checkpoint file name."""
    name = os.path.basename(checkpoint)
    for variant, short in CONFIGS.items():
        if f"hiera_{variant}" in name:
            if name.startswith("sam2.1"):
                return f"configs/sam2.1/sam2.1_hiera_{short}.yaml"
            return f"configs/sam2/sam2_hiera_{short}.yaml"
    raise SystemExit(f"cannot guess the model config of {name}; use --config")


def load_dump(dump_dir, name, shape):
    path = os.path.join(dump_dir, f"crop000_{name}.f32")
    return np.fromfile(path, np.float32).reshape(shape)


def compare(what, ref, ours):
    rel = np.linalg.norm(ref - ours) / np.linalg.norm(ref)
    cos = float((ref * ours).sum() / (np.linalg.norm(ref) * np.linalg.norm(ours)))
    print(
        f"{what}: relative L2 error {rel:.3g}, cosine {cos:.6f}, "
        f"max abs difference {np.abs(ref - ours).max():.3g}"
    )


def box_iou(a, b):
    """IoU of inclusive xyxy boxes, as torchvision computes it."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump_dir")
    parser.add_argument("checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--points", type=int, default=32)
    parser.add_argument("--iou", type=float, default=0.8)
    parser.add_argument("--stability", type=float, default=0.95)
    args = parser.parse_args()

    rgb_path = min(glob.glob(os.path.join(args.dump_dir, "crop000_rgb_*.u8")))
    h, w = map(int, re.search(r"_(\d+)x(\d+)\.u8$", rgb_path).groups())
    rgb = np.fromfile(rgb_path, dtype=np.uint8).reshape(h, w, 3)
    inp = load_dump(args.dump_dir, "input", (3, 1024, 1024))
    emb = load_dump(args.dump_dir, "embedding", (64, 64, 256))
    s0 = load_dump(args.dump_dir, "feat_s0", (256, 256, 32))
    s1 = load_dump(args.dump_dir, "feat_s1", (128, 128, 64))

    sam = build_sam2(
        args.config or config_for(args.checkpoint), args.checkpoint, device="cpu"
    )
    gen = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=args.points,
        pred_iou_thresh=args.iou,
        stability_score_thresh=args.stability,
    )

    with torch.no_grad():
        pred = gen.predictor
        pred.set_image(rgb)
        ref_inp = pred._transforms(rgb).numpy()
        ref_emb = pred._features["image_embed"][0].permute(1, 2, 0).numpy()
        ref_s0, ref_s1 = (
            f[0].permute(1, 2, 0).numpy() for f in pred._features["high_res_feats"]
        )

    print(f"encoder input: max abs difference {np.abs(ref_inp - inp).max():.3g}")
    compare("embedding", ref_emb, emb)
    compare("features s0", ref_s0, s0)
    compare("features s1", ref_s1, s1)

    with torch.no_grad():
        ref = gen.generate(rgb)
    ours = np.loadtxt(
        os.path.join(args.dump_dir, "masks.csv"), delimiter=",", skiprows=1, ndmin=2
    )
    print(f"masks: reference {len(ref)}, i.sam2.opencl {len(ours)}")

    ref_boxes = [
        (
            m["bbox"][0],
            m["bbox"][1],
            m["bbox"][0] + m["bbox"][2],
            m["bbox"][1] + m["bbox"][3],
        )
        for m in ref
    ]
    ref_scores = {
        i: (m["predicted_iou"], m["stability_score"]) for i, m in enumerate(ref)
    }
    matched = 0
    ious = []
    for row in ours:
        scores = [box_iou(row[:4], rb) for rb in ref_boxes]
        best = int(np.argmax(scores)) if scores else -1
        ious.append(scores[best] if scores else 0.0)
        if scores and scores[best] > 0.9:
            matched += 1
            print(
                f"  box {row[:4].astype(int).tolist()}: IoU {scores[best]:.3f}, "
                f"predicted_iou {row[5]:.4f} vs {ref_scores[best][0]:.4f}, "
                f"stability {row[6]:.4f} vs {ref_scores[best][1]:.4f}"
            )
    if ious:
        print(
            f"masks with a reference box IoU > 0.9: {matched}/{len(ours)} "
            f"(median best IoU {np.median(ious):.3f})"
        )


if __name__ == "__main__":
    main()
