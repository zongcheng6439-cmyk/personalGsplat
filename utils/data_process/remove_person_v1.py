#!/usr/bin/env python3
"""Generate valid-pixel masks from images using YOLO person bounding boxes.

From the repository root (/Users/zongcheng/code/3dgs/personalGsplat):
    python utils/data_process/remove_person_v1.py --project_dir /path/to/scene --device cuda:0
Replace /path/to/scene with a directory containing images/.
Outputs: masks/ with original relative names; optional previews in bound/.
Also called by utils/data_process/prepare_citygs_data.py.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from common import find_images

def optimize_mask(mask):
    """Expand person regions slightly and keep the result binary."""
    people = Image.fromarray(255 - np.asarray(mask, dtype=np.uint8), mode="L")
    people = people.filter(ImageFilter.MaxFilter(5))
    people = people.filter(ImageFilter.GaussianBlur(1.0))
    return np.where(np.asarray(people) > 127, 0, 255).astype(np.uint8)


def save_result(result, source, image_root, mask_root, bbox_root):
    """Called by main with the original input path, not a YOLO-generated name."""
    source = Path(source)
    relative = source.relative_to(image_root)
    image = result.orig_img
    height, width = image.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)

    boxes = [] if result.boxes is None else result.boxes.data.cpu().numpy()
    for x1, y1, x2, y2, confidence, _ in boxes:
        x1, x2 = np.clip([round(x1), round(x2)], 0, width - 1)
        y1, y2 = np.clip([round(y1), round(y2)], 0, height - 1)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 0

    if len(boxes):
        mask = optimize_mask(mask)
    mask_path = mask_root / relative
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the exact relative filename expected by colmap_adc.py, but always
    # use a lossless PNG payload so JPEG source names cannot corrupt a binary mask.
    Image.fromarray(mask, mode="L").save(mask_path, format="PNG")

    if bbox_root is not None:
        preview_path = bbox_root / relative
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_bgr = np.asarray(result.plot())
        Image.fromarray(preview_bgr[..., ::-1]).save(preview_path, format="PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--input_image", type=Path, default=Path("images"))
    parser.add_argument("--output_mask", type=Path, default=Path("masks"))
    parser.add_argument("--output_bbox", type=Path, default=Path("bound"))
    parser.add_argument("--save_bbox", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")

    image_root = (args.project_dir / args.input_image).resolve()
    mask_root = (args.project_dir / args.output_mask).resolve()
    bbox_root = (
        (args.project_dir / args.output_bbox).resolve() if args.save_bbox else None
    )
    images = find_images(image_root)
    if not images:
        parser.error(f"no images found below {image_root}")
    pending = [
        path
        for path in images
        if args.overwrite
        or not (mask_root / path.relative_to(image_root)).is_file()
        or (mask_root / path.relative_to(image_root)).stat().st_mtime_ns
        < path.stat().st_mtime_ns
    ]
    if not pending:
        print(f"Masks already exist for all {len(images)} images: {mask_root}")
        return

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ultralytics is required to generate missing masks; install it with "
            "`python -m pip install ultralytics`"
        ) from exc

    model = YOLO(args.model)
    progress = tqdm(total=len(pending), desc="person masks")
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        results = model.predict(
            source=[str(path) for path in batch],
            classes=[0],
            conf=args.confidence,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch_size,
            stream=True,
            verbose=False,
        )
        # List inputs produce results in input order; some Ultralytics versions
        # replace their paths with image0.jpg, image1.jpg, etc.
        results = list(results)
        if len(results) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} results, got {len(results)}")
        for source, result in zip(batch, results):
            save_result(result, source, image_root, mask_root, bbox_root)
            progress.update()
    progress.close()
    print(f"Images: {len(images)}, generated: {len(pending)}, masks: {mask_root}")


if __name__ == "__main__":
    main()
