"""Train every partition with depth-supervised Texture MCMC variants.

``--method`` selects per-GS anchors, weighted voxel anchors, or voxel-plane
anchors. All methods share the same block partition, coarse PLY, GS budget,
texture maps, dense depth maps and GPU scheduler.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TQDM_PROGRESS = re.compile(r"(\d{1,3})%\|.*?\|\s*(\d+)/(\d+)")


class BlockProgress:
    """Keep one stable terminal progress row for every block."""

    def __init__(
        self, tasks: List["BlockTask"], steps_per_image: int, enabled: bool = True
    ):
        self.enabled = enabled and sys.stderr.isatty()
        self.bars = {}
        if not self.enabled:
            return
        for position, task in enumerate(tasks):
            total = max(1, task.image_count * steps_per_image)
            self.bars[task.name] = tqdm(
                total=total,
                desc=f"GPU- block {task.name} PENDING",
                position=position,
                leave=True,
                dynamic_ncols=True,
                unit="step",
            )

    def start(self, task: "BlockTask", gpu_id: int) -> None:
        if not self.enabled:
            print(
                f"[START] block={task.name} gpu={gpu_id} "
                f"images={task.image_count} cap_max={task.cap_max:,}"
            )
            return
        bar = self.bars[task.name]
        bar.set_description(f"GPU{gpu_id} block {task.name} RUNNING")
        bar.refresh()

    def consume(self, task: "BlockTask", line: str) -> None:
        """Extract the child gsplat tqdm state; the complete line stays in its log."""
        if not self.enabled:
            return
        clean = ANSI_ESCAPE.sub("", line).replace("\r", "").strip()
        matches = list(TQDM_PROGRESS.finditer(clean))
        if not matches:
            return
        match = matches[-1]
        bar = self.bars[task.name]
        current = min(int(match.group(2)), bar.total)
        bar.n = max(bar.n, current)
        detail = clean[: match.start()].rstrip(" :")
        if detail:
            bar.set_postfix_str(detail[-100:], refresh=False)
        bar.refresh()

    def finish(
        self, task: "BlockTask", gpu_id: int, status: str, elapsed: float
    ) -> None:
        if not self.enabled:
            print(f"[DONE] block={task.name} status={status} gpu={gpu_id}")
            return
        bar = self.bars[task.name]
        if status in {"SUCCESS", "SKIPPED"}:
            bar.n = bar.total
        bar.set_description(f"GPU{gpu_id} block {task.name} {status}")
        bar.set_postfix_str(f"elapsed={elapsed / 60:.1f}min", refresh=False)
        bar.refresh()

    def close(self) -> None:
        if not self.enabled:
            return
        for bar in reversed(list(self.bars.values())):
            bar.close()

    def write(self, message: str) -> None:
        tqdm.write(message) if self.enabled else print(message)


@dataclass
class BlockTask:
    block_id: int
    partition_id: Tuple[int, int]
    camera_json: Path
    area: float
    image_count: int
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]
    initial_gs: int = 0
    initial_ply: str = ""
    cap_max: int = 0

    @property
    def name(self) -> str:
        return f"{self.partition_id[0]:03d}_{self.partition_id[1]:03d}"


@dataclass
class BlockResult:
    block: str
    method: str
    gpu: int
    status: str
    return_code: int
    image_count: int
    bbox_area: float
    initial_gs: int
    initial_ply: str
    cap_max: int
    max_steps: int
    elapsed_seconds: float
    result_dir: str
    log_file: str


def gpu_free_memory_gb() -> Dict[int, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        int(gpu_id.strip()): float(free_mib.strip()) / 1024.0
        for gpu_id, free_mib in (line.split(",") for line in output.splitlines())
    }


def _block_id_from_path(path: Path) -> Tuple[int, int]:
    match = re.fullmatch(r"cameras-(\d+)_(\d+)\.json", path.name)
    if match is None:
        raise ValueError(f"Unexpected block camera filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_block_tasks(block_dir: Path, bbox_enlarge: float) -> Tuple[List[BlockTask], dict]:
    if bbox_enlarge < 0:
        raise ValueError("bbox_enlarge must be non-negative")
    partitions_path = block_dir / "partitions.pt"
    if not partitions_path.is_file():
        raise FileNotFoundError(f"Missing partition metadata: {partitions_path}")
    try:
        partitions = torch.load(partitions_path, map_location="cpu", weights_only=False)
    except TypeError:
        partitions = torch.load(partitions_path, map_location="cpu")

    coordinates = partitions["partition_coordinates"]
    ids = torch.as_tensor(coordinates["id"], dtype=torch.long)
    partition_size = torch.as_tensor(
        partitions["scene_config"]["partition_size"], dtype=torch.float64
    )
    id_to_index = {tuple(map(int, block_id)): i for i, block_id in enumerate(ids.tolist())}

    tasks = []
    camera_jsons = sorted(block_dir.glob("cameras-*_*.json"), key=_block_id_from_path)
    for camera_json in camera_jsons:
        partition_id = _block_id_from_path(camera_json)
        if partition_id not in id_to_index:
            raise ValueError(
                f"Block {partition_id} from {camera_json} is absent from partitions.pt"
            )
        index = id_to_index[partition_id]
        if partition_size.ndim == 0:
            size = partition_size.repeat(2)
        elif partition_size.ndim == 1:
            size = partition_size
        else:
            size = partition_size[index]
        block_xy = torch.as_tensor(coordinates["xy"], dtype=torch.float64)[index]
        bbox_min = block_xy - bbox_enlarge * size
        bbox_max = block_xy + (1.0 + bbox_enlarge) * size
        with camera_json.open("r", encoding="utf-8") as file:
            cameras = json.load(file)
        if not isinstance(cameras, list):
            raise ValueError(f"Camera JSON root must be a list: {camera_json}")
        tasks.append(
            BlockTask(
                # Use the authoritative ordering stored by partition.py.
                block_id=index,
                partition_id=partition_id,
                camera_json=camera_json,
                area=float(torch.prod(size).item()),
                image_count=len(cameras),
                bbox_min=tuple(map(float, bbox_min.tolist())),
                bbox_max=tuple(map(float, bbox_max.tolist())),
            )
        )
    if not tasks:
        raise FileNotFoundError(f"No cameras-<row>_<col>.json found in {block_dir}")
    return tasks, partitions


def find_latest_coarse_ply(coarse_dir: Path) -> Path:
    candidates = []
    for path in (coarse_dir / "ply").glob("point_cloud_*.ply"):
        match = re.search(r"(\d+)(?=\.ply$)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    point_cloud_dir = coarse_dir / "point_cloud"
    if point_cloud_dir.is_dir():
        for path in point_cloud_dir.rglob("*.ply"):
            relative = str(path.relative_to(point_cloud_dir))
            match = re.search(r"iteration[_-]?(\d+)", relative)
            if match:
                candidates.append((int(match.group(1)), path))
    if candidates:
        return max(candidates, key=lambda item: (item[0], str(item[1])))[1]
    plys = sorted(coarse_dir.rglob("*.ply"))
    if len(plys) == 1:
        return plys[0]
    raise FileNotFoundError(f"Cannot determine latest coarse PLY under {coarse_dir}")


def count_initial_block_gaussians(
    tasks: List[BlockTask], partitions: dict, ply_path: Path
) -> None:
    """Count each block's coarse Gaussians without copying the source PLY."""
    source_ply = PlyData.read(str(ply_path))
    vertex = source_ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    rotation = torch.as_tensor(
        partitions.get("extra_data", {}).get("rotation_transform", torch.eye(4))
    )[:3, :3].numpy()
    xy = (xyz @ rotation.T)[:, :2]
    scene_config = partitions["scene_config"]
    if scene_config.get("contract", False):
        aabb = torch.as_tensor(scene_config["aabb"]).numpy()
        xy = (xy - aabb[:2]) / (aabb[2:] - aabb[:2]) * 2.0 - 1.0
        magnitude = np.linalg.norm(xy, ord=np.inf, axis=-1, keepdims=True)
        outside = magnitude[:, 0] > 1.0
        xy[outside] = (2.0 - 1.0 / magnitude[outside]) * (
            xy[outside] / magnitude[outside]
        )
        xy = xy / 4.0 + 0.5
    for task in tasks:
        block_ply = (
            task.camera_json.parent
            / "block_init"
            / f"point_cloud_{task.name}.ply"
        )
        if block_ply.is_file():
            task.initial_gs = PlyData.read(str(block_ply))["vertex"].count
            task.initial_ply = str(block_ply)
            print(
                f"Using saved block init: block_id={task.block_id} "
                f"partition_id={task.partition_id} gs={task.initial_gs:,}"
            )
            continue
        bbox_min = np.asarray(task.bbox_min)
        bbox_max = np.asarray(task.bbox_max)
        mask = np.all((xy >= bbox_min) & (xy <= bbox_max), axis=-1)
        task.initial_gs = int(mask.sum())
        task.initial_ply = str(ply_path)
        print(
            f"Counted initial block GS: block_id={task.block_id} "
            f"partition_id={task.partition_id} gs={task.initial_gs:,}"
        )


def assign_cap_max(tasks: List[BlockTask], scene_target_gs: int) -> None:
    """Distribute the whole-scene GS budget evenly across all blocks."""
    if scene_target_gs <= 0:
        raise ValueError("scene_target_gs must be positive")
    base, remainder = divmod(scene_target_gs, len(tasks))
    for index, task in enumerate(tasks):
        task.cap_max = base + int(index < remainder)
        if task.cap_max < task.initial_gs:
            print(
                f"Block {task.name} visibility-expanded initialization has "
                f"{task.initial_gs:,} GSs, above its equal budget "
                f"{task.cap_max:,}; using the initial count as cap_max."
            )
            task.cap_max = task.initial_gs


def block_max_steps(task: BlockTask, steps_per_image: int) -> int:
    return max(1, task.image_count * steps_per_image)


def completed_block_ply(result_dir: Path, max_steps: int) -> Path | None:
    """Return a non-empty final PLY for exactly this training length."""
    ply_dir = result_dir / "ply"
    # Current trainers name the final zero-based iteration max_steps - 1.
    # Also accept max_steps for completed-step naming used by some exporters.
    for step in (max_steps, max_steps - 1):
        path = ply_dir / f"gsplat_point_cloud_{step}.ply"
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def skipped_block_result(
    task: BlockTask, args, result_dir: Path, max_steps: int
) -> BlockResult:
    return BlockResult(
        block=task.name,
        method=args.method,
        gpu=-1,
        status="SKIPPED",
        return_code=0,
        image_count=task.image_count,
        bbox_area=task.area,
        initial_gs=task.initial_gs,
        initial_ply=task.initial_ply,
        cap_max=task.cap_max,
        max_steps=max_steps,
        elapsed_seconds=0.0,
        result_dir=str(result_dir),
        log_file=str(result_dir / "train.log"),
    )


def build_command(task: BlockTask, args, result_dir: Path) -> Tuple[List[str], int]:
    refine_every = max(1, round(task.image_count * 0.3))
    depth_final_ramp_steps = max(1, task.image_count * 4)
    max_steps = block_max_steps(task, args.steps_per_image)
    refine_stop = round(max_steps * args.refine_stop_ratio)
    command = [
        args.python,
        "-u",
        str(args.trainer),
        "--coarse-dir", str(args.coarse_dir),
        "--data-dir", str(args.data_dir),
        "--result-dir", str(result_dir),
        "--block-cameras-json", str(task.camera_json),
        "--data-factor", str(args.data_factor),
        "--init-block-bbox-enlarge", str(args.bbox_enlarge),
        "--sh-degree", str(args.sh_degree),
        "--means-lr", str(args.means_lr),
        "--save-ply",
        "--depth-loss",
        "--depth-lambda", str(args.depth_lambda),
        "--depth-refine-lambda", str(args.depth_refine_lambda),
        "--depth-final-ramp-steps", str(depth_final_ramp_steps),
        "--disable-viewer",
        "--max-steps", str(max_steps),
        "--strategy.cap-max", str(task.cap_max),
        "--strategy.active-noise-steps", "-1",
        "--strategy.noise-scale", str(args.noise_scale),
        "--strategy.noise-injection-stop-iter", "-1",
        "--strategy.refine-start-iter", str(refine_every),
        "--strategy.refine-stop-iter", str(refine_stop),
        "--strategy.refine-every", str(refine_every),
        "--strategy.min-opacity", str(args.min_opacity),
        "--strategy.texture-weight", str(args.texture_weight),
        "--strategy.grad-weight", str(args.grad_weight),
        "--strategy.use-texture-score",
        "--strategy.use-grad-score",
    ]
    if args.method == "texture_voxel":
        command.extend(
            [
                "--strategy.surface-anchor-max-distance",
                str(args.anchor_max_distance),
                "--strategy.anchor-ply", str(args.anchor_ply),
                "--strategy.anchor-voxel-size", str(args.anchor_voxel_size),
            ]
        )
    elif args.method == "voxel_plane":
        command.extend(
            [
                "--strategy.plane-anchor-ply", str(args.plane_anchor_ply),
                "--strategy.anchor-voxel-size", str(args.anchor_voxel_size),
                "--strategy.initial-plane-thickness",
                str(args.initial_plane_thickness),
                "--strategy.plane-normal-tolerance",
                str(args.plane_normal_tolerance),
                "--strategy.anchor-max-distance",
                str(args.anchor_max_distance),
            ]
        )
    else:
        command.extend(
            [
                "--strategy.surface-anchor-max-distance",
                str(args.anchor_max_distance),
            ]
        )
    return command, max_steps


def train_block(task: BlockTask, gpu_id: int, args) -> BlockResult:
    result_dir = args.result_dir / "block" / task.name
    result_dir.mkdir(parents=True, exist_ok=True)
    log_file = result_dir / "train.log"
    info_file = result_dir / "block_training_info.json"
    command, max_steps = build_command(task, args, result_dir)
    start_time = time.monotonic()
    status = "DRY_RUN" if args.dry_run else "RUNNING"
    return_code = 0

    block_info = {
        "block": task.name,
        "method": args.method,
        "block_id": task.block_id,
        "partition_id": list(task.partition_id),
        "camera_json": str(task.camera_json),
        "bbox_min": list(task.bbox_min),
        "bbox_max": list(task.bbox_max),
        "bbox_area": task.area,
        "bbox_enlarge_per_side": args.bbox_enlarge,
        "image_count": task.image_count,
        "depth_final_ramp_steps": max(1, task.image_count * 4),
        "initial_gs": task.initial_gs,
        "initial_ply": task.initial_ply,
        "cap_max": task.cap_max,
        "scene_target_gs": args.scene_target_gs,
        "max_steps": max_steps,
        "gpu": gpu_id,
        "status": status,
        "return_code": None,
        "elapsed_seconds": None,
    }
    info_file.write_text(
        json.dumps(block_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    args.progress.start(task, gpu_id)
    if args.dry_run:
        print(" ".join(command))
    else:
        try:
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            with log_file.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=args.repo_dir,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    args.progress.consume(task, line)
                return_code = process.wait()
            status = "SUCCESS" if return_code == 0 else "FAILED"
        except Exception as error:
            return_code = -1
            status = "FAILED"
            with log_file.open("a", encoding="utf-8") as log:
                log.write(f"Launcher error: {error!r}\n")

    elapsed = time.monotonic() - start_time
    block_info.update(
        status=status,
        return_code=return_code,
        elapsed_seconds=round(elapsed, 2),
    )
    info_file.write_text(
        json.dumps(block_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.progress.finish(task, gpu_id, status, elapsed)
    return BlockResult(
        block=task.name,
        method=args.method,
        gpu=gpu_id,
        status=status,
        return_code=return_code,
        image_count=task.image_count,
        bbox_area=task.area,
        initial_gs=task.initial_gs,
        initial_ply=task.initial_ply,
        cap_max=task.cap_max,
        max_steps=max_steps,
        elapsed_seconds=round(elapsed, 2),
        result_dir=str(result_dir),
        log_file=str(log_file),
    )


def print_summary(results: List[BlockResult]) -> None:
    method = results[0].method if results else "unknown"
    print(f"\nBlock training summary: method={method}")
    print("block       status    gpu  images  area        initial_gs      cap_max     steps    minutes")
    for result in sorted(results, key=lambda item: item.block):
        print(
            f"{result.block:<11} {result.status:<9} {result.gpu:>3} "
            f"{result.image_count:>7} {result.bbox_area:>10.3f} "
            f"{result.initial_gs:>12,} {result.cap_max:>12,} {result.max_steps:>9,} "
            f"{result.elapsed_seconds / 60:>9.1f}"
        )


def parse_args():
    repo_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=(
            "texture_anchor",
            "texture_voxel",
            "voxel_plane",
        ),
        default="voxel_plane",
        help="All methods use texture maps, dense depth maps and anchors",
    )
    parser.add_argument("--data_dir", type=Path, default="/media/zc/SSD/ar/XX/B3/B3_seg/")
    parser.add_argument("--block_dir", type=Path, default="/media/zc/SSD/ar/XX/B3/B3_seg/partition/partitions-dim_3_3_visibility_0.2/")
    parser.add_argument("--coarse_dir", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, default="/media/zc/SSD/ar/XX/B3/B3_seg/gsaplt_large/")
    parser.add_argument(
        "--trainer",
        type=Path,
        default=None,
        help="Optional override; normally selected automatically from --method",
    )
    parser.add_argument("--anchor_ply", type=Path, default=None)
    parser.add_argument("--plane_anchor_ply", type=Path, default=None)
    parser.add_argument("--anchor_voxel_size", type=float, default=0.1)
    parser.add_argument("--initial_plane_thickness", type=float, default=1e-4)
    parser.add_argument("--plane_normal_tolerance", type=float, default=0.05)
    parser.add_argument("--anchor_max_distance", type=float, default=0.5)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--min_free_gb", type=float, default=20.0)
    parser.add_argument("--gpu_wait_seconds", type=float, default=30.0)
    parser.add_argument(
        "--launch_stagger_seconds", type=float, default=30.0,
        help="Delay consecutive launches so coarse PLY loading does not start simultaneously",
    )
    parser.add_argument("--data_factor", type=int, default=2)
    parser.add_argument(
        "--bbox_enlarge", type=float, default=0.0,
        help="Legacy fallback only; saved block_init PLYs are never bbox-expanded",
    )
    parser.add_argument("--steps_per_image", type=int, default=50)
    parser.add_argument("--refine_stop_ratio", type=float, default=0.8)
    parser.add_argument("--scene_target_gs", type=int, default=8_000_000)
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--means_lr", type=float, default=1.6e-5)
    parser.add_argument("--noise_scale", type=float, default=10.0)
    parser.add_argument("--depth_lambda", type=float, default=0.05)
    parser.add_argument("--depth_refine_lambda", type=float, default=0.01)
    parser.add_argument("--min_opacity", type=float, default=0.005)
    parser.add_argument("--texture_weight", type=float, default=2.0)
    parser.add_argument("--grad_weight", type=float, default=1.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.repo_dir = repo_dir
    default_trainers = {
        "texture_anchor": repo_dir / "utils/train/train_texture_anchor_block.py",
        "texture_voxel": repo_dir / "utils/train/train_weighted_anchor_block.py",
        "voxel_plane": repo_dir / "utils/train/train_voxel_plane_block.py",
    }
    args.trainer = (
        args.trainer.expanduser() if args.trainer else default_trainers[args.method]
    )
    if not args.trainer.is_file():
        parser.error(f"trainer does not exist: {args.trainer}")
    if args.method == "texture_voxel":
        if args.anchor_ply is not None:
            args.anchor_ply = args.anchor_ply.expanduser()
        if args.anchor_ply is None or not args.anchor_ply.is_file():
            parser.error("texture_voxel requires an existing --anchor_ply")
    if args.method == "voxel_plane":
        if args.plane_anchor_ply is not None:
            args.plane_anchor_ply = args.plane_anchor_ply.expanduser()
        if args.plane_anchor_ply is None or not args.plane_anchor_ply.is_file():
            parser.error("voxel_plane requires an existing --plane_anchor_ply")
        if args.initial_plane_thickness <= 0:
            parser.error("initial_plane_thickness must be positive")
        if args.plane_normal_tolerance <= 0:
            parser.error("plane_normal_tolerance must be positive")
    if args.anchor_max_distance <= 0:
        parser.error("anchor_max_distance must be positive")
    if args.depth_lambda < 0 or args.depth_refine_lambda < 0:
        parser.error("depth loss weights must be non-negative")
    if not 0 < args.refine_stop_ratio <= 1:
        parser.error("refine_stop_ratio must be in (0, 1]")
    if args.steps_per_image < 1:
        parser.error("steps_per_image must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    args.block_dir = args.block_dir.resolve()
    args.coarse_dir = args.coarse_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.result_dir = args.result_dir.resolve()
    args.trainer = args.trainer.resolve()
    if args.anchor_ply is not None:
        args.anchor_ply = args.anchor_ply.resolve()
    if args.plane_anchor_ply is not None:
        args.plane_anchor_ply = args.plane_anchor_ply.resolve()
    tasks, partitions = load_block_tasks(args.block_dir, args.bbox_enlarge)
    coarse_ply = find_latest_coarse_ply(args.coarse_dir)
    print(f"Counting initial block Gaussians from {coarse_ply}")
    count_initial_block_gaussians(tasks, partitions, coarse_ply)
    assign_cap_max(tasks, args.scene_target_gs)
    print(
        f"Loaded {len(tasks)} blocks; "
        f"method={args.method}, "
        f"scene_target_gs={args.scene_target_gs:,}, "
        f"block_cap_sum={sum(task.cap_max for task in tasks):,}"
    )
    for task in tasks:
        print(
            f"  block={task.name} images={task.image_count} "
            f"area={task.area:.3f} initial_gs={task.initial_gs:,} "
            f"cap_max={task.cap_max:,}"
        )

    results: List[BlockResult] = []
    pending = []
    running = {}
    args.progress = BlockProgress(tasks, args.steps_per_image, enabled=not args.dry_run)
    for task in tasks:
        result_dir = args.result_dir / "block" / task.name
        max_steps = block_max_steps(task, args.steps_per_image)
        final_ply = completed_block_ply(result_dir, max_steps)
        if final_ply is None:
            pending.append(task)
            continue
        args.progress.write(
            f"[SKIP] block={task.name} already reached "
            f"max_steps={max_steps:,}: {final_ply}"
        )
        args.progress.finish(task, -1, "SKIPPED", 0.0)
        results.append(skipped_block_result(task, args, result_dir, max_steps))

    gpu_count = (
        1
        if args.dry_run or not pending
        else max(1, len(gpu_free_memory_gb()))
    )
    try:
        with ThreadPoolExecutor(max_workers=gpu_count) as executor:
            while pending or running:
                busy_gpus = set(running.values())
                memory = ({gpu_id: math.inf for gpu_id in range(gpu_count)}
                          if args.dry_run else gpu_free_memory_gb())
                free_gpus = [
                    gpu_id for gpu_id, free_gb in memory.items()
                    if gpu_id not in busy_gpus and free_gb >= args.min_free_gb
                ]
                while pending and free_gpus:
                    task = pending.pop(0)
                    gpu_id = free_gpus.pop(0)
                    future = executor.submit(train_block, task, gpu_id, args)
                    running[future] = gpu_id
                    if pending and free_gpus and not args.dry_run and args.launch_stagger_seconds > 0:
                        time.sleep(args.launch_stagger_seconds)

                if running:
                    completed, _ = wait(
                        running, timeout=args.gpu_wait_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in completed:
                        running.pop(future)
                        results.append(future.result())
                elif pending:
                    args.progress.write(
                        f"No GPU has {args.min_free_gb:.1f} GB free; "
                        f"{len(pending)} blocks waiting..."
                    )
                    time.sleep(args.gpu_wait_seconds)
    finally:
        args.progress.close()

    print_summary(results)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.result_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Summary written to {summary_path}")
    has_failure = any(result.return_code != 0 for result in results)
    complete_path = args.result_dir / "block_training_complete.json"
    if has_failure:
        complete_path.unlink(missing_ok=True)
    else:
        complete_path.write_text(
            json.dumps(
                {"status": "SUCCESS", "block_count": len(results)},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Completion marker written to {complete_path}")
    return 1 if has_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
