"""Build and run CGAL plane segmentation without polluting the repository.

Example:
    python utils/data_process/cgal_plane_segmentation/run.py \
        --input /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/sparse/0/extracted.ply \
        --output /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/cgal_planes.ply \
        --anchor-output /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/cgal_voxel_plane_anchors.ply \
        --anchor-voxel-size 0.1 \
        --overwrite
        

If ``--anchor-output`` is omitted, ``OUTPUT_anchors.ply`` is generated.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "gsplat_cgal_plane_build",
    )
    parser.add_argument("--rebuild", action="store_true")
    args, segmentation_args = parser.parse_known_args()
    if not segmentation_args or "--help" in segmentation_args:
        segmentation_args = ["--help"]

    source_dir = Path(__file__).resolve().parent
    build_dir = args.build_dir.resolve()
    binary = build_dir / "cgal_plane_segmentation"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    build_command = ["cmake", "--build", str(build_dir), "--parallel", "2"]
    if args.rebuild:
        build_command.append("--clean-first")
    subprocess.run(build_command, check=True)
    subprocess.run([str(binary), *segmentation_args], check=True)


if __name__ == "__main__":
    main()
