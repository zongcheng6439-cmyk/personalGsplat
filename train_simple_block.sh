cd /code/gsplat
source /opt/gsplat-venv/bin/activate

python train_block_densification.py voxel_plane \
    --data-dir /lutu/parking01/ \
    --coarse-dir /lutu/parking01/citygs_weighted/coarse \
    --block-cameras-json /lutu/parking01/partition/partitions-dim_2_2_visibility_0.2/cameras-000_000.json \
    --result-dir /lutu/parking01/citygs_weighted/block/000_000 \
    --data-factor 1 \
    --test-every 0 \
    --max-steps 300000 \
    --save-ply \
    --disable-viewer \
    --sh-degree 0 \
    --depth-loss \
    --depth-refine-lambda 0.01 \
    --depth-lambda 0.05 \
    --strategy.noise_scale 30 \
    --strategy.cap-max 3004855 \
    --strategy.refine-start-iter 3000 \
    --strategy.refine-stop-iter 270000 \
    --strategy.refine-every 1500 \
    --strategy.plane-anchor-ply /lutu/parking01/cgal_voxel_plane_anchors.ply 




  source /opt/gsplat-venv/bin/activate

# 1. 卸载已安装版本
python -m pip uninstall -y gsplat

# 2. 删除 site-packages 中可能残留的 gsplat
find /opt/gsplat-venv/lib/python3.10/site-packages \
    -maxdepth 1 \
    \( -name 'gsplat' -o -name 'gsplat-*' -o -name 'gsplat.*' \) \
    -exec rm -rf -- {} +

# 3. 删除源码目录中的旧编译结果
rm -rf \
    /home/gsplat/build \
    /home/gsplat/dist \
    /home/gsplat/gsplat.egg-info \
    /home/gsplat/.eggs

# 4. 删除 gsplat 的 PyTorch JIT 编译缓存
if [ -d /root/.cache/torch_extensions ]; then
    find /root/.cache/torch_extensions \
        -type d -iname '*gsplat*' \
        -prune -exec rm -rf -- {} +
fi

# 5. 清理 pip 中 gsplat 的缓存
python -m pip cache remove gsplat || true

cd /code/gsplat
set -o pipefail

BUILD_EXPERIMENTAL=0 \
MAX_JOBS=24 \
PYTHONUNBUFFERED=1 \
python -u -m pip install -vvv . \
    --no-build-isolation \
    --no-deps \
    --no-cache-dir


