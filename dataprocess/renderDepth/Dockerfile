# 用途：在用户指定 V2 基础上安装 LetsGo 深度光栅器，不编译/替换 gsplat。
# 输入：LetsGo 官方子模块；输出：本地 gsplat-letsgo-depth:local 镜像。
# 在本目录运行：docker build --build-arg CUDA_ARCH=7.5 -t gsplat-letsgo-depth:local .
FROM harbor.aibee.cn/robot/gsplat:V2
ARG CUDA_ARCH=7.5
ENV PATH=/opt/gsplat-venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY LetsGo/submodules/diff-gaussian-rasterization /opt/letsgo-rasterizer
RUN TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" MAX_JOBS=2 python -m pip install -v --no-build-isolation --no-deps /opt/letsgo-rasterizer \
    && python -c "import torch, scipy, plyfile, tqdm; import diff_gaussian_rasterization"
WORKDIR /code
ENTRYPOINT ["python"]
