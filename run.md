
ssh zc@192.168.88.85 

/home/zc/code/gsplat-self

sudo docker run -it --rm --shm-size 8G  --gpus all \
    --entrypoint /bin/bash \
    -p 8100:8100 \
    -v /media/zc/SSD/GS/scannetTpp/068ba2946c/:/data/ \
    -v /home/zc/code/gsplat-self/personalGsplat:/code/gsplat \
    harbor.aibee.cn/robot/gsplat:V2 

source /opt/gsplat-venv/bin/activate
cd /code/gsplat
python setup.py install 

pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple

COARSE_GPU=4 utils/train/train_citygs_mcmc.sh /data/parking/lutu/parking01/ /data/parking/lutu/parking01/voxel_plane_0914/ all

COARSE_GPU=3 DATA_FACTOR=2 TARGET_GS=20000000 \
BLOCK_X=2 BLOCK_Y=2 \
utils/train/train_citygs_mcmc.sh /data/parking/zgcyh_b3/B3_seg/ /data/parking/zgcyh_b3/B3_seg/voxel_plane_0918/ all

COARSE_GPU=6 DATA_FACTOR=2 TARGET_GS=100000000 \
BLOCK_X=4 BLOCK_Y=4 \
utils/train/train_citygs_mcmc.sh /data/parking/zgcyh_b3/B3_full/ /data/parking/zgcyh_b3/B3_full/voxel_plane_0914/ all


COARSE_GPU=3 DATA_FACTOR=2 TARGET_GS=20000000 \
BLOCK_X=2 BLOCK_Y=2 \
utils/train/train_citygs_mcmc.sh /data/parking/zgcyh_b3/B3_seg/ /data/parking/zgcyh_b3/B3_seg/voxel_plane_0918/ all


apt install -y cmake build-essential

# rendering depth 
/data/algorithm/user/zcli/code
sudo sh run_hier.sh 
152qwertyuiop!@#$
cd /home/zcli/code/gaussians_nvs/gs_depth_prune
sh  render_depth.sh




