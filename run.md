
ssh zc@192.168.88.85 

/home/zc/code/gsplat-self
rsync -avh --progress \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='*.egg-info/' \
  --exclude='.venv*/' \
  --exclude='.cache/' \
  --exclude='__pycache__/' \
  --exclude='*.so' \
  --exclude='.DS_Store' \
  /Users/zongcheng/code/3dgs/gsplat/ \
  zc@192.168.88.85:/home/zc/code/gsplat-self/