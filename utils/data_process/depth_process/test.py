import cv2
import os 
import numpy as np
from colmap import qvec2rotmat


def save_ply(image_meta, world_pred, save_dir, color=[255,0 ,0]):
    name = image_meta.name
    file_name = name.replace('/', '_').replace('.png', '.ply')
    save_path = os.path.join(save_dir, file_name)

    N = world_pred.shape[0]
    with open(save_path, "w") as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        # All points red (255,0,0)
        for p in world_pred:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {color[0]} {color[1]} {color[2]}\n")

def evaluate_sparse_reprojection(
        image_meta,
        camera,
        points3d_ordered,
        point3d_error_ordered,
        depth_map,
        scale,
        offset,
        point_max_error=1.5,
        inverse_depth=False
):
    """
    验证DepthAnything反投影是否正确

    Parameters
    ----------
    image_meta : COLMAP image
    camera : COLMAP camera
    points3d_ordered : Nx3 world points
    point3d_error_ordered : reconstruction error
    depth_map : DA3 depth
    scale offset
    inverse_depth:
        False:
            Z = scale*d+offset

        True:
            invZ = scale*d+offset
            Z=1/invZ
    """

    #########################################
    # intrinsic
    #########################################

    if camera.model in ["PINHOLE", "OPENCV"]:
        fx, fy, cx, cy = camera.params[:4]

    elif camera.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        f, cx, cy = camera.params[:3]
        fx = fy = f

    else:
        raise NotImplementedError(camera.model)

    #########################################
    # resize intrinsic
    #########################################

    sx = depth_map.shape[1] / camera.width
    sy = depth_map.shape[0] / camera.height

    fx *= sx
    fy *= sy
    cx *= sx
    cy *= sy

    #########################################
    # valid sparse points
    #########################################

    ids = image_meta.point3D_ids

    mask = ids >= 0
    mask &= ids < len(points3d_ordered)

    ids = ids[mask]

    xys = image_meta.xys[mask]

    errors = point3d_error_ordered[ids]

    good = errors < point_max_error

    ids = ids[good]

    xys = xys[good]

    gt_world = points3d_ordered[ids]

    #########################################
    # sample depth
    #########################################

    maps = np.stack([
        xys[:,0]*sx,
        xys[:,1]*sy
    ],axis=1).astype(np.float32)

    mono = cv2.remap(
        depth_map,
        maps[:,0],
        maps[:,1],
        interpolation=cv2.INTER_LINEAR
    )[:,0]

    #########################################
    # recover metric depth
    #########################################

    if inverse_depth:

        invz = scale*mono+offset

        z = 1.0/(invz+1e-8)

    else:

        z = scale*mono+offset

    #########################################
    # backproject
    #########################################

    u = maps[:,0]
    v = maps[:,1]

    X = (u-cx)/fx*z
    Y = (v-cy)/fy*z

    cam_pred = np.stack([X,Y,z],axis=1)

    #########################################
    # camera -> world
    #########################################

    R = qvec2rotmat(image_meta.qvec)

    t = image_meta.tvec

    world_pred = (R.T @ (cam_pred-t).T).T

    #########################################
    # error
    #########################################
    save_ply(image_meta, world_pred, 
             color = [255, 0, 0],
             save_dir= '/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/da2_p3d/')
    
    save_ply(image_meta, gt_world, 
             color=[255, 255, 0],
             save_dir= '/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/da2_p3d_gt/')

    err = np.linalg.norm(world_pred-gt_world,axis=1)

    rmse = np.sqrt(np.mean(err**2))

    mae = np.mean(err)

    med = np.median(err)

    print("------------------------------------")
    print(image_meta.name)
    print("Point Num :",len(err))
    print("RMSE      :",rmse)
    print("MAE       :",mae)
    print("Median    :",med)
    print("------------------------------------")

    return world_pred, gt_world

def build_world_map(
        image_meta,
        camera,
        depth_map,
        scale,
        offset,
        inverse_depth=False,
        save_dir = None,
):

    if camera.model in ["PINHOLE","OPENCV"]:

        fx,fy,cx,cy = camera.params[:4]

    elif camera.model in ["SIMPLE_PINHOLE","SIMPLE_RADIAL"]:

        f,cx,cy = camera.params[:3]

        fx=f
        fy=f

    else:
        raise NotImplementedError

    H,W = depth_map.shape

    sx = W/camera.width
    sy = H/camera.height

    fx*=sx
    fy*=sy
    cx*=sx
    cy*=sy

    if inverse_depth:

        invz = scale*depth_map+offset

        z = 1/(invz+1e-8)

    else:

        z = scale*depth_map+offset

    u,v=np.meshgrid(np.arange(W),np.arange(H))

    X=(u-cx)/fx*z
    Y=(v-cy)/fy*z

    cam=np.stack([X,Y,z],axis=-1)

    R=qvec2rotmat(image_meta.qvec)

    t=image_meta.tvec

    world=(R.T@(cam.reshape(-1,3)-t).T).T

    if save_dir is not None:
        save_ply(image_meta, world, 
             color=[255, 255, 255],
             save_dir= save_dir)
    
    world=world.reshape(H,W,3)
    return world


def select(image_meta,
           points3d_ordered,
           point3d_error_ordered,
           world_map,
           point_max_error = 1.5,           
           ):

    ids=image_meta.point3D_ids

    mask=ids>=0
    mask&=ids<len(points3d_ordered)

    ids=ids[mask]

    xy=image_meta.xys[mask]

    err=point3d_error_ordered[ids]

    good=err<point_max_error

    ids=ids[good]

    xy=xy[good]

    gt=points3d_ordered[ids]

    x=np.round(xy[:,0]).astype(int)
    y=np.round(xy[:,1]).astype(int)

    x=np.clip(x,0,world_map.shape[1]-1)
    y=np.clip(y,0,world_map.shape[0]-1)

    pred=world_map[y,x]
    save_ply(image_meta, pred, 
             color=[0, 255, 0],
             save_dir= '/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/da2_p3d_gt_select/')
    
    return pred 