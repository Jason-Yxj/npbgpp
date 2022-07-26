import os
import cv2
import numpy as np
import open3d as o3d
from SensorData import SensorData
from tqdm import tqdm
from shutil import rmtree

def fusion(temp_dir, scene_dir):

    # load depth intrinsic
    K = np.fromfile(os.path.join(temp_dir, 'intrinsic', 'intrinsic_depth.txt'), dtype=np.float32, sep=' ').reshape(4, 4)
    K = o3d.camera.PinholeCameraIntrinsic(
        width = 640,
        height = 480,
        fx = K[0, 0],
        fy = K[1, 1],
        cx = K[0, 2],
        cy = K[1, 2]
    )

    # load poses
    poses = {}
    for filename in os.listdir(os.path.join(temp_dir, 'pose')):
        c2w = np.fromfile(os.path.join(temp_dir, 'pose', filename), dtype=np.float32, sep=' ').reshape(4, 4)
        # scannet 里面有的pose是inf，一定要检查！！！！
        if np.isinf(c2w).sum() > 0 or np.isnan(c2w).sum() > 0:
            continue
        poses[filename[:-4]] = c2w
    if len(poses.keys()) == 0:
        return False, None

    # depth to points
    points_merged = o3d.geometry.PointCloud()
    for frame_name in tqdm(poses.keys(), desc='fusion'):
        c2w = poses[frame_name]
        depth_path = os.path.join(temp_dir, 'depth', f'{frame_name}.png')
        depth = o3d.io.read_image(depth_path)
        pts = o3d.geometry.PointCloud().create_from_depth_image(
            depth,
            K,
            depth_scale = 1000.
        )
        d = np.asarray(pts.points)[:, 2]
        index, = np.where((d <= 5.5) & (d >= .5))
        pts = pts.select_by_index(index.tolist())
        pts = pts.transform(c2w)
        points_merged += pts

    points_merged = points_merged.voxel_down_sample(voxel_size = 0.02)
    points_merged, _ = points_merged.remove_radius_outlier(8, 0.1)    
    o3d.io.write_point_cloud(os.path.join(scene_dir, 'full.ply'), points_merged)

    return True, list(poses.keys())

def image_process(frames, temp_dir, scene_dir, crop, scale):

    frames = sorted(frames, key=int)

    K = np.fromfile(os.path.join(temp_dir, 'intrinsic', 'intrinsic_color.txt'), dtype=np.float32, sep=' ').reshape(4, 4)
    K[0, 2] -= crop[0]
    K[1, 2] -= crop[1]
    K[0, 0] /= scale
    K[1, 1] /= scale
    K[0, 2] /= scale
    K[1, 2] /= scale

    temp_pose_dir = os.path.join(temp_dir, 'pose')
    extrinsics = []
    temp_color_dir = os.path.join(temp_dir, 'color')
    scene_color_dir = os.path.join(scene_dir, 'images')
    os.makedirs(scene_color_dir, exist_ok = True)
    for frame in tqdm(frames, desc = 'image process'):
        c2w = np.fromfile(os.path.join(temp_pose_dir, f'{frame}.txt'), dtype=np.float32, sep=' ').reshape(4, 4)
        # poses.append(np.stack((c2w, K), axis = 0))
        extrinsics.append(c2w)

        img = cv2.imread(os.path.join(temp_color_dir, f'{frame}.jpg'))
        img = img[crop[1]:-crop[1], crop[0]:-crop[0]]
        H, W = img.shape[:2]
        img = cv2.resize(img, (W // scale, H // scale))
        cv2.imwrite(os.path.join(scene_color_dir, f'{frame}.jpg'), img)
    # poses = np.array(poses, dtype = np.float32)
    # frames = np.array(frames, dtype = np.int32)
    # np.savez_compressed(os.path.join(scene_dir, 'poses.npz'), frames = frames, poses = poses)
    intrinsics = {'width': W, 'height': H, 'K': K[:3, :3]}
    np.save(os.path.join(scene_dir, 'intrinsics.npy'), intrinsics)
    np.save(os.path.join(scene_dir, 'extrinsics.npy'), extrinsics)

def main(base_dir, save_dir, frame_skip = 20, crop = (16, 20), scale = 2):

    for scene_name in sorted(os.listdir(base_dir)):
        print('=' * 80)
        temp_scene_dir = os.path.join(save_dir, scene_name + '_temp')
        save_scene_dir = os.path.join(save_dir, scene_name)
        if os.path.isdir(save_scene_dir):
            continue

        os.makedirs(temp_scene_dir, exist_ok = True)
        os.makedirs(save_scene_dir, exist_ok = True)
        
        raw_path = os.path.join(base_dir, scene_name, f'{scene_name}.sens')
        sd = SensorData(raw_path)
        sd.export_color_images(os.path.join(temp_scene_dir, 'color'), frame_skip = frame_skip)
        sd.export_depth_images(os.path.join(temp_scene_dir, 'depth'), frame_skip = frame_skip)
        sd.export_poses(os.path.join(temp_scene_dir, 'pose'), frame_skip = frame_skip)
        sd.export_intrinsics(os.path.join(temp_scene_dir, 'intrinsic'))

        # depth to points
        success, frames = fusion(temp_scene_dir, save_scene_dir)
        if success:
            image_process(frames, temp_scene_dir, save_scene_dir, crop, scale)
        else:
            rmtree(save_scene_dir)
        rmtree(temp_scene_dir)

main(
    '/cwang/home/yxj/datasets/scannet/scans_train/',
    '/cwang/home/yxj/datasets/scannet/cache/scans_train/'
)

main(
    '/cwang/home/yxj/datasets/scannet/scans_test/',
    '/cwang/home/yxj/datasets/scannet/cache/scans_test/'
)
