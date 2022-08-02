from turtle import right
import numpy as np
from sklearn.metrics import jaccard_score
import yaml
import os
import open3d as o3d
from tqdm import tqdm
from SensorData import SensorData


def main(state, cfg, base_dir, save_dir, frame_skip):
    cfg_train = cfg[state]

    for scene_cfg in cfg_train:
        scene_name = scene_cfg['dataset_name']
        one_scene(base_dir, save_dir, scene_name, frame_skip)

def one_scene(base_dir, save_dir, scene_name, frame_skip):
    sens_path = os.path.join(base_dir, scene_name, f'{scene_name}.sens')
    save_scene_dir = os.path.join(save_dir, scene_name)
    os.makedirs(save_scene_dir, exist_ok = True)

    sd = SensorData(sens_path)
    W, H = sd.color_width, sd.color_height
    K = sd.intrinsic_color
    intrinsics_path = os.path.join(save_scene_dir, 'intrinsics.npy')
    print('exporting camera intrinsics to', intrinsics_path)
    intrinsics = {'width': W, 'height': H, 'K': K[:3, :3]}
    np.save(intrinsics_path, intrinsics)
    
    sd.export_color_images(os.path.join(save_scene_dir, 'images'), frame_skip = frame_skip)
    
    extrinsics = []
    extrinsics_path = os.path.join(save_scene_dir, 'extrinsics.npy')
    print('exporting', len(sd.frames)//frame_skip, 'camera poses to', extrinsics_path)
    for f in tqdm(range(0, len(sd.frames), frame_skip), desc='export pose'):
        c2w = sd.frames[f].camera_to_world
        w2c = np.linalg.inv(c2w)
        extrinsics.append(w2c)
    np.save(extrinsics_path, extrinsics)

def cpy_ply(save_dir):
    ply_dir = os.path.dirname(save_dir)
    for scene_name in os.listdir(save_dir):
        scene_ply_dir = os.path.join(ply_dir, scene_name)
        save_scene_dir = os.path.join(save_dir, scene_name)
        os.system('cp ' + f'{scene_ply_dir}/full.ply ' + save_scene_dir)
    print('Done!')

def delete_bad(save_dir, fram_skip):
    for scene_name in os.listdir(save_dir):
        save_scene_dir = os.path.join(save_dir, scene_name)
        img_dir = os.path.join(save_scene_dir, 'images')
        ext_path = os.path.join(save_scene_dir, 'extrinsics.npy')
        exts= np.load(ext_path)
        right_exts = []
        for idx, ext in enumerate(exts):
            if np.isinf(ext).sum() > 0 or np.isnan(ext).sum() > 0:
                img_path = os.path.join(img_dir, f'{idx * fram_skip}.jpg')
                os.system('rm ' + img_path)
            else:
                right_exts.append(ext)
        np.save(ext_path, right_exts)
    print('Done!')

base_dir = '/cwang/home/yxj/datasets/scannet/scans_train/'
save_dir = '/cwang/home/yxj/datasets/scannet/cache/scans_train/'

config_path = '/cwang/home/yxj/Project/npbgpp/configs/datasets/scannet_pretrain.yaml'
f = open(config_path)
cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

# main('train', cfg, base_dir, save_dir, frame_skip = 20)
# cpy_ply(save_dir)
# delete_bad(save_dir, fram_skip = 20)
one_scene(base_dir, save_dir, 'scene0100_00', frame_skip = 20)
one_scene(base_dir, save_dir, 'scene0101_00', frame_skip = 20)