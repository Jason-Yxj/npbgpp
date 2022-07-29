import numpy as np
import yaml
import os
import open3d as o3d
from tqdm import tqdm
from SensorData import SensorData


def main(state, cfg, base_dir, save_dir, frame_skip):
    cfg_train = cfg[state]
    # if state == 'val' or state == 'test':
    #     state = 'train'
    base_dir = os.path.join(base_dir, f'scans_{state}')
    save_dir = os.path.join(save_dir, f'scans_{state}')

    extrinsics = []
    for scene_cfg in cfg_train:
        scene_name = scene_cfg['dataset_name']
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
        
        extrinsics_path = os.path.join(save_scene_dir, 'extrinsics.npy')
        print('exporting', len(sd.frames)//frame_skip, 'camera poses to', extrinsics_path)
        for f in tqdm(range(0, len(sd.frames), frame_skip), desc='export pose'):
            c2w = sd.frames[f].camera_to_world
            w2c = np.linalg.inv(c2w)
            extrinsics.append(w2c)
        np.save(extrinsics_path, extrinsics)

def cpy_ply(save_dir):
    ply_dir = save_dir
    save_dir = os.path.join(save_dir, 'scans_train')
    for scene_name in os.listdir(save_dir):
        scene_ply_dir = os.path.join(ply_dir, scene_name)
        save_scene_dir = os.path.join(save_dir, scene_name)
        os.system('cp ' + f'{scene_ply_dir}/full.ply ' + save_scene_dir)
    print('Done!')


base_dir = '/cwang/home/yxj/datasets/scannet/'
save_dir = '/cwang/home/yxj/datasets/scannet/cache/'

config_path = '/cwang/home/yxj/Project/npbgpp/configs/datasets/scannet_pretrain.yaml'
f = open(config_path)
cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

# main('train', cfg, base_dir, save_dir, frame_skip = 20)
cpy_ply(save_dir)