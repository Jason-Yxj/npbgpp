import numpy as np
import os

path = '/cwang/home/yxj/datasets/scannet/cache/scans_train/'
list = os.listdir(path)
for f in list:
    sub_path = os.path.join(path, f) # /cwang/home/yxj/datasets/scannet/cache/scans_train/scene_xxxx
    sub_list = os.listdir(sub_path)
    for sub_f in sub_list:
        if sub_f == 'intrinsics.npy':
            camera = np.load(os.path.join(sub_path, sub_f), allow_pickle=True).item()
            camera['height'] = camera.pop('heigth')
            np.save(os.path.join(sub_path, sub_f), camera)

print('Done!')
