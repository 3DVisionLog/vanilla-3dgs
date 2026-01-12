import os
import json
import torch
import numpy as np
from PIL import Image
import math

def load_data(base_dir):
    with open(os.path.join(base_dir, f'transforms_train.json'), 'r') as f:
        meta = json.load(f)

    datas = []
        
    for frame in meta['frames']:
        img_path = os.path.join(base_dir, frame['file_path'] + '.png')

        image = torch.from_numpy(np.array(Image.open(img_path)) / 255.0).float()
        image = image[:, :, :3] # RGBA -> RGB (4채널 제거)

        c2w = torch.tensor(frame["transform_matrix"]).float()

        H, W = image.shape[:2]
        camera_angle_x = meta['camera_angle_x']
        focal = .5 * W / math.tan(0.5 * camera_angle_x)

        datas.append({
            "c2w": c2w,
            "image": image,
            "hwf": [H, W, focal]
        })

    return datas