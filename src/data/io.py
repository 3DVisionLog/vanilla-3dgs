import os
import json
import torch
import numpy as np
from PIL import Image
import math
import struct
from scipy.spatial.transform import Rotation

def load_image_tensor(path):
    return (
        torch.from_numpy(np.array(Image.open(path)))
        .float()
        .div(255.0)[..., :3]
    )

def load_nerf_dataset(base_dir, split="train"):
    with open(os.path.join(base_dir, f"transforms_{split}.json")) as f:
        meta = json.load(f)

    datas = []
    camera_angle_x = meta["camera_angle_x"]

    for frame in meta["frames"]:
        img_path = os.path.join(base_dir, frame["file_path"] + ".png")
        image = load_image_tensor(img_path)

        H, W = image.shape[:2]
        focal = 0.5 * W / math.tan(0.5 * camera_angle_x)

        c2w = torch.tensor(frame["transform_matrix"]).float()
        c2w[0:3, 1:3] *= -1 # NeRF 데이터셋을 3dgs에서 쓸려면 이거 필수아니겟삼?
        w2c = torch.linalg.inv(c2w)

        datas.append({
            "image": image,
            "w2c": w2c,
            "hwf": [H, W, focal],
        })

    return datas

"""바이너리 파일에서 특정 바이트만큼 읽어 포맷에 맞게 변환"""
def read_next_bytes(fid, num_bytes, format_char_sequence, endian="<"):
    return struct.unpack(endian + format_char_sequence, fid.read(num_bytes))

def load_cameras_bin(base_dir):
    cameras = {}
    path = os.path.join(base_dir, "sparse/cameras.bin")

    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_cameras):
            cam_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")

            # 대부분 SIMPLE_PINHOLE / PINHOLE 기준
            params = read_next_bytes(fid, 32, "dddd")
            focal = params[0]

            cameras[cam_id] = {
                "H": height,
                "W": width,
                "focal": focal,
            }

    return cameras

def load_colmap_dataset(base_dir):
    cameras = load_cameras_bin(base_dir)
    datas = []

    path = os.path.join(base_dir, "sparse/images.bin")
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "IdddddddI")
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]

            # image name
            name_bytes = []
            while True:
                c = read_next_bytes(fid, 1, "c")[0]
                if c == b"\x00": break
                name_bytes.append(c)
            img_name = b"".join(name_bytes).decode("utf-8")

            # skip points2d
            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2d * 24)

            # pose
            R = Rotation.from_quat(
                [qvec[1], qvec[2], qvec[3], qvec[0]]
            ).as_matrix()

            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = tvec

            cam = cameras[cam_id]
            image = load_image_tensor(os.path.join(base_dir, "images", img_name))

            datas.append({
                "image": image,
                "w2c": torch.from_numpy(w2c).float(),
                "hwf": [cam["H"], cam["W"], cam["focal"]],
            })

    return datas

def load_points3D_bin(base_dir):
    xyzs, rgbs = [], []
    path = os.path.join(base_dir, "sparse/points3D.bin")

    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):
            read_next_bytes(fid, 8, "Q")          # point3D_id
            X, Y, Z = read_next_bytes(fid, 24, "ddd")
            R, G, B = read_next_bytes(fid, 3, "BBB")
            read_next_bytes(fid, 8, "d")          # error

            track_len = read_next_bytes(fid, 8, "Q")[0]
            fid.read(track_len * 8)                # track data skip

            xyzs.append([X, Y, Z])
            rgbs.append([R, G, B])

    xyz = np.asarray(xyzs, dtype=np.float32)
    rgb = np.asarray(rgbs, dtype=np.float32)  # 0~255

    return xyz, rgb