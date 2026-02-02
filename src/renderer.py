import torch
import torch.nn as nn

from .gaussian.model import GaussianModel
from .gaussian.projection import gaussians_to_screen
from .gaussian.density import densify_and_prune
from .utils import get_conic, get_radii
from .rasterizer.rasterizer import GaussianRasterizerFunction
from .rasterizer.old.render import render

class GaussianRenderer(nn.Module):
    def __init__(self, num_points, xyz, rgb, config, device):
        super().__init__()
        self.model = GaussianModel(num_points, xyz, rgb).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': [self.model.xyz], 'lr': config["lr"]["xyz"]},
            {'params': [self.model.sh_coeffs], 'lr': config["lr"]["sh_coeffs"]},
            {'params': [self.model.opacity_logit], 'lr': config["lr"]["opacity_logit"]},
            {'params': [self.model.scale_log], 'lr': config["lr"]["scale_log"]},
            {'params': [self.model.rot_quat], 'lr': config["lr"]["rot_quat"]},
        ]) # ], lr=config["lr"]["default"])

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.5,
        )

    def forward(self, w2c, focal, H, W, use_cuda=True):
        # World 좌표계 기준 방향 벡터 계산(점 위치 - 카메라 위치)
        # view_dirs = xyz_cam @ w2c[:3, :3].T 일케 했었는데... 느리대 이거
        cam_pos = -w2c[:3, :3].T @ w2c[:3, 3]  # Camera position (World 좌표계)
        cam_pos = cam_pos.to(self.model.xyz.device)
        view_dirs = self.model.xyz - cam_pos # (N, 3)
        view_dirs = view_dirs / view_dirs.norm(dim=1, keepdim=True)  # 정규화

        # g3d: xyz, cov3d, rgb, opacity
        # g2d: means2d, cov2d, rgb, opacity
        g3d = self.model(view_dirs)
        _, g2d = gaussians_to_screen(g3d, w2c, focal, H, W)

        means2d = g2d["uv"]
        cov2d   = g2d["cov2d"]
        conics  = get_conic(cov2d)
        rgb     = g2d["rgb"]
        opacity = g2d["opacity"]
        radii = get_radii(cov2d)
        
        if use_cuda:
            img = GaussianRasterizerFunction.apply(means2d, conics, opacity, rgb, radii, H, W)
        else: 
            img = render(_, g2d, H, W)

        return img.clamp(0.0, 1.0)
    
    def densify(self, scene_extent, config, step):
        new_gaussian = densify_and_prune(
            self.model,
            min_opacity=0.01,
            threshold_grad=0.0002,
            scene_extent=scene_extent
        )

        self.model.replace_gaussians(new_gaussian)

        param_list = [
            {'params': [self.model.xyz], 'lr': config["lr"]["xyz"], 'initial_lr': config["lr"]["xyz"], 'name': 'xyz'},
            {'params': [self.model.sh_coeffs], 'lr': config["lr"]["sh_coeffs"], 'initial_lr': config["lr"]["sh_coeffs"], 'name': 'sh'},
            {'params': [self.model.opacity_logit], 'lr': config["lr"]["opacity_logit"], 'initial_lr': config["lr"]["opacity_logit"], 'name': 'opacity'},
            {'params': [self.model.scale_log], 'lr': config["lr"]["scale_log"], 'initial_lr': config["lr"]["scale_log"], 'name': 'scale'},
            {'params': [self.model.rot_quat], 'lr': config["lr"]["rot_quat"], 'initial_lr': config["lr"]["rot_quat"], 'name': 'rotation'},
        ]
        self.optimizer = torch.optim.Adam(param_list, lr=config["lr"]["default"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.5,
            last_epoch=step 
        )