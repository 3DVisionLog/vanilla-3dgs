import torch
import torch.nn as nn

class GaussianModel(nn.Module):
    """
    학습 가능한 Gaussian 집합
    """
    def __init__(self, num_points=100, xyz=None, rgb=None, degree=1):
        super().__init__()
        if xyz is not None:
            self.xyz = nn.Parameter(torch.from_numpy(xyz).float())
        else:
            self.xyz = nn.Parameter((torch.rand(num_points, 3) * 2 - 1))

        # Σ = (RS)(RS)^{T} 용
        self.scale_log = nn.Parameter(torch.full((num_points, 3), -3.0))
        rot = torch.zeros((num_points, 4))
        rot[:, 0] = 1.0 # 쿼터니언 만들기
        self.rot_quat = nn.Parameter(rot)

        # RGB(3채널) x 4개 계수(Degree 0 ~ 1)
        self.degree = degree
        sh_coeffs = torch.zeros((num_points, 3, (self.degree + 1)**2))
        if rgb is not None:
            sh_coeffs[:, :, 0] = torch.from_numpy(rgb).float() / 255.0
        self.sh_coeffs = nn.Parameter(sh_coeffs)
        
        self.opacity_logit = nn.Parameter(torch.full((num_points, 1), -2.197))

    def build_rotation(self, q): # 쿼터니언 -> 회전 행렬 (R) 변환
        q = torch.nn.functional.normalize(q) # 쿼터니언 정규화 (Unit Quaternion)
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        return torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - r*z), 2*(x*z + r*y),
            2*(x*y + r*z), 1 - 2*(x*x + z*z), 2*(y*z - r*x),
            2*(x*z - r*y), 2*(y*z + r*x), 1 - 2*(x*x + y*y)
        ], dim=-1).reshape(-1, 3, 3)
    
    def eval_sh(self, dirs):
        # SH 상수 (Basis functions)
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199

        # Degree 0 (상수항 - 모든 방향에서 똑같음)
        result = C0 * self.sh_coeffs[:, :, 0]

        if self.degree > 0:
            x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

            # Degree 1 (방향항 - x, y, z 방향에 따라 달라짐)
            result = result - C1 * y.unsqueeze(1) * self.sh_coeffs[:, :, 1] + \
                            C1 * z.unsqueeze(1) * self.sh_coeffs[:, :, 2] - \
                            C1 * x.unsqueeze(1) * self.sh_coeffs[:, :, 3]

        return result
    
    def forward(self, view_dirs):
        """
        init에서 sigmoid, exp 해줬더니 학습에서 터짐..
        아마 Optimizer이 조정하면서 막 -1000 씩 해버리면서 그러나봐!
        """
        opacity = torch.sigmoid(self.opacity_logit) # 불투명도는 0~1 사이 값
        scale = torch.exp(self.scale_log) # 반지름은 양수만 가능
        rgb = self.eval_sh(view_dirs)

        # Σ = (RS)(RS)^{T}
        S = torch.diag_embed(scale)  # RS 계산하려고 3x3 대각행렬 만듬
        R = self.build_rotation(self.rot_quat)
        cov3d = (R@S) @ (R@S).transpose(1, 2) # 지금 전부 (n, r, l) 일케 3차원이라 .T 쓰면 안됨

        return {
            "xyz": self.xyz,
            "cov3d": cov3d,
            "rgb": rgb,
            "opacity": opacity
        }
    
    def replace_gaussians(self, new_g):
        self.xyz = nn.Parameter(new_g["xyz"])
        self.sh_coeffs = nn.Parameter(new_g["sh_coeffs"])
        self.opacity_logit = nn.Parameter(new_g["opacity_logit"])
        self.scale_log = nn.Parameter(new_g["scale_log"])
        self.rot_quat = nn.Parameter(new_g["rot_quat"])

    def opacity_reset(self, new_opa=0.01):
        reset_logit = torch.log(
            torch.tensor(new_opa, device=self.opacity_logit.device)
            / (1 - new_opa)
        )
        with torch.no_grad():
            self.opacity_logit.fill_(reset_logit)

    def get_params(self):
        return [
            self.xyz, 
            self.scale_log, 
            self.opacity_logit, 
            self.rot_quat, 
            self.sh_coeffs
        ]