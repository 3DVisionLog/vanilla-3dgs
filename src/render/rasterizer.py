import torch
from torch.utils.cpp_extension import load
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

rasterizer_cuda = load(
    name="rasterizer_cuda",
    sources=[os.path.join(curr_dir, "rasterizer_kernel.cu")],
    verbose=True
)

class GaussianRasterizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, opacity, rgb, radii, H, W):
        means2d = means2d.contiguous()
        conics = conics.contiguous()
        opacity = opacity.contiguous()
        rgb = rgb.contiguous()
        radii = radii.contiguous()
        
        # C++ Forward 호출
        final_image = rasterizer_cuda.render_forward(
            means2d, conics, opacity, rgb, radii, H, W
        )
        
        ctx.save_for_backward(means2d, conics, opacity, rgb, radii, final_image)
        return final_image

    @staticmethod
    def backward(ctx, grad_img):
        # grad_img: Loss에서 넘어온 dLoss_dImage (H, W, 3)
        means2d, conics, opacity, rgb, radii, final_image = ctx.saved_tensors
        grad_img = grad_img.contiguous()
        
        # C++ Backward 호출
        grads = rasterizer_cuda.render_backward(
            means2d, conics, opacity, rgb, radii, final_image, grad_img
        )
        
        grad_means2d, grad_conics, grad_opacity, grad_rgb = grads
        
        # forward 인자랑 순서랑 개수 똑같아야 함(H, W는 미분 없으니 None 리턴)
        return grad_means2d, grad_conics, grad_opacity, grad_rgb, None, None, None