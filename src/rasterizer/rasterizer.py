import torch
from torch.utils.cpp_extension import load
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

rasterizer_cuda = load(
    name="rasterizer_cuda",
    sources=[
        os.path.join(curr_dir, "bindings.cpp"),
        os.path.join(curr_dir, "tile_binning.cu"),
        os.path.join(curr_dir, "rasterizer.cu")],
    verbose=True
)

def emit_and_sort_tile_pairs(means2d, radii, H, W):
    # tiles_ids: 복제된 타일 ID들
    # point_ids: 해당 타일에 속하는 원본 가우시안 인덱스
    tile_ids, point_ids = rasterizer_cuda.emitGaussianTilePairs(means2d, radii, H, W)

    sorted_tile_ids, order = torch.sort(tile_ids) # Radix Sort
    sorted_point_ids = point_ids[order] # 복제된 점들이 원래 몇 번 점이었는지

    return sorted_tile_ids, sorted_point_ids

def build_tile_ranges(sorted_tile_ids, H, W, device):
    """
    Tile Ranges 계산 (tild_id별 가우시안 시작점, 끝점)
    """
    unique_ids, counts = torch.unique_consecutive(sorted_tile_ids, return_counts=True)

    grid_width = (W + 16 - 1) // 16
    grid_height = (H + 16 - 1) // 16
    num_tiles = grid_width * grid_height

    tile_ranges = torch.zeros((num_tiles, 2), dtype=torch.int32, device=device)
    
    ends = torch.cumsum(counts, dim=0).int() # 누적합으로 끝 인덱스 계산
    starts = ends - counts.int()

    tile_ranges[unique_ids, 0] = starts
    tile_ranges[unique_ids, 1] = ends

    return tile_ranges # (N, 2) start, end

def gather_tiled_attributes(sorted_point_ids, means2d, conics, opacity, rgb, radii):
    return (
        means2d[sorted_point_ids].contiguous(),
        conics[sorted_point_ids].contiguous(),
        opacity[sorted_point_ids].contiguous(),
        rgb[sorted_point_ids].contiguous(),
        radii[sorted_point_ids].contiguous(),
    )

class GaussianRasterizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, opacity, rgb, radii, H, W):
        sorted_tile_ids, sorted_pt_ids = emit_and_sort_tile_pairs(means2d, radii, H, W) # sort만 안 한 상태

        tile_ranges = build_tile_ranges(sorted_tile_ids, H, W, means2d.device)

        tiled_means, tiled_conics, tiled_opacity, tiled_rgb, tiled_radii = gather_tiled_attributes(
            sorted_pt_ids, means2d, conics, opacity, rgb, radii
        )
        
        # C++ Forward 호출
        final_image = rasterizer_cuda.forwardCUDA(
            tiled_means, tiled_conics, tiled_opacity, tiled_rgb, tiled_radii, tile_ranges, H, W
        )
        
        ctx.save_for_backward(
            tiled_means, tiled_conics, tiled_opacity, tiled_rgb, tiled_radii, tile_ranges,
            sorted_pt_ids, # grad 복구할 때 쓰려고 저장
            final_image
        )
        ctx.num_original_points = means2d.shape[0] # 원본 점의 개수
        # t_means 같은걸 ctx에 저장할건데 그건 다 복제된 개수만큼의 크기를 가지니까
        
        return final_image

    @staticmethod
    # grad_img: Loss에서 넘어온 dLoss_dImage (H, W, 3)
    def backward(ctx, grad_img):
        t_means, t_conics, t_opac, t_rgb, t_radii, tile_ranges, sort_indices, final_img = ctx.saved_tensors
        grad_img = grad_img.contiguous()
        
        # C++ Backward 호출
        grads = rasterizer_cuda.backwardCUDA(
            t_means, t_conics, t_opac, t_rgb, t_radii, tile_ranges, final_img, grad_img
        )
        
        grad_t_means, grad_t_conics, grad_t_opac, grad_t_rgb = grads
        
        # forward 인자랑 순서랑 개수 똑같아야 함(H, W는 미분 없으니 None 리턴)
        # 원본 크기만큼의 빈 텐서 만들기
        N = ctx.num_original_points
        grad_means = torch.zeros((N, 2), device=t_means.device)
        grad_conics = torch.zeros((N, 3), device=t_means.device)
        grad_opac = torch.zeros((N, 1), device=t_means.device)
        grad_rgb = torch.zeros((N, 3), device=t_means.device)
        
        # 타일별 grad를 원래 grad로 sactter add
        # grad_...[i] 자체가 걍 i번째 복사본이 자기 타일에 만든 grad라는 의미임
        grad_means.index_add_(0, sort_indices, grad_t_means)
        grad_conics.index_add_(0, sort_indices, grad_t_conics)
        grad_opac.view(-1).index_add_(0, sort_indices, grad_t_opac.view(-1))
        grad_rgb.index_add_(0, sort_indices, grad_t_rgb)
        
        return grad_means, grad_conics, grad_opac, grad_rgb, None, None, None