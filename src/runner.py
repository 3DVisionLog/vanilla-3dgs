from src.nerf.render import volume_render
from src.nerf.sampling import sample_z_vals
from src.nerf.encoder import positional_encode

def run_nerf(model, rays_o, rays_d, z_vals):
    # 좌표 정의 // 좌표(pts) = 출발점(o) + 거리(z_vals) x 방향(d)
    n_rays, n_samples = z_vals.shape

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # (N_rand, 64, 3)
    flat_pts = pts.reshape(-1, 3) # MLP는 (B, D) 일케 2차원만 받으니까 flatten 해줘야함
    encoded_pts = positional_encode(flat_pts, L=10) # (n_rays * n_samples, 3)
    
    # dirs_expanded = rays_d[..., None, :].expand(pts.shape)
    # flat_dirs = dirs_expanded.reshape(-1, 3).to(device)
    # encoded_dirs = positional_encode(flat_dirs, L=4)
    encoded_dirs = positional_encode(rays_d, L=4)   # (n_rays, D)
    encoded_dirs = encoded_dirs[:, None, :].expand(-1, n_samples, -1)
    encoded_dirs = encoded_dirs.reshape(-1, encoded_dirs.shape[-1])

    raw = model(encoded_pts, encoded_dirs) # (n_rays * n_samples, 4)
    raw = raw.reshape(rays_o.shape[0], n_samples, 4) # (n_rays, n_samples, 4)

    return raw