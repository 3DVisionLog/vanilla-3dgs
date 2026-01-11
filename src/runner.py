from src.nerf.render import volume_render
from src.nerf.sampling import sample_z_vals
from src.nerf.encoder import positional_encode

def run_nerf(model, rays_o, rays_d, n_samples):
    # 좌표 정의 // 좌표(pts) = 출발점(o) + 거리(z_vals) x 방향(d)
    near, far = 2.0, 6.0
    z_vals = sample_z_vals(near, far, n_samples, rays_o.shape[0], train=False)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # (N_rand, 64, 3)
    dirs_expanded = rays_d[..., None, :].expand(pts.shape)

    # MLP는 (B, D) 일케 2차원만 받으니까 flatten 해줘야함
    flat_pts = pts.reshape(-1, 3).to(model.device)
    flat_dirs = dirs_expanded.reshape(-1, 3).to(model.device)

    encoded_pts = positional_encode(flat_pts, L=10) # (n_rays * n_samples, 3)
    encoded_dirs = positional_encode(flat_dirs, L=4)

    raw = model(encoded_pts, encoded_dirs) # (n_rays * n_samples, 4)
    raw = raw.reshape(rays_o.shape[0], n_samples, 4) # (n_rays, n_samples, 4)

    rgb = volume_render(raw, z_vals)

    return rgb


# dirs_expanded = rays_d[..., None, :].expand(pts.shape)
# encoded_dirs = dir_encoder(dirs_expanded.reshape(-1, 3))
# 이거 느림

# encoded_dirs = dir_encoder(rays_d)              # (n_rays, D)
# encoded_dirs = encoded_dirs[:, None, :].expand(
#     -1, config["n_samples"], -1
# ).reshape(-1, encoded_dirs.shape[-1])