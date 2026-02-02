import torch

@torch.no_grad() # 얘는 backprop 필요 없으니까
def densify_and_prune(model, min_opacity=0.05, threshold_grad=0.0002, scene_extent=1.0):
    """
    Gaussian Splatting의 핵심: 점 개수 조절 (Adaptive Density Control)
    1. Clone: 작고 위치 변화가 큰 점 -> 복제
    2. Split: 크고 위치 변화가 큰 점 -> 쪼개기
    3. Prune: 투명한 점 -> 삭제
    """
    # 1. 기준값 설정
    # 픽셀을 디테일 챙겨서 채울려면... 결국 xyz 위치랑 가우시안의 크기가 중요해서 그거 두개가 기준임
    # SH grad가 크다고 그 점을 복사하거나 나눌 필요 없지? opacity나 scale도 마찬가지임
    # 근데 scale은 클론할 지,, 스플릿 할지를 결정해주는 기준이래
    grads = model.xyz.grad.norm(dim=1) # grad가 크다? -> 위치가 틀려서 빨리 이동해야 하는 점이라는 뜻
    scales = torch.exp(model.scale_log).max(dim=1).values # x,y,z 중 가장 큰 축 기준

    # 2. mask 생성 (위치라 생각하셈.. 벡터화 기법임)
    # clone은 grad 크고, 크기가 작은 애들
    # split은 grad 크고, 크기가 큰 애들
    clone_mask = (grads > threshold_grad) & (scales <= scene_extent * 0.01)
    split_mask = (grads > threshold_grad) & (scales > scene_extent * 0.01)

    # 3. 신규 생성할 텐서들 모음
    # split 당한 원본 점은 제외해야함!!!
    new_xyz_list = [model.xyz.data[~split_mask]] # [model.xyz.data]
    new_sh_list = [model.sh_coeffs.data[~split_mask]]
    new_opa_list = [model.opacity_logit.data[~split_mask]]
    new_sca_list = [model.scale_log.data[~split_mask]]
    new_rot_list = [model.rot_quat.data[~split_mask]]
    # clone이랑 split은 안겹치니 걍 ~split_mask로 처리하면 됨

    # 4. clone은 걍 원래 있는 점 복사임
    # 일단은 같은 점에 생기고, optimizer이 나중에 밀어냄
    if clone_mask.any():
        new_xyz_list.append(model.xyz.data[clone_mask])
        new_sh_list.append(model.sh_coeffs.data[clone_mask])
        new_opa_list.append(model.opacity_logit.data[clone_mask])
        new_sca_list.append(model.scale_log.data[clone_mask])
        new_rot_list.append(model.rot_quat.data[clone_mask])

    # 5. split은... 원래 있는 점 지우고 새 점 두 개 추가함
    if split_mask.any():
        # 원래 점의 분포 속에서 랜덤하게 2점 씩 뽑음
        stds = torch.exp(model.scale_log.data[split_mask])
        means = model.xyz.data[split_mask]

        # 논문 기준 1/1.6배 하래! 로그라서 뺌 log(1.6)=0.47
        new_scale = model.scale_log.data[split_mask] - 0.47

        for i in range(2):
            new_xyz_list.append(torch.normal(mean=means, std=stds))
            new_sca_list.append(new_scale)

            # 나머지 속성은 걍 그대로 복사
            new_sh_list.append(model.sh_coeffs.data[split_mask])
            new_opa_list.append(model.opacity_logit.data[split_mask])
            new_rot_list.append(model.rot_quat.data[split_mask])

    # concat으로 모든 리스트를 연결
    full_xyz = torch.cat(new_xyz_list, dim=0)
    full_sh = torch.cat(new_sh_list, dim=0)
    full_opa = torch.cat(new_opa_list, dim=0)
    full_sca = torch.cat(new_sca_list, dim=0)
    full_rot = torch.cat(new_rot_list, dim=0)

    # 6. Prune (삭제) // 투명도가 너무 낮은 애들 & 크기가 너무 커진 애들
    prune_mask = torch.sigmoid(full_opa).squeeze() < min_opacity
    big_mask = torch.exp(full_sca).max(dim=1).values > scene_extent * 1.0 # 장면 전체보다 크면 삭제

    final_mask = ~(prune_mask | big_mask)

    return {
        "xyz": full_xyz[final_mask],
        "sh_coeffs": full_sh[final_mask],
        "opacity_logit": full_opa[final_mask],
        "scale_log": full_sca[final_mask],
        "rot_quat": full_rot[final_mask]
    }