from .io import load_colmap_dataset, load_nerf_dataset

def load_data(base_dir, dataset_type="colmap"):
    if dataset_type == "nerf":
        return load_nerf_dataset(base_dir)
    elif dataset_type == "colmap":
        return load_colmap_dataset(base_dir)
    else:
        raise ValueError("Unknown dataset type")
