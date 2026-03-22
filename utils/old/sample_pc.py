import numpy as np
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm
from pathlib import Path
import json
def poisson_disk_sample_mesh(mesh: trimesh.Trimesh, num_samples: int,stitch_idx=None, initial_factor: float = 5.0,stitch_scale=1) -> np.ndarray:
    """
    从 Trimesh 网格中使用泊松盘采样方法采样 N 个点。
    
    参数:
    - mesh: trimesh.Trimesh 对象
    - num_samples: 要采样的点数
    - initial_factor: 初始超采样因子，用于从中筛选出泊松点

    返回:
    - (N, 3) 的 numpy 数组，包含采样点的坐标
    """
    # 步骤 1: 超采样一组候选点（远多于 num_samples）
    candidate_count = int(num_samples * initial_factor)
    candidates, _ = trimesh.sample.sample_surface(mesh, candidate_count)
    
    # 步骤 2: 估计候选点间的最小距离（估算期望半径）
    area = mesh.area
    r = np.sqrt(area / (num_samples * np.pi))  # 理想的最小半径

    # 步骤 3: 使用KD树过滤成泊松盘采样（排除距离小于 r 的点）
    tree = cKDTree(candidates)
    sampled = []
    selected = np.full(len(candidates), False)

    for i in range(len(candidates)):
        if selected[i]:
            continue
        pt = candidates[i]
        sampled.append(pt)
        idxs = tree.query_ball_point(pt, r)
        selected[idxs] = True
        if len(sampled) >= num_samples:
            break
    
    pc=np.array(sampled)
    
    if stitch_idx is None:
        return pc

    pc_tree=cKDTree(pc)
    stitch_coords_orig = mesh.vertices[stitch_idx]
    # import ipdb
    # ipdb.set_trace()
    
    if stitch_scale==1:
        _, new_stitch_idx = pc_tree.query(stitch_coords_orig, k=1)
    else:
        
        _, new_stitch_idx = pc_tree.query(stitch_coords_orig, k=stitch_scale)  # new_stitch_idx shape: (N, 2)
        new_stitch_idx = np.unique(new_stitch_idx.flatten())  # 先展开再去重
    # assert new_stitch_idx.shape[0]==len(stitch_idx),"采样的 stitch 点比原来少"
    return pc,new_stitch_idx    


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def compute_seam_distances(pc: np.ndarray, seam_labels: np.ndarray) -> np.ndarray:
    """
    计算每个点到缝线的最短欧氏距离 (numpy 版)。

    Args:
        pc (np.ndarray): 点云坐标，形状 (N, 3)
        seam_labels (np.ndarray): 缝线标签，形状 (N,), dtype=bool

    Returns:
        np.ndarray: 每个点到最近缝线点的距离，形状 (N,)
    """
    # 提取缝线点
    seam_pts = pc[seam_labels]
    if seam_pts.size == 0:
        # 没有缝线点时，返回全 inf
        return np.full(pc.shape[0], np.inf, dtype=np.float32)
    # 构建 KD-Tree
    tree = cKDTree(seam_pts)
    # 查询最近邻
    dist, _ = tree.query(pc, k=1)
    return dist


def _do_sample(mesh_path,json_path,saved_path,sample_size,stitch_scale=1):
    # 1. 加载网格 & 缝线索引
    mesh = trimesh.load(mesh_path, process=False)
    with open(json_path, "r") as f:
        stitch_idx = json.load(f)

    # 2. Poisson Disk 采样
    pc, stitch_idx = poisson_disk_sample_mesh(mesh, sample_size, stitch_idx,stitch_scale=stitch_scale)
    # pc=pc_normalize(pc)
    pc = np.asarray(pc, dtype=np.float32)
    idx = np.asarray(stitch_idx, dtype=int)

    # 3. 构建缝线标签
    mask = np.zeros(sample_size, dtype=bool)
    mask[idx] = True

    # 4. 计算距离
    dist = compute_seam_distances(pc, mask)

    # 5. 保存结果
    np.savez(
        saved_path,
        pc=pc,
        label=mask,
        dist=dist
    )
def do_sample(root_dir: str,save_root:str, sample_size: int = 10000,stitch_scale=1):
    """
    按目录批量处理模型：
    - 读取 OBJ + JSON (缝线索引)
    - Poisson Disk 采样
    - 计算每点到缝线距离
    - 保存为 NPZ

    Args:
        root_dir (str): 存放模型子目录的根路径
        sample_size (int): 每个模型采样点数
    """
    root = Path(root_dir)
    for it_path in tqdm(list(root.iterdir()), desc="Sampling..."):
        name = it_path.name
        mesh_path = it_path / f"{name}.obj"
        json_path = it_path / f"{name}.json"
        
        
        save_path=Path(save_root)/(it_path.parents[1].name+it_path.name)
        save_path.mkdir(exist_ok=True,parents=True)
        
        _do_sample(mesh_path,json_path,save_path/"sampled.npz",sample_size,stitch_scale)

if __name__ == "__main__":
    from multiprocessing import Pool
    from pathlib import Path
    def run_task(i):
        root = f"/amax/lm/Datahouse/GarmentCodeDataReady/train/{str(i).zfill(2)}/data"

        for it_path in Path(root).iterdir():
            name=it_path.name
            print(it_path/"sampled.json","finished.")
            _do_sample(it_path/f"{name}.obj",it_path/f"{name}.json",it_path/"sampled",20000,stitch_scale=2)

    with Pool(processes=16) as pool:  # 根据你机器的CPU核心数调整进程数
        pool.map(run_task, range(36))


    