import trimesh
from scipy.spatial import cKDTree
import numpy as np
from collections import defaultdict
import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil
from utils.utils import merge_stitch,find_border

"""
    将缝线点合并，并将缝线索引保存为 json 文件
"""
def do_merge_save_stitch(root):
    root_path = Path(root)
    for it_path in tqdm(root_path.iterdir(),desc="merge and save stitch..."):
        it_name=it_path.name
        mesh_path=str(it_path/f"{it_name}.obj")
        mesh_path_rename=str(it_path/f"{it_name}_ori.obj")
        new_mesh_path=str(it_path/f"{it_name}.obj")
        
        shutil.move(mesh_path,mesh_path_rename)
        
        json_path=str(it_path/f"{it_name}.json")
        mesh=trimesh.load(mesh_path_rename)
        new_mesh,stitch_idx=merge_stitch(mesh)
        
        border_idx=find_border(new_mesh)
        stitch_idx.extend(border_idx)
        stitch_idx=list(set(stitch_idx))
        with open(json_path, "w") as f:
            json.dump(stitch_idx, f, indent=2)
        new_mesh.export(new_mesh_path, include_texture=False)


if __name__ == "__main__":
    from multiprocessing import Pool

    def run_task(i):
        root = f"/amax/lm/Datahouse/GarmentCodeDataReady/test/{str(i).zfill(2)}/data"
        do_merge_save_stitch(root)

    with Pool(processes=16) as pool:  # 根据你机器的CPU核心数调整进程数
        pool.map(run_task, range(36))

    