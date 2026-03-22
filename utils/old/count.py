from pathlib import Path
import numpy as np
import ipdb
from tqdm import tqdm
root="D:/GarmentCodeDataReady/train/00/data"
root=Path(root)

stitch_num_lst=[]
idx=-1
for i,it_dir in enumerate(root.iterdir()):
    # ipdb.set_trace()
    name=it_dir.name
    npz_path=it_dir/f"{name}_sampled.npz"
    npz=np.load(npz_path)
    tmp=npz["label"].sum()
    stitch_num_lst.append(tmp)

stitch_num_lst=np.array(stitch_num_lst)
        
import ipdb
ipdb.set_trace()
# print(stitch_num)
# print(idx)

path="D:/GarmentCode_5000_0_gt/train/002237/002237_sampled.npz"
npz=np.load(path)

ipdb.set_trace()
