import os
import shutil
root_from_lst=[
                "/amax/lm/Datahouse/GarmentCodeDataReady/train/00/data",
                "/amax/lm/Datahouse/GarmentCodeDataReady/train/01/data",
                "/amax/lm/Datahouse/GarmentCodeDataReady/train/02/data",
                "/amax/lm/Datahouse/GarmentCodeDataReady/test/00/data",
                "/amax/lm/Datahouse/GarmentCodeDataReady/test/01/data",
                "/amax/lm/Datahouse/GarmentCodeDataReady/test/02/data",
               ]
root_to_lst=[
    "/amax/lm/Datahouse/GarmentCodeDataNPZ/GarmentCodeDataNPZ_2w_train",
    "/amax/lm/Datahouse/GarmentCodeDataNPZ/GarmentCodeDataNPZ_2w_train",
    "/amax/lm/Datahouse/GarmentCodeDataNPZ/GarmentCodeDataNPZ_2w_train",
    "/amax/lm/Datahouse/GarmentCodeDataNPZ/GarmentCodeDataNPZ_2w_test",
    "/amax/lm/Datahouse/GarmentCodeDataNPZ/GarmentCodeDataNPZ_2w_test",
    "/amax/lm/Datahouse/GarmentCodeDataNPZ/GarmentCodeDataNPZ_2w_test",
]




for root_from,root_to in zip(root_from_lst,root_to_lst):
    files=os.listdir(root_from)

    for file in files:
        file_path=os.path.join(root_from,file,"complex_stitch_sample.npz")
        to_path=os.path.join(root_to,root_from.split("/")[-2]+file,"complex_stitch_sample.npz")
        shutil.copy(file_path,to_path)
        
        