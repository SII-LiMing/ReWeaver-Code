import numpy as np
from PIL import Image

def center_human(img_arr, background_color=255):
    """
    将 img_arr 中的人像放置到画布中心
    img_arr: (3, H, W) 且范围在 [0, 1] 的 float32 数组
    """
    # 1. 转回 (H, W, C) 并还原到 0-255 方便处理
    temp_hwc = (img_arr.transpose(1, 2, 0) * 255).astype(np.uint8)
    H, W, _ = temp_hwc.shape

    # 2. 找到非背景像素的坐标
    # 假设背景是纯白 (255, 255, 255)
    # 如果背景不是纯白，可以用 mask_arr < 128 来定位
    mask = np.any(temp_hwc < background_color - 10, axis=-1) 
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img_arr # 全白图直接返回

    # 3. 计算人像的 Bounding Box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    human_h = y_max - y_min
    human_w = x_max - x_min

    # 4. 计算将人像中心移动到画布中心所需的偏移量
    center_y, center_x = H // 2, W // 2
    human_center_y, human_center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
    
    offset_y = center_y - human_center_y
    offset_x = center_x - human_center_x

    # 5. 使用 Roll 或平移创建新图像
    # 创建纯白背景
    centered_img = np.ones_like(temp_hwc) * 255
    
    # 计算粘贴范围，防止越界
    new_y1, new_y2 = max(0, y_min + offset_y), min(H, y_max + offset_y)
    new_x1, new_x2 = max(0, x_min + offset_x), min(W, x_max + offset_x)
    
    # 对应的原图切片范围
    src_y1, src_y2 = new_y1 - offset_y, new_y2 - offset_y
    src_x1, src_x2 = new_x1 - offset_x, new_x2 - offset_x

    centered_img[new_y1:new_y2, new_x1:new_x2] = temp_hwc[src_y1:src_y2, src_x1:src_x2]

    # 6. 转回 (3, H, W) float32
    return (centered_img.transpose(2, 0, 1) / 255.).astype(np.float32)