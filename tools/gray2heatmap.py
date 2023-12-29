import cv2
import glob
from tqdm import tqdm
import os
import numpy as np

save_dir = './temp'
os.makedirs(save_dir, exist_ok=True)

images_path = sorted(glob.glob(r'/home/wangh20/projects/DRAEM_attention/tools/result/HANIT/Dongci/0100/101' + "/*heatmap.jpg"))
for img_path in tqdm(images_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    heatmap = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_JET)
    import pdb; pdb.set_trace()


