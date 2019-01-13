from DIM.utils.utils import *
from DIM.options import Options

from skimage import io
import matplotlib.pyplot as plt
import os.path as osp


opts = Options().parse(train_mode=False)

image  = load_image(osp.join(opts["dataset_root"], "train", "input", "GT01.png"))
trimap = load_trimap(osp.join(opts["dataset_root"], "train", "trimap", "GT01.png"))
alpha  = load_alpha_matte(osp.join(opts["dataset_root"], "train", "gt", "GT01.png"))

# Test 1
print('Image shape: ', image.shape)
hwc = CHW2HWC(np.array([image]))
print('CHW TO HWC: ', hwc.shape)

# Test 2
chw = HWC2CHW(hwc)
print('HWC TO CHW: ', chw.shape)

# Test 3
fg, bg = extract_FG_BG(image[None], alpha[None])
io.imsave(osp.join(opts["log_root"], "image.png"), CHW2HWC(image[None])[0])
io.imsave(osp.join(opts["log_root"], "fg.png"), CHW2HWC(fg)[0])
io.imsave(osp.join(opts["log_root"], "bg.png"), CHW2HWC(bg)[0])

# Test 4
print(np.max(alpha))
print(alpha.dtype)
img = blend(alpha[None], fg, bg)
io.imsave(osp.join(opts["log_root"], "bimage.png"), CHW2HWC(img)[0])
