import os
import random
import os.path as osp
import numpy as np
import cupy as cp
from skimage import io
import matplotlib.pyplot as plt

from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu

from chainercv.transforms import scale
from chainercv.transforms import resize
from chainercv.transforms import flip
from chainercv.transforms import random_crop
from chainercv.transforms import center_crop
from chainercv.transforms import random_flip

from DIM.options import Options
from DIM.utils.utils import load_image, load_trimap, load_alpha_matte

class Preprocess(object):
   def __init__(self, H, W):
      self.H = H
      self.W = W
      self.size = (H, W)

   # Change this to `random_crop`
   def crop(self, args, shape, crop_type=random_crop):
      """ Randomly crops the inputs to size `shape`

      Args:
         args     : tuple with (image, tri_map, alpha_matte)
         shape    : size (H, W) to crop to
         crop_type: Random during train and center during test
      """
      img, tri, alp = args
      img, params = crop_type(img, shape, return_param=True)
      x_slice, y_slice = params["x_slice"], params["y_slice"]
      if tri is None:
         return img
      tri = tri[:, y_slice, x_slice]
      alp = alp[:, y_slice, x_slice]
      return img, tri, alp

   def resize(self, args, size):
      """ Randomly crops the inputs to size `shape`

      Args:
         args: tuple with (image, tri_map, alpha_matte)
         size: size (H, W) to resize to
      """
      img, tri, alp = args
      img = resize(img, size)
      if tri is None:
         return img
      tri = resize(tri, size)
      if alp is None:
         return img, tri
      alp = resize(alp, size)
      return img, tri, alp

   def flip(self, args):
      """ Randomly flipped either horizontally / vertically

      Args:
         args: tuple with (image, tri_map, alpha_matte)
      """
      img, tri, alp = args
      img, params = random_flip(img, return_param=True)
      if params["x_flip"]:
         tri = flip(tri, x_flip=True)
         alp = flip(alp, x_flip=True)
      if params["y_flip"]:
         tri = flip(tri, y_flip=True)
         alp = flip(alp, y_flip=True)
      return img, tri, alp

   def is_valid(self, alp):
      """ Checks if the generated alpha matte is valid
      Note: Currently flagging those images with less than 7%
      """
      area = np.prod(alp.shape)
      nonzero_area = np.count_nonzero(alp)
      zero_area    = area - nonzero_area

      if zero_area <= area * 0.07 or nonzero_area <= area * 0.07:
         return False
      return True

   def train_mode_p(self, img, tri, alp):
      """ Preprocessing during train mode

      Args:
      img: Input image
      tri: Input tri map
      alp: Input alpha matte
      """
      # Step 1 (Resize to HxW) 
      img, tri, alp = self.crop((img, tri, alp), (self.H, self.W))

      # Step 2 (Random cropping to 480x480, 640x640 and back to 320x320)
      if False:
         img, tri, alp = self.crop((img, tri, alp), (480, 480))
         img, tri, alp = self.crop((img, tri, alp), (640, 640))
         img, tri, alp = self.resize((img, tri, alp), (self.H, self.W))

      # Step 3 (Random flipping)
      img, tri, alp = self.flip((img, tri, alp))
      valid = self.is_valid(alp)
      return img, tri, alp, valid 

   def test_mode_p(self, img, tri, alp):
      """ Preprocessing during test mode

      Args:
      img: Input image
      tri: Input tri map
      alp: Input alpha matte
      """
      img, tri, alp = self.crop((img, tri, alp), (self.H, self.W),
                                crop_type=center_crop)
      return img, tri, alp

   def __call__(self, img, tri, alp, train=False):
      if train:
         img, tri, alp, valid = self.train_mode_p(img, tri, alp)
      else:
         img, tri, alp = self.test_mode_p(img, tri, alp)

      return img, tri, alp, valid

class Loader(object):
   """ Dataset loader
   """

   def __init__(self, opts):
      self.H = opts["H"]
      self.W = opts["W"]
      self.gpu_id = opts["gpu_id"] >= 0
      self.train_mode = opts["train_mode"]
      self.batch_size = opts["batch_size"]
      self.type = "train" if self.train_mode else "test"
      self.preprocess = Preprocess(self.H, self.W)

      # Load the ids
      self.dataset_root = opts["dataset_root"]
      self.trimaps = os.listdir(osp.join(self.dataset_root, self.type, "trimap"))
      assert len(self.trimaps) >= 1, "No trimaps found"
      self.curr_trimap_idx = -1

      # Variables
      self.img_path = osp.join(self.dataset_root, self.type, "input")
      self.tri_path = osp.join(self.dataset_root, self.type, "trimap")
      self.alp_path = osp.join(self.dataset_root, self.type, "gt")
      self.bgs_path = osp.join(self.dataset_root, self.type, "bgs")

      self.bags = os.listdir(self.bgs_path)
      self.curr_bg_idx = -1
      self.ids = [file for file in os.listdir(self.img_path)]

   def __len__(self):
      return len(self.ids)

   def load_single(self, idx, trimap_base_dir):
      """ Loads inputs and outputs at index: `idx`"""
      count = 0
      while True:
         img_o = load_image(osp.join(self.img_path, self.ids[idx]))
         tri_o = load_trimap(osp.join(self.tri_path, trimap_base_dir, self.ids[idx]))
         alp_o = load_alpha_matte(osp.join(self.alp_path, self.ids[idx]))

         img, tri, alp, is_valid = self.preprocess(img_o, tri_o, alp_o, self.train_mode)
         if is_valid:
            return img, tri, alp
         else:
            count += 1
            if count > 10:
               img, tri, alp = self.preprocess.resize((img_o, tri_o, alp_o), (self.H, self.W))
               return img, tri, alp
      return img, tri, alp

   def load_background(self):
      """ Loads a background image
      """
      if self.curr_bg_idx == -1:
         self.curr_bg_idx = 0
      elif self.curr_bg_idx == len(self.bags):
         np.random.shuffle(self.bags)
         self.curr_bg_idx = 0

      img = load_image(osp.join(self.bgs_path, self.bags[self.curr_bg_idx]))
      self.curr_bg_idx += 1
      img = self.preprocess.resize((img, None, None), (self.H, self.W))
      return img

   def load_batch(self, start_idx, end_idx, change_tri_map):
      """ Loads batch of data

      Args:
         start_idx: Starting index
         end_idx  : Ending index
      """
      if change_tri_map and self.curr_trimap_idx == -1:
         self.curr_trimap_idx = 0
      elif change_tri_map:
         next_val = self.curr_trimap_idx+1
         np.random.shuffle(self.ids)
         self.curr_trimap_idx = next_val if next_val < len(self.trimaps) else 0

      # Select the background image
      images  = np.empty((self.batch_size, 3, self.H, self.W), dtype=np.float32)
      trimaps = np.empty((self.batch_size, 1, self.H, self.W), dtype=np.float32)
      outputs = np.empty((self.batch_size, 1, self.H, self.W), dtype=np.float32)
      backgrounds = np.empty((self.batch_size, 3, self.H, self.W), dtype=np.float32)
      files   = []
      for idx, curr_idx in enumerate(range(start_idx, end_idx)):
         img, tri, alp = self.load_single(curr_idx, self.trimaps[self.curr_trimap_idx])
         images[idx]  = img
         trimaps[idx] = tri
         outputs[idx] = alp
         backgrounds[idx] = self.load_background()
         files.append(self.ids[curr_idx])

      return images, trimaps, outputs, files, backgrounds

if __name__ == '__main__':
   opts = Options().parse(train_mode=True)
   loader = Loader(opts)
   idx = 12
   images, trimaps, outputs = loader.load_single(idx)
   print('Idx: {:2d} Images: {} Trimaps: {} Outputs: {}'.format(
      idx, images.shape, trimaps.shape, outputs.shape))
