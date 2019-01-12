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

from options import Options

class Preprocess(object):
   def __init__(self, H, W):
      self.H = H
      self.W = W
      self.size = (H, W)

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
      tri = tri[y_slice, x_slice]
      alp = alp[y_slice, x_slice]
      return img, tri, alp

   def resize(self, args, size):
      """ Randomly crops the inputs to size `shape`

      Args:
         args: tuple with (image, tri_map, alpha_matte)
         size: size (H, W) to resize to
      """
      img, tri, alp = args
      img = resize(img, size)
      tri = resize(tri, size)
      alp = resize(alp, size)
      return img, tri, alp

   def flip(self, args):
      """ Randomly flipped either horizontally / vertically

      Args:
         args: tuple with (image, tri_map, alpha_matte)
      """
      img, tri, alp = args
      img, params = random_flip(img, return_param=True)
      x_flip, y_flip = , params["y_flip"]
      if params["x_flip"]:
         tri = flip(tri, x_flip=True)
         alp = flip(alp, x_flip=True)
      if params["y_flip"]:
         tri = flip(tri, y_flip=True)
         alp = flip(alp, y_flip=True)
      return img, tri, alp

   def train_mode(self, img, tri, alp):
      """ Preprocessing during train mode

      Args:
      img: Input image
      tri: Input tri map
      alp: Input alpha matte
      """
      # Step 1 (Resize to HxW)
      img, tri, alp = self.crop((img, tri, alp), (self.H, self.W))

      # Step 2 (Random cropping to 480x480, 640x640 and back to 320x320)
      img, tri, alp = self.crop((img, tri, alp), (480, 480))
      img, tri, alp = self.crop((img, tri, alp), (640, 640))
      img, tri, alp = self.resize((img, tri, alp), (self.H, self.W))

      # Step 3 (Random flipping)
      img, tri, alp = self.flip((img, tri, alp))

   def test_mode(self, img, tri, alp):
      """ Preprocessing during test mode

      Args:
      img: Input image
      tri: Input tri map
      alp: Input alpha matte
      """
      img, tri, alp = self.crop((img, tri, alp), (self.H, self.W),
                                crop_type=center_crop)
      return img, tri, alp

   def __call__(self, img, tri, alp, train_mode):
      img = np.swapaxes(img, (2, 0, 1))
      tri = np.swapaxes(tri, (2, 0, 1))
      alp = np.swapaxes(alp, (2, 0, 1))
      
      if train_mode:
         img, tri, alp = self.train_mode(img, tri, alp)
      else:
         img, tri, alp = self.test_mode(img, tri, alp)

      img = np.swapaxes(img, (1, 2, 0))
      tri = np.swapaxes(tri, (1, 2, 0))
      alp = np.swapaxes(alp, (1, 2, 0))
      return img, tri, alp

class Loader(object):
   """ Dataset loader
   """

   def __init__(self, opts):
      self.H = opts["H"]
      self.W = opts["W"]
      self.use_cuda = opts["use_cuda"]
      self.train_mode = opts["train_mode"]

      self.preprocess = Preprocess(self.H, self.W)

      # Load the ids
      self.dataset_root = opts["dataset_root"]
      self.ids = []

      # Variables
      self.img_path = osp.join(self.dataset_root, "")
      self.tri_path = osp.join(self.dataset_root, "")
      self.alp_path = osp.join(self.dataset_root, "")

   def __len__(self):
      return len(self.ids)

   def load_image(self, idx):
      """ Loads image at index: `idx`"""
      img_file = self.ids[idx]
      return io.imread(self.img_path.format(img_file))

   def load_trimap(self, idx):
      """ Loads tri map at index: `idx`"""
      tri_file = self.ids[idx]
      return io.imread(self.tri_path.format(tri_file))

   def load_alpha_matte(self, idx):
      """ Loads alpha matte at index: `idx`"""
      alp_file = self.ids[idx]
      return io.imread(self.alp_path.format(alp_file))

   def load_single(self, idx):
      """ Loads inputs and outputs at index: `idx`"""
      img = self.load_image(idx)
      tri = self.load_trimap(idx)
      alp = self.load_alpha_matte(idx)

      img, tri, alp = self.preprocess(img, tri, alp, self.train_mode)

      inputs  = np.dstack([img, tri])
      outputs = alp

      return inputs, outputs

   def load_batch(self, start_idx, end_idx):
      """ Loads batch of data

      Args:
         start_idx: Starting index
         end_idx  : Ending index
      """
      inputs  = np.empty((self.H, self,W, 4), dtype=np.float32)
      outupts = np.empty((self.H, self.W, 1), dtype=np.float32)
      for idx, curr_idx in enumerate(range(start_idx, end_idx)):
         ins_outs = self.load_single(curr_idx)
         inputs[idx]  = ins_outs[0]
         outputs[idx] = ins_outs[1]

      if self.use_cuda:
         inputs, outputs = to_gpu(inputs), to_gpu(outputs)
      return inputs, outputs


if __name__ == '__main__':
   opts = Options().parse(train=True)
   loader = Loader(opts)
   img, tri, alp = loader.load_single(12)
   print('Image: {}\nTriMap: {}\nMatte: {}'.format(img.shape, tri.shape, alp.shape))

   opts = Options().parse(train=False)
   loader = Loader(opts)
   img, tri, alp = loader.load_single(12)
   print('Image: {}\nTriMap: {}\nMatte: {}'.format(img.shape, tri.shape, alp.shape))
