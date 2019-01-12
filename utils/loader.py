import os
import os.path as osp
import numpy as np
import cupy as cp
from skimage import io

from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu

class Preprocess(object):
   def __init__(self):
      pass

   def __call__(self, )


class Loader(object):
   def __init__(self, opts):
      self.H = opts["H"]
      self.W = opts["W"]
      self.use_cuda = opts["use_cuda"]

      # Load the ids
      self.dataset_root = opts["dataset_root"]
      self.ids = []

      # Variables
      self.img_path = osp.join(self.dataset_root, "")
      self.tri_path = osp.join(self.dataset_root, "")
      self.alp_path = osp.join(self.dataset_root, "")

   def load_image(self, idx):
      img_file = self.ids[idx]
      return io.imread(self.img_path.format(img_file))

   def load_trimap(self, idx):
      tri_file = self.ids[idx]
      return io.imread(self.tri_path.format(tri_file))

   def load_alpha_matte(self, idx):
      alp_file = self.ids[idx]
      return io.imread(self.alp_path.format(alp_file))

   def load_single(self, idx):
      img = self.load_image(idx)
      tri = self.load_image(idx)
      alp = self.load_image(idx)

      inputs  = np.dstack([img, tri])
      outputs = alp

      return inputs, outputs

   def load_batch(self, start_idx, end_idx):
      inputs  = np.empty((self.H, self,W, 4), dtype=np.float32)
      outupts = np.empty((self.H, self.W, 1), dtype=np.float32)
      for idx, curr_idx in enumerate(range(start_idx, end_idx)):
         ins_outs = self.load_single(curr_idx)
         inputs[idx]  = ins_outs[0]
         outputs[idx] = ins_outs[1]

      if self.use_cuda:
         inputs, outputs = to_gpu(inputs), to_gpu(outputs)
      return inputs, outputs

