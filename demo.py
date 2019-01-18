import numpy as np
import cupy as cp
import time
import os
import os.path as osp
from skimage import io

from nnet.model import EncoderDecoder
from options import Options
from utils.loader import Loader, Preprocess
from utils.utils import extract_FG_BG, blend, CHW2HWC, MEAN
from utils.utils import load_image, load_trimap
from utils.utils import mkdirs

import chainer
from chainer import optimizers
from chainer import serializers

from chainer.backends import cuda
from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu

def main(opts):
   H = opts["H"]
   W = opts["W"]

   # Setup the model
   print('Loading model...', end=' ')
   model = EncoderDecoder(opts)
   print('Done')
   if opts["gpu_id"] >= 0:
      model.to_gpu()

   # Load checkpoint
   print("Loading checkpoint...", end=' ')
   serializers.load_npz(opts["ckpt_path"], model)
   print('Done')

   # Load the data
   img = load_image(opts["img_path"])
   tri = load_trimap(opts["tri_path"])
   images, trimaps = Preprocess(H, W).resize((img, tri, None), (H, W))

   inputs = np.empty((1, 4, H, W), dtype=np.float32)
   inputs[0, :3, ...] = images - MEAN
   inputs[0, 3:, ...] = trimaps

   # Transfer to GPU
   inputs /= 255.0
   if opts["gpu_id"] >= 0:
      inputs  = to_gpu(inputs)

   # Forward pass
   with chainer.using_config('train', False), \
         chainer.function.no_backprop_mode():
      begin_time = time.time()
      predictions = model(inputs)
      end_time = time.time()

   # Predict RGB
   predicted_matte = (to_cpu(predictions.data) * 255).astype(np.uint8)
   predicted_alp = to_cpu(predictions.data)
   io.imsave(opts["img_path"].replace(".png", "_target_trimap.png"), np.squeeze(trimaps))
   predicted_matte = np.where(np.equal(trimaps, 128),
                              predicted_alp,
                              trimaps / 255)

   predicted_matte = (predicted_matte*255).astype(np.uint8)
   fgs, bgs = extract_FG_BG(images, predicted_matte)
   predicted_RGB = (blend(predicted_matte,
         fgs, bgs)).astype(np.uint8)

   # Save
   io.imsave(opts["img_path"].replace(".png", "_predicted_matte.png"), np.squeeze(predicted_matte[0]))
   new_bgs = os.listdir(opts["newimgs_dir"])
   if len(new_bgs) > 0:
      for idx, bg in enumerate(new_bgs):
         print("Processing {:2d}/{:2d} images".format(idx+1, len(new_bgs)))
         img_path = osp.join(opts["newimgs_dir"], bg)
         extension = img_path.split('.')[-1]

         # Load new image
         img = load_image(img_path)
         org_h, org_w = img.shape[1:]
         images, trimaps = Preprocess(H, W).resize((img, tri, None), (H, W))

         # Blend
         predicted_RGB = (blend(predicted_matte,
               fgs, images)).astype(np.uint8)
         images, trimaps = Preprocess(H, W).resize((predicted_RGB[0], tri, None), (org_h, org_w))

         # Save the image
         io.imsave(img_path.replace(".{}".format(extension), "_predicted_RGB.png"), CHW2HWC(images[None])[0])

if __name__ == '__main__':
   opts = Options().parse(train_mode=False)
   main(opts)