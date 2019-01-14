import numpy as np
import cupy as cp
import time
import os
import os.path as osp
from skimage import io

from nnet.model import EncoderDecoder
from options import Options
from utils.loader import Loader, Preprocess
from utils.utils import extract_FG_BG, blend, CHW2HWC
from utils.utils import load_image, load_trimap
from utils.utils import mkdirs

import chainer
from chainer import optimizers
from chainer import serializers

from chainer.backends import cuda
from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu

def save(args, epoch, idx, root):
   p_alp = CHW2HWC(args[0])
   t_alp = CHW2HWC(args[1])
   p_RGB = CHW2HWC(args[2])
   t_RGB = CHW2HWC(args[3])

   mkdirs([osp.join(root, 'epoch_{}'.format(epoch))])

   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_alp_iidx_{}_idx_{}.png".format(idx, 0)), np.squeeze(t_alp[0]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_alp_iidx_{}_idx_{}.png".format(idx, 0)), np.squeeze(p_alp[0]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_RGB_iidx_{}_idx_{}.png".format(idx, 0)), t_RGB[0])
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_RGB_iidx_{}_idx_{}.png".format(idx, 0)), p_RGB[0])

   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_alp_iidx_{}_idx_{}.png".format(idx, 1)), np.squeeze(t_alp[1]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_alp_iidx_{}_idx_{}.png".format(idx, 1)), np.squeeze(p_alp[1]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_RGB_iidx_{}_idx_{}.png".format(idx, 1)), t_RGB[1])
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_RGB_iidx_{}_idx_{}.png".format(idx, 1)), p_RGB[1])

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
   inputs[0, :3, ...] = images
   inputs[0, 3:, ...] = trimaps

   # Transfer to GPU
   if opts["gpu_id"] >= 0:
      inputs  = to_gpu(inputs) / 255.0
      images  = to_gpu(images) / 255.0
      trimaps = to_gpu(trimaps)

   # Forward pass
   with chainer.using_config('train', False), \
         chainer.function.no_backprop_mode():
      begin_time = time.time()
      predictions = model(inputs)
      end_time = time.time()

   # Predict RGB
   predicted_matte = (to_cpu(predictions.data) * 255).astype(np.uint8)
   predicted_alp = to_cpu(predictions.data)
   predicted_matte = np.where(np.equal(to_cpu(trimaps), 128),
                              predicted_alp,
                              to_cpu(trimaps))

   fgs, bgs = extract_FG_BG(images, outputs)
   predicted_RGB = (blend((predicted_matte*255).astype(np.uint8), fgs, bgs) / 255.0).astype(np.float32)

   if opts["gpu_id"] >= 0:
      predicted_RGB = to_gpu(predicted_RGB)

   #    # Save outputs
   #    args = [(predicted_matte * 255).astype(np.uint8), 
   #            (to_cpu(outputs) * 255).astype(np.uint8),
   #            (to_cpu(predicted_RGB) * 255).astype(np.uint8),
   #            (to_cpu(images) * 255).astype(np.uint8)]
   #    save(args, epoch, idx, opts["log_root"])

if __name__ == '__main__':
   opts = Options().parse(train_mode=False)
   main(opts)
