import numpy as np
import cupy as cp
import time
import os
import os.path as osp
from skimage import io

from nnet.model import EncoderDecoder
from options import Options
from utils.loader import Loader
from utils.utils import extract_FG_BG, blend, CHW2HWC

from chainer import optimizers
from chainer import serializers

from chainer.backends import cuda
from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu

def mkdirs(paths):
   for path in paths:
      os.mkdir(path)

def save(args, epoch, root):
   p_alp = CHW2HWC(args[0])
   t_alp = CHW2HWC(args[1])
   p_RGB = CHW2HWC(args[2])
   t_RGB = CHW2HWC(args[3])

   mkdirs([osp.join(root, 'epoch_{}'.format(epoch))])

   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_alp_idx_{}.png".format(0)), np.squeeze(t_alp[0]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_alp_idx_{}.png".format(0)), np.squeeze(p_alp[0]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_RGB_idx_{}.png".format(0)), t_RGB[0])
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_RGB_idx_{}.png".format(0)), p_RGB[0])

   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_alp_idx_{}.png".format(1)), np.squeeze(t_alp[1]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_alp_idx_{}.png".format(1)), np.squeeze(p_alp[1]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_RGB_idx_{}.png".format(1)), t_RGB[1])
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_RGB_idx_{}.png".format(1)), p_RGB[1])

def main(opts):
   model = EncoderDecoder(opts)
   loader = Loader(opts)
   optimizer = optimizers.Adam(alpha=opts["base_lr"])

   MAX_EPOCHS = opts["max_epochs"]
   BATCH_SIZE = opts["batch_size"]
   H = opts["H"]
   W = opts["W"]
   n_imgs = len(loader)
   n_batches = n_imgs // BATCH_SIZE

   if opts["gpu_id"] >= 0:
      model.to_gpu()
   optimizer.setup(model)
   formatter = "Epoch: [{:3d}/{:3d}] Batch: [{:2d}/{:2d}] Time: {:.3f}s LR: {:.6f} Loss: {}"

   # Start the training process
   for epoch in range(1, MAX_EPOCHS+1):
      for idx, (batch_begin, batch_end) in enumerate(
            zip(range(0, n_imgs+1, BATCH_SIZE), 
                range(BATCH_SIZE, n_imgs+1, BATCH_SIZE))):

         # Prepare the data         
         images, trimaps, outputs = loader.load_batch(batch_begin, batch_end)
         fgs, bgs = extract_FG_BG(images, outputs)
         inputs = np.empty((BATCH_SIZE, 4, H, W), dtype=np.float32)
         inputs[:, :3, ...] = images
         inputs[:, 3:, ...] = trimaps

         # Transfer to GPU
         if opts["gpu_id"] >= 0:
            inputs  = to_gpu(inputs) / 255.0
            images  = to_gpu(images) / 255.0
            trimaps = to_gpu(trimaps)
            outputs = to_gpu(outputs) / 255.0

         # print(inputs.dtype)
         # print(images.dtype)
         # print(trimaps.dtype)
         # print(outputs.dtype)

         # print(np.max(inputs))
         # print(np.max(images))
         # print(np.max(trimaps))
         # print(np.max(outputs))

         # Forward pass
         begin_time = time.time()
         predictions = model(inputs)
         end_time = time.time()

         # Predict RGB
         predicted_RGB = (blend((to_cpu(predictions.data) * 255).astype(np.uint8), fgs, bgs) / 255.0).astype(np.float32)
         # print(predictions.dtype)
         # print(np.max(to_cpu(predictions.data)))

         # print(predicted_RGB.dtype)
         # print(np.max(predicted_RGB))
         # exit(0)

         # Compute loss
         if opts["gpu_id"] >= 0:
            predicted_RGB = to_gpu(predicted_RGB)
         loss = model.loss([predictions, predicted_RGB], [outputs, images], trimaps)
         loss_data = np.squeeze(to_cpu(loss.data))

         # Update the weights
         model.cleargrads()
         loss.backward()
         optimizer.update()

         # Save outputs
         args = [(to_cpu(predictions.data) * 255).astype(np.uint8), 
                 (to_cpu(outputs) * 255).astype(np.uint8),
                 (to_cpu(predicted_RGB) * 255).astype(np.uint8),
                 (to_cpu(images) * 255).astype(np.uint8)]
         save(args, epoch, opts["log_root"])

         # log
         print(formatter.format(epoch, MAX_EPOCHS+1, idx+1, n_batches, end_time-begin_time, optimizer.lr, str(loss_data)))

      # Checkpoint
      if np.mod(epoch, 100) == 0:
         serializers.save_npz(osp.join(opts["log_root"], "ckpt_{}".format(epoch)), model)

if __name__ == '__main__':
   opts = Options().parse(train_mode=True)
   main(opts)
