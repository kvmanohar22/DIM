import numpy as np
import cupy as cp
import time
import os
import os.path as osp
from skimage import io

from nnet.model import EncoderDecoder
from options import Options
from utils.loader import Loader
from utils.utils import extract_FG_BG, blend, CHW2HWC, mkdirs, MEAN

from chainer import optimizers
from chainer import serializers

from chainer.backends import cuda
from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu


def save(args, epoch, idx, root, ids):
   p_alp = CHW2HWC(args[0])
   t_alp = CHW2HWC(args[1])
   p_RGB = CHW2HWC(args[2])
   t_RGB = CHW2HWC(args[3])

   mkdirs([osp.join(root, 'epoch_{}'.format(epoch))])

   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_alp_{}".format(ids[0], 0)), np.squeeze(t_alp[0]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_alp_{}".format(ids[0], 0)), np.squeeze(p_alp[0]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_RGB_{}".format(ids[0], 0)), t_RGB[0])
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_RGB_{}".format(ids[0], 0)), p_RGB[0])

   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_alp_{}".format(ids[1], 1)), np.squeeze(t_alp[1]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_alp_{}".format(ids[1], 1)), np.squeeze(p_alp[1]))
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "t_RGB_{}".format(ids[1], 1)), t_RGB[1])
   io.imsave(osp.join(root, "epoch_{}".format(epoch), "p_RGB_{}".format(ids[1], 1)), p_RGB[1])

def main(opts):
   model = EncoderDecoder(opts)
   loader = Loader(opts)
   optimizer = optimizers.Adam(alpha=opts["base_lr"], amsgrad=True)

   MAX_EPOCHS = opts["max_epochs"]
   BATCH_SIZE = opts["batch_size"]
   H = opts["H"]
   W = opts["W"]
   n_imgs = len(loader)
   n_batches = n_imgs // BATCH_SIZE

   if opts["gpu_id"] >= 0:
      model.to_gpu()
   optimizer.setup(model)
   formatter = "Epoch: [{:3d}/{:3d}] Batch: [{:2d}/{:2d}] Time: {:.3f}s LR: {:.12f} Loss: {}"

   # Start the training process
   total_loss = []
   for epoch in range(1, MAX_EPOCHS+1):
      epoch_loss = []
      for idx, (batch_begin, batch_end) in enumerate(
            zip(range(0, n_imgs+1, BATCH_SIZE), 
                range(BATCH_SIZE, n_imgs+1, BATCH_SIZE))):

         # Prepare the data         
         images, trimaps, outputs, ids = loader.load_batch(batch_begin, batch_end, idx==0)
         fgs, bgs = extract_FG_BG(images, outputs)
         inputs = np.empty((BATCH_SIZE, 4, H, W), dtype=np.float32)
         inputs[:, :3, ...] = images - MEAN
         inputs[:, 3:, ...] = trimaps

         # Transfer to GPU
         if opts["gpu_id"] >= 0:
            inputs  = to_gpu(inputs) / 255.0
            images  = to_gpu(images) / 255.0
            trimaps = to_gpu(trimaps)
            outputs = to_gpu(outputs) / 255.0

         # Forward pass
         begin_time = time.time()
         predictions = model(inputs)
         end_time = time.time()

         # Predict RGB
         predicted_alp = to_cpu(predictions.data)
         predicted_matte = np.where(np.equal(to_cpu(trimaps), 128),
                                    predicted_alp,
                                    to_cpu(trimaps) / 255.0)
         predicted_RGB = (blend((predicted_matte*255).astype(np.uint8), fgs, bgs) / 255.0).astype(np.float32)

         # Compute loss
         if opts["gpu_id"] >= 0:
            predicted_RGB = to_gpu(predicted_RGB)
         loss = model.loss([predictions, predicted_RGB], [outputs, images], trimaps)
         loss_data = np.squeeze(to_cpu(loss.data))
         epoch_loss.append(loss_data)

         # Update the weights
         model.cleargrads()
         loss.backward()
         optimizer.update()

         # log
         if np.mod(idx, 5) == 0:
            print(formatter.format(epoch, MAX_EPOCHS+1, idx+1, n_batches, end_time-begin_time, optimizer.lr, str(loss_data)))

         # Save images
         if np.mod(epoch, 10) == 0:
            args = [(predicted_matte * 255).astype(np.uint8), 
                    (to_cpu(outputs) * 255).astype(np.uint8),
                    (to_cpu(predicted_RGB) * 255).astype(np.uint8),
                    (to_cpu(images) * 255).astype(np.uint8)]
            save(args, epoch, idx, opts["log_root"], ids)
      print('-'*10+' Avg epoch loss: {:.3f}'.format(np.mean(epoch_loss)))
      total_loss.append(np.mean(epoch_loss))

      # Checkpoint
      if np.mod(epoch, 20) == 0:
         serializers.save_npz(osp.join(opts["log_root"], "ckpt_{}".format(epoch)), model)
         np.save(osp.join(opts["log_root"], "epoch_{}.loss".format(epoch)), np.array(total_loss))

if __name__ == '__main__':
   opts = Options().parse(train_mode=True)
   main(opts)
