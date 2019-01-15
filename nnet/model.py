import numpy as np
import cupy as cp
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu
from chainer.initializers import GlorotUniform
from chainer.initializers import LeCunUniform

from chainercv.links import VGG16

from DIM.options import Options

class EncoderDecoder(chainer.Chain):
   def __init__(self, opts):
      super(EncoderDecoder, self).__init__()
      self.ini_c   = 4
      self.out_c   = 64
      self.alpha   = opts["alpha"]
      self.epsilon = opts["epsilon"]
      self.batch_size = opts["batch_size"]

      with self.init_scope():
         self.conv1_1 = L.Convolution2D(self.ini_c, self.out_c, ksize=3, stride=1, pad=1)
         self.conv1_2 = L.Convolution2D(self.out_c, self.out_c, ksize=3, stride=1, pad=1)

         self.conv2_1 = L.Convolution2D(self.out_c  , self.out_c*2, ksize=3, stride=1, pad=1)
         self.conv2_2 = L.Convolution2D(self.out_c*2, self.out_c*2, ksize=3, stride=1, pad=1)

         self.conv3_1 = L.Convolution2D(self.out_c*2, self.out_c*4, ksize=3, stride=1, pad=1)
         self.conv3_2 = L.Convolution2D(self.out_c*4, self.out_c*4, ksize=3, stride=1, pad=1)
         self.conv3_3 = L.Convolution2D(self.out_c*4, self.out_c*4, ksize=3, stride=1, pad=1)

         self.conv4_1 = L.Convolution2D(self.out_c*4, self.out_c*8, ksize=3, stride=1, pad=1)
         self.conv4_2 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=3, stride=1, pad=1)
         self.conv4_3 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=3, stride=1, pad=1)

         self.conv5_1 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=3, stride=1, pad=1)
         self.conv5_2 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=3, stride=1, pad=1)
         self.conv5_3 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=3, stride=1, pad=1)

         self.conv6_1 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=1, stride=1, pad=1,
                                        initialW=None)
         self.bn5     = L.BatchNormalization(self.out_c*8)
         self.dconv5  = L.Deconvolution2D(None, self.out_c*8, ksize=9, initialW=None)

         self.conv7_1 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=5, stride=1, pad=1,
                                        initialW=None)
         self.bn4     = L.BatchNormalization(self.out_c*8)
         self.dconv4  = L.Deconvolution2D(None, self.out_c*8, ksize=3, initialW=None)

         self.conv8_1 = L.Convolution2D(self.out_c*8, self.out_c*4, ksize=5, stride=1, pad=1,
                                        initialW=None)
         self.bn3     = L.BatchNormalization(self.out_c*4)
         self.dconv3  = L.Deconvolution2D(None, self.out_c*4, ksize=5, initialW=None, pad=1)

         self.conv9_1 = L.Convolution2D(self.out_c*4, self.out_c*2, ksize=5, stride=1, pad=1,
                                        initialW=None)
         self.bn2     = L.BatchNormalization(self.out_c*2)
         self.dconv2  = L.Deconvolution2D(None, self.out_c*2, ksize=3, initialW=None)

         self.convf_1 = L.Convolution2D(self.out_c*2, self.out_c*1, ksize=5, stride=1, pad=1,
                                        initialW=None)
         self.bn1     = L.BatchNormalization(self.out_c*1)
         self.dconv1  = L.Deconvolution2D(None, self.out_c*1, ksize=3, initialW=None)

         self.conv0_1 = L.Convolution2D(self.out_c*1, self.out_c, ksize=5, stride=1, pad=1,
                                        initialW=None)
         self.bn0     = L.BatchNormalization(self.out_c)

         self.convv   = L.Convolution2D(self.out_c, 1, ksize=1, stride=1, pad=1, initialW=None)

      # Load VGG network weights
      if opts["train_mode"]:
         self.load_vgg_weights()

   def __call__(self, x):
      x = F.relu(self.conv1_1(x))
      x = F.relu(self.conv1_2(x))
      x = F.max_pooling_2d(x, ksize=2, stride=2)

      x = F.relu(self.conv2_1(x))
      x = F.relu(self.conv2_2(x))
      x = F.max_pooling_2d(x, ksize=2, stride=2)

      x = F.relu(self.conv3_1(x))
      x = F.relu(self.conv3_2(x))
      x = F.relu(self.conv3_3(x))
      x = F.max_pooling_2d(x, ksize=2, stride=2)

      x = F.relu(self.conv4_1(x))
      x = F.relu(self.conv4_2(x))
      x = F.relu(self.conv4_3(x))
      x = F.max_pooling_2d(x, ksize=2, stride=2)

      x = F.relu(self.conv5_1(x))
      x = F.relu(self.conv5_2(x))
      x = F.relu(self.conv5_3(x))
      x = F.max_pooling_2d(x, ksize=2, stride=2)

      x = F.relu(self.bn5(self.conv6_1(x)))
      x = self.dconv5(x)
      x = F.unpooling_2d(x, stride=2, ksize=5, pad=1)

      x = F.relu(self.bn4(self.conv7_1(x)))
      x = self.dconv4(x)
      x = F.unpooling_2d(x, stride=2, ksize=5, pad=1)

      x = F.relu(self.bn3(self.conv8_1(x)))
      x = self.dconv3(x)
      x = F.unpooling_2d(x, stride=2, ksize=5, pad=1)

      x = F.relu(self.bn2(self.conv9_1(x)))
      x = self.dconv2(x)
      x = F.unpooling_2d(x, stride=2, ksize=5, pad=1)

      x = F.relu(self.bn1(self.convf_1(x)))
      x = self.dconv1(x)

      x = F.relu(self.bn0(self.conv0_1(x)))
      x = F.sigmoid(self.convv(x))

      return x

   def load_vgg_weights(self):
      """ Loads the vgg weights
      """
      print('Loading VGG weights onto encoder...', end=' ')
      vgg = VGG16(pretrained_model='imagenet')
      # self.conv1_1.copyparams(vgg.conv1_1.conv)
      self.conv1_2.copyparams(vgg.conv1_2.conv)

      self.conv2_1.copyparams(vgg.conv2_1.conv)
      self.conv2_2.copyparams(vgg.conv2_2.conv)

      self.conv3_1.copyparams(vgg.conv3_1.conv)
      self.conv3_2.copyparams(vgg.conv3_2.conv)
      self.conv3_3.copyparams(vgg.conv3_3.conv)

      self.conv4_1.copyparams(vgg.conv4_1.conv)
      self.conv4_2.copyparams(vgg.conv4_2.conv)
      self.conv4_3.copyparams(vgg.conv4_3.conv)

      self.conv5_1.copyparams(vgg.conv5_1.conv)
      self.conv5_2.copyparams(vgg.conv5_2.conv)
      self.conv5_3.copyparams(vgg.conv5_3.conv)
      print('Done')

   def loss(self, outputs, targets, trimaps):
      """ Compute the two losses
      """

      def alpha_prediction_loss(outputs, targets, wl):
         """ Computes alpha prediction loss

         Args:
            outputs: Predicted alpha matte
            targets: Target alpha matte
         """
         eps = cp.zeros_like(outputs, dtype=cp.float32) + self.epsilon
         loss = F.sqrt(F.square(outputs - targets) + F.square(eps))
         loss = F.mean(F.sum(loss * wl, axis=(1, 2, 3)) / F.sum(wl, axis=(1, 2, 3)))
         return loss

      def composition_loss(outputs, targets, wl):
         """ Computes composition loss

         Args:
            outputs: Predicted RGB
            targets: Target RGB
         """
         eps = cp.zeros_like(outputs, dtype=cp.float32) + self.epsilon
         loss = F.sqrt(F.square(outputs - targets) + F.square(eps))
         loss = F.mean(F.sum(loss * wl, axis=(1, 2, 3)) / F.sum(wl, axis=(1, 2, 3)))
         return loss

      output_matte, output_RGB = outputs
      target_matte, target_RGB = targets

      wl = F.where(cp.equal(trimaps, 128),
              cp.zeros_like(outputs[0], dtype=cp.float32)+1.0,
              cp.zeros_like(outputs[0], dtype=cp.float32)+0.0)
      wl2 = F.repeat(wl.data, 3, axis=1)

      l1 = alpha_prediction_loss(output_matte, target_matte, wl)
      l2 = composition_loss(output_RGB, target_RGB, wl2)
      return self.alpha * l1 + (1 - self.alpha) * l2

if __name__ == '__main__':
   opts = Options().parse()
   model = EncoderDecoder(opts)
   x = np.zeros((1, 4, opts["H"], opts["W"]), dtype=np.float32)
   if opts["gpu_id"] >= 0:
      model.to_gpu()
      x = to_gpu(x)
   print(model(x).shape)
