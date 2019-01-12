import numpy as np
import time

import chainer
import chainer.functions as F
import chainer.links as L

from chainer.backends.cuda import to_gpu
from chainer.backends.cuda import to_cpu


class EncoderDecoder(chainer.Chain):
   def __init__(self):
      super(EncoderDecoder, self).__init__()
      self.ini_c = 4
      self.out_c  = 64

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

         self.conv6_1 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=1, stride=1, pad=1)
         self.bn5     = L.BatchNormalization(self.out_c*8)
         self.dconv5  = L.Deconvolution2D(None, self.out_c*8, ksize=9)

         self.conv7_1 = L.Convolution2D(self.out_c*8, self.out_c*8, ksize=5, stride=1, pad=1)
         self.bn4     = L.BatchNormalization(self.out_c*8)
         self.dconv4  = L.Deconvolution2D(None, self.out_c*8, ksize=3)

         self.conv8_1 = L.Convolution2D(self.out_c*8, self.out_c*4, ksize=5, stride=1, pad=1)
         self.bn3     = L.BatchNormalization(self.out_c*4)
         self.dconv3  = L.Deconvolution2D(None, self.out_c*4, ksize=5, pad=1)

         self.conv9_1 = L.Convolution2D(self.out_c*4, self.out_c*2, ksize=5, stride=1, pad=1)
         self.bn2     = L.BatchNormalization(self.out_c*2)
         self.dconv2  = L.Deconvolution2D(None, self.out_c*2, ksize=3)

         self.convf_1 = L.Convolution2D(self.out_c*2, self.out_c*1, ksize=5, stride=1, pad=1)
         self.bn1     = L.BatchNormalization(self.out_c*1)
         self.dconv1  = L.Deconvolution2D(None, self.out_c*1, ksize=3)

         self.conv0_1 = L.Convolution2D(self.out_c*1, self.out_c, ksize=5, stride=1, pad=1)
         self.bn0     = L.BatchNormalization(self.out_c)

         self.convv   = L.Convolution2D(self.out_c, 1, ksize=1, stride=1, pad=1)

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

if __name__ == '__main__':
   model = EncoderDecoder().to_gpu()
   x = np.zeros((1, 4, 320, 320), dtype=np.float32)
   s = time.time()
   print(model(to_gpu(x)).shape)
