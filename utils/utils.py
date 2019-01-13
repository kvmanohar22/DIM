import numpy as np
from skimage import io

def blend(alpha, FG, BG):
   """ Generates a new image given alpha values

   Args:
      alpha: (N, 1, H, W) Alpha matte
      FG   : (N, 3, H, W) Fore-ground image
      BG   : (N, 3, H, W) Background image
   """
   alpha = np.repeat(alpha, 3, axis=1) / 255
   img   = alpha * FG + (1 - alpha) * BG
   return img.astype(np.uint8)

def extract_FG(image, alpha):
   """ Extracts fore-ground and background images

   Args:
      image: (N, 3, H, W) Image
      alpha: (N, 1, H, W) alpha matte
   """
   alpha = np.repeat(alpha, 3, axis=1) / 255
   FG = alpha * image
   return FG.astype(np.uint8)

def extract_BG(image, alpha):
   """ Extracts fore-ground and background images

   Args:
      image: (N, 3, H, W) Image
      alpha: (N, 1, H, W) alpha matte
   """
   alpha = np.repeat(alpha, 3, axis=1) / 255
   BG = (1 - alpha) * image
   return BG.astype(np.uint8)

def extract_FG_BG(image, alpha):
   """ Extracts fore-ground and background images

   Args:
      image: (N, 3, H, W) Image
      alpha: (N, 1, H, W) alpha matte
   """
   FG = extract_FG(image, alpha)
   BG = extract_BG(image, alpha)
   return FG, BG

def CHW2HWC(images):
   """ Convert images from CHW format to HWC format

   Args:
      images: (N, C, H, W)
   """
   return np.swapaxes(np.swapaxes(images, 1, 3), 1, 2)

def HWC2CHW(images):
   """ Convert images from HWC format to CHW format
   
   Args:
      images: (N, H, W, C)
   """
   return np.swapaxes(np.swapaxes(images, 1, 3), 2, 3)

def load_image(path):
   """ Loads the image in CHW format """
   img = io.imread(path)
   return HWC2CHW(np.array([img]))[0]

def load_trimap(path):
   """ Loads trimap in CHW format """
   trimap = io.imread(path)
   if trimap.ndim == 3:
      trimap = trimap[..., 0][None]
   else:
      trimap = trimap[None]
   return trimap

def load_alpha_matte(path):
   """ Loads alpha matte in CHW format """
   alp = io.imread(path)
   if alp.ndim == 3:
      alp = alp[..., 0][None]
   else:
      alp = alp[None]
   return alp
