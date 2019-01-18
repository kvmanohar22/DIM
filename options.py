import argparse
import os

class Options(object):
   def __init__(self):
      self.parser = argparse.ArgumentParser()
      self.initialized = False

   def initialize(self):
      # Directory options
      self.parser.add_argument('--dataset_root', type=str, default='',
                               help='Root of dataset directory')
      self.parser.add_argument('--log_root', type=str, default='log',
                               help="Directory to log data")

      # Image options
      self.parser.add_argument('--H', type=int, default=320)
      self.parser.add_argument('--W', type=int, default=320)

      # Training options
      self.parser.add_argument('--train_mode', action='store_true', default=100)
      self.parser.add_argument('--max_epochs', type=int, default=100)
      self.parser.add_argument('--xyz', type=str, default='')
      self.parser.add_argument('--batch_size', type=int, default=32)
      self.parser.add_argument('--base_lr', type=float, default=1e-5)
      self.parser.add_argument('--gpu_id', type=int, default=-1)
      self.parser.add_argument('--ckpt_frq', type=int, default=10,
                               help="Frequency at which checkpoints are generated")
      self.parser.add_argument('--alpha', type=float, default=0.5,
                               help="Alpha value to weigh two types of losses")
      self.parser.add_argument('--epsilon', type=float, default=1e-6,
                               help="Small value used in loss computation")

      # Testing options
      self.parser.add_argument('--ckpt_path', type=str, default='ckpt/demo.npy',
                               help='Path to Checkpoint')
      self.parser.add_argument('--img_path', type=str, default='imgs/demo/imgs/img0.png',
                               help='Path to Image')
      self.parser.add_argument('--tri_path', type=str, default='imgs/demo/trimaps/tri0.png',
                               help='Path to Trimap')
      self.parser.add_argument('--newimgs_dir', type=str, default='imgs/demo/bgs',
                               help='Directory containing new image to blend')

   def print(self, opts):
      print('-'*40)
      print(' '*15, 'Options')
      print('-'*40)
      for k, v in opts.items():
         print('{:^20s}: {}'.format(k, v))
      print('-'*40)

   def parse(self, train_mode=False):
      if not self.initialized:
         self.initialize()
      args = self.parser.parse_args()
      args.train_mode = train_mode

      # Create directories
      if not os.path.exists(args.log_root):
         os.mkdir(args.log_root)

      opts = vars(args)
      self.print(opts)

      return opts

if __name__ == '__main__':
   opts = Options().parse()
