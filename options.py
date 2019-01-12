import argparse

class Options(object):
   def __init__(self):
      self.parser = argparse.ArgumentParser()
      self.initialized = False

   def initialize(self):
      # Directory options
      self.parser.add_argument('--project_root', type=str, default='', 
                               help='Root of the project')
      self.parser.add_argument('--dataset_root', type=str, default='',
                               help='Root of dataset directory')

      # Image options
      self.parser.add_argument('--H', type=int, default=320)
      self.parser.add_argument('--W', type=int, default=320)

      # Training options
      self.parser.add_argument('--train_mode', action='store_true', default=100)
      self.parser.add_argument('--max_epochs', type=int, default=100)
      self.parser.add_argument('--batch_size', type=int, default=32)
      self.parser.add_argument('--base_lr', type=float, default=1e-5)
      self.parser.add_argument('--gpu_id', type=int, default=-1)
      self.parser.add_argument('--display_frq', type=int, default=10,
                               help="Frequency at which log data is displayed during training")
      self.parser.add_argument('--ckpt_frq', type=int, default=10,
                               help="Frequency at which checkpoints are generated")

      # Testing options

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
      opts = vars(args)
      self.print(opts)

      return opts

if __name__ == '__main__':
   opts = Options().parse()
