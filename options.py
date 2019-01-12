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

      # Training options

      # Testing options

   def print(self, opts):
      for k, v in opts.items():
         print('{:^20s}: {}'.format(k, v))

   def parse(self, train_mode=False):
      if not self.initialized:
         self.initialize()
      args = self.parser.parse_args()
      args.train = train_mode
      opts = vars(args)
      self.print(opts)

      return opts

if __name__ == '__main__':
   opts = Options().parse()
