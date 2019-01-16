import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

def main():
   loss_c = np.load("../archive/loss_constant.npy")
   loss_a = np.load("../archive/loss_adam.npy")
   loss_m = np.load("../archive/loss_momentum.npy")

   fig = plt.figure()
   plt.plot(loss_c, label='Constant LR')
   plt.plot(loss_a, label='Adam (alpha=1e-5)')
   plt.plot(loss_m, label='Momentum SGD (0.9)')

   plt.legend()
   plt.title('Loss with different learning rate schemes')
   plt.xlabel('epochs')
   plt.ylabel('loss')
   plt.savefig('../imgs/loss.png')

if __name__ == '__main__':
   main()
