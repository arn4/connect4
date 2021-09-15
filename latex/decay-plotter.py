import matplotlib.pyplot as plt
import numpy as np

# Matplotlib style
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble='\\usepackage{amsmath} \\usepackage{siunitx}')
plt.style.use('seaborn')

def exp_decay_10k(x, d, me):
  return np.maximum(d**x, me)

def lin_decay_10k(x, d, me):
  return np.maximum(1.-d*x, me)

if __name__ == "__main__":
  x = np.linspace(0, 1e4, 10000)
  fig, ax = plt.subplots(figsize=(5,4))
  ax.plot(x, lin_decay_10k(x, 0.0001563772, 0.1), label='Linear')
  ax.plot(x, exp_decay_10k(x, 0.9996, 0.1), label='Exponential')
  ax.set_xlabel('Episodes')
  ax.set_ylabel('\\(\\varepsilon\\)')
  ax.legend()

  fig.savefig('img/decay.pdf', format = 'pdf', bbox_inches = 'tight')