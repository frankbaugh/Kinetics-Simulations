import matplotlib.pyplot as plt
import numpy as np

d = np.loadtxt("folding.dat", unpack=True)
urea, D, I, N = d

plt.figure(figsize = (12, 8))
plt.plot(urea, D, label = "D", marker = "o")
plt.plot(urea, I, label = "I", marker = "x")
plt.plot(urea, N, label = "N", marker = "^")
plt.xlabel('[Urea] / M')
plt.ylabel('Mol ratio')
plt.title("Equilibrium Mol Ratios against Urea Concentration")
plt.legend(loc="center right")
plt.savefig("Mol ratio plot", dpi=300)
plt.show()
