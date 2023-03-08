import matplotlib.pyplot as plt
import numpy as np
from random import sample
import pandas as pd

## Stackexchange user "NumPy's loadtxt is impressively slow"

df = pd.read_pickle('BZdf_newest.pkl')
ax = plt.gca()

plt.yscale('log')
df.plot(kind='line',x='Time',y='X', color='blue', ax=ax)
df.plot(kind='line',x='Time',y='Y', color='green', ax=ax)
df.plot(kind='line',x='Time',y='Z', color='red', ax=ax)
plt.yscale('log')
plt.xlabel('Time / s')
plt.ylabel('Concentration / [M]')
plt.title("Simulated Oregonator Concentrations")
plt.savefig("test plot", dpi = 300)
plt.legend(loc="upper right")
plt.show()
