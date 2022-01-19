import matplotlib.pyplot as plt
import numpy as np

snr = np.arange(0.01, 1000,0.01)
x = np.array([0.01, 0.1, 1, 10, 20, 40, 100])
util = np.zeros(shape=(x.shape[0], snr.shape[0]))

for i, y in enumerate(x):
    util[i] = y * np.log(snr)

lineObjects = plt.plot(snr, util.T)
plt.legend(iter(lineObjects), ('0.01', '0.1', '1', '10', '20', '40', '100'))
# plt.plot(snr, util.T)
plt.title('$Υ(t,u) x \Omega(n_u,b_u)$ x SNR(t,u)')
plt.ylabel('Y(t,u)')
plt.xlabel('SNR(t,u)')
plt.show()