import numpy as np
import gzip

M = np.loadtxt(gzip.open("data/X06.gz"))
print("shape:", M.shape)

real = M.reshape(2016, 144, 5)[:, :, 0]
tm = real.reshape(2016, 12, 12)
print("traffic matrices:", tm.shape)

hourly = tm.reshape(168, 12, 12, 12).sum(axis=1)
print("hourly:", hourly.shape)
