import glob, os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import imagesc
from tqdm import tqdm

root = '/media/ghc/Ghc_data3/OAI_diffusion_final/diffusion_classification/'

#diff_list = sorted(glob.glob(root + 'diffddpm/*'))
diff_list = sorted(glob.glob(root + 'diff/*'))
aeffphi0_list = sorted(glob.glob(root + 'aeffphi0/*'))


painhist = []
aeffhist = []

for i in range(0, 500, 5):#tqdm(range(len(diff_list))):
    aeffphi0 = tiff.imread(aeffphi0_list[i])[:]
    diff = tiff.imread(diff_list[i])[:]
    pain = (diff >= 0.25) / 1

    painphi0 = aeffphi0 * pain

    painhist.append(np.histogram(painphi0.flatten(), bins=36, range=(1, 359))[0])
    aeffhist.append(np.histogram(aeffphi0.flatten(), bins=36, range=(1, 359))[0])

pp = np.stack(painhist, 0)
aa = np.stack(aeffhist, 0)
fig=plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.scatter(np.linspace(0, 1, pp.shape[1]), pp.mean(0))
plt.scatter(np.linspace(0, 1, aa.shape[1]), aa.mean(0))
plt.subplot(212)
plt.scatter(np.linspace(0, 1, aa.shape[1]), pp.mean(0)/(aa.mean(0)+1))
plt.ylim(0, 1)
plt.show()