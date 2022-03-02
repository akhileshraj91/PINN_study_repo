import numpy as np
import os.path, random
import torch
from torch.utils.data import Dataset

print("Torch version {}".format(torch.__version__))

# get training data
dir = "./"
if True:
    # download
    if not os.path.isfile('data-airfoils.npz'):
        import requests

        print("Downloading training data (300MB), this can take a few minutes the first time...")
        with open("data-airfoils.npz", 'wb') as datafile:
            resp = requests.get('https://dataserv.ub.tum.de/s/m1615239/download?path=%2F&files=dfp-data-400.npz',
                                verify=False)
            datafile.write(resp.content)

npfile = np.load(dir + 'data-airfoils.npz')

print("Loaded data, {} training, {} validation samples".format(len(npfile["inputs"]), len(npfile["vinputs"])))

print("Size of the inputs array: " + format(npfile["inputs"].shape))

import pylab

# helper to show three target channels: normalized, with colormap, side by side
def showSbs(a1,a2, stats=False, bottom="NN Output", top="Reference", title=None):
  c=[]
  for i in range(3):
    b = np.flipud( np.concatenate((a2[i],a1[i]),axis=1).transpose())
    min, mean, max = np.min(b), np.mean(b), np.max(b);
    if stats: print("Stats %d: "%i + format([min,mean,max]))
    b -= min; b /= (max-min)
    c.append(b)
  fig, axes = pylab.subplots(1, 1, figsize=(16, 5))
  axes.set_xticks([]); axes.set_yticks([]);
  im = axes.imshow(np.concatenate(c,axis=1), origin='upper', cmap='magma')

  pylab.colorbar(im); pylab.xlabel('p, ux, uy'); pylab.ylabel('%s           %s'%(bottom,top))
  if title is not None: pylab.title(title)

NUM=72
showSbs(npfile["inputs"][NUM],npfile["targets"][NUM], stats=False, bottom="Target Output", top="Inputs", title="3 inputs are shown at the top (mask, in-ux, in-uy), with the 3 output channels (p,ux,uy) at the bottom")
