from DeconvNetFromNet import *

# Remove use_cpu=True if you have enough GPU memory
deconvNet = DeconvNet(use_cpu=False)
deconvNet.train()
