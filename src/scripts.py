from astrometry import *
from makeImage import stamp
from showImage import *


def demo(showStamp=0):

    # make image
    Npix1D = 23
    Bkgd = 1000
    s = stamp(Npix1D, Bkgd)
    
    # add a Gaussian source
    muX = 0.0
    muY = 0.0
    alpha = 2.0
    Amplitude = 10000.0
    s.addGaussianSource(muX, muY, alpha, Amplitude)
    # and add noise
    sigmaNoise = 100.0
    addsourcenoise = 1
    s.addNoise(sigmaNoise, addsourcenoise)

    # and show it
    if (showStamp):
        diffimage = s.image - s.sourceImage
        FourPanelStampPlot(s.oneDpixels, s.imageNoNoise, s.sourceImage, s.image, diffimage)

    return s

# make a stamp
stamp1 = demo(1)


