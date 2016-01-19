from astrometry import *
from makeImage import stamp
from showImage import *


def demo():

    # make image
    Npix1D = 23
    Bkgd = 1000
    stamp1 = stamp(Npix1D, Bkgd)
    
    # add a Gaussian source
    muX = 0.0
    muY = 0.0
    alpha = 1.0
    Amplitude = 10000.0
    stamp1.addGaussianSource(muX, muY, alpha, Amplitude)
    # and add noise
    sigmaNoise = 100.0
    addsourcenoise = 1
    stamp1.addNoise(sigmaNoise, addsourcenoise)

    # and show it showStamp
    if (1):
        oneDpixels = stamp1.oneDpixels
        nonoise = stamp1.imageNoNoise 
        psf = stamp1.sourceImage
        image = stamp1.image
        diffimage = image - psf
        FourPanelStampPlot(oneDpixels, nonoise, psf, image, diffimage)

    return stamp1

# make a stamp
stamp1 = demo()


