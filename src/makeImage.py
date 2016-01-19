#!/usr/bin/env python

"""
   Definition and functions for a simple image (stamp) data structure: 
   a square image with Npix*Npix pixels, where input Npix is odd 
   N.B. the center of the central pixel of the stamp has coordinates (0,0) 
   Functions:
     - getOneDpixels(self, Npix)
     - set2Dpixels(self, Xpixels, Ypixels, Bkgd)
     - addGaussianSource(self, muX, muY, alpha)  
     - addDoubleGaussianSource(self, muX, muY, alpha)
     - addNoise(self, sigmaNoise, addsourcenoise=0, gain=1.0)
"""

import numpy as np
import math
from scipy import optimize
from scipy import interpolate


class stamp(object):

    def __init__(self, Npix, Bkgd):

        # make 1D pixel array
        self.getOneDpixels(Npix)

        # make 2D image and set to background Nkgd
        self.set2Dpixels(self.oneDpixels[:, np.newaxis], self.oneDpixels, Bkgd)
       
        # for self-awareness of added sources (debugging, etc)
        self.sourceAdded = 0 
        self.addGaussian = 0 
        self.addDoubleGaussian = 0 

    def getOneDpixels(self, Npix):
        # if Npix is even, increment by 1
        if (Npix/2*2 == Npix): Npix += 1
        NpixHalf = np.int(Npix/2)
        self.oneDpixels = np.linspace(-NpixHalf, NpixHalf, Npix)
        self.Npix = Npix

    def set2Dpixels(self, Xpixels, Ypixels, Bkgd):
        # make and set image to the background value
        r = np.sqrt(Xpixels**2 + Ypixels**2)
        self.image = np.empty(r.shape)
        self.image.fill(Bkgd)
        self.Xpixels = Xpixels
        self.Ypixels = Ypixels

    def addGaussianSource(self, muX, muY, alpha, A): 
        # add circular 2D gaussian at pixel coordinates (muX, muY)
        # its width in pixels is alpha and the total count is A
        r = np.sqrt((self.Xpixels-muX)**2 + (self.Ypixels-muY)**2)
        sourceImage = A*np.exp(-r**2/2/alpha**2) / (2*math.pi*alpha**2)
        self.sourceImage = sourceImage
        self.sourceAdded = 1 
        self.image += sourceImage
        self.addGaussian += 1

    def addDoubleGaussianSource(self, muX, muY, alpha, A): 
        # add double gaussian at pixel coordinates (muX, muY)
        # the alpha ratio is 1:2 and the amplitude ratio is 1:10 
        r = np.sqrt((self.Xpixels-muX)**2 + (self.Ypixels-muY)**2)
        sourceImage = 0.909*A*np.exp(-r**2/2/alpha**2) / (2*math.pi*alpha**2)
        alpha2 = alpha*2
        sourceImage += 0.091*A*np.exp(-r**2/2/alpha2**2) / (2*math.pi*alpha2**2)
        self.sourceImage = sourceImage
        self.sourceAdded = 1 
        self.image += sourceImage
        self.addDoubleGaussian += 1 

    def addNoise(self, sigmaNoise, addsourcenoise=0, gain=1.0):  
        # make a copy of noiseless input image and add gaussian noise
        self.imageNoNoise = np.copy(self.image)
        self.image += np.random.normal(0, sigmaNoise, self.image.shape)
        variance = 0*self.imageNoNoise + sigmaNoise**2 
        if (addsourcenoise):
            sourceVariance = np.copy(self.sourceImage)/gain
            self.image += np.random.normal(0, np.sqrt(sourceVariance), self.image.shape)
            variance += sourceVariance
        self.variance = variance


