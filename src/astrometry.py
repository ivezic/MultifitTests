from tools2Dgauss import *
from figPlots import *


### PLOTS ARE LISTED FIRST

def chi2plotMarginalAstro(chiPixXcen, chiPixCmod, chi2image, sigtrue, sigmaML, CmodML):

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.90, wspace=0.29, hspace=0.46)

    ## go from chi2 to ln(L): 
    # since L = exp(-chi2/2)
    # lnL = -1/2 * chi2
    lnL = -0.5*chi2image
    lnL -= lnL.max()
    lnL[lnL < -10] = -10  # truncate for clean plotting

    ## lnL image 
    ax = fig.add_axes([0.35, 0.35, 0.45, 0.6], xticks=[], yticks=[])
    ax.set_title('ln(L) image', fontsize=14)
    # pretty color map
    plt.imshow(lnL, origin='lower', 
           extent=(chiPixXcen[0], chiPixXcen[-1], chiPixCmod[0], chiPixCmod[-1]),
	   cmap=plt.cm.RdYlGn, aspect='auto')

    # colorbar
    cax = plt.axes([0.82, 0.35, 0.02, 0.6])
    cb = plt.colorbar(cax=cax)
    cb.set_label(r'$lnL(X_{centroid}, C_{mod})-lnLmax$', fontsize=14)
    plt.clim(np.min(lnL), np.max(lnL))

    # contours  WHY IS THIS NOT WORKING?? 
    plt.contour(chiPixXcen, chiPixCmod, convert_to_stdev(lnL),
            levels=(0.683, 0.955, 0.997), colors='k')

    # mark true values     
    ax.plot(sigtrue, 1000.0, 'o', color='red', alpha=0.75)
    # mark ML solution: (sigmaML, CmodML)
    ax.plot(sigmaML, CmodML, 'x', color='white', alpha=0.99, lw=35)

    # compute marginal projections
    p_XcenP = np.exp(lnL).sum(0)
    p_CmodP = np.exp(lnL).sum(1)

    ax1 = fig.add_axes([0.35, 0.1, 0.45, 0.23], yticks=[])
    ax1.plot(chiPixXcen, p_XcenP, '-k')
    ax1.set_xlabel(r'$X centroid$ (pixel)', fontsize=12)
    ax1.set_ylabel(r'$p(X)$', fontsize=12)
    ax1.set_xlim(np.min(chiPixXcen), np.max(chiPixXcen))

    ax2 = fig.add_axes([0.15, 0.35, 0.18, 0.6], xticks=[])
    ax2.plot(p_CmodP, chiPixCmod, '-k')
    ax2.set_xlabel(r'$p(C)$', fontsize=12)
    ax2.set_ylabel(r'$C$ (counts)', fontsize=12)
    ax2.set_xlim(ax2.get_xlim()[::-1])  # reverse x axis
    ax2.set_ylim(np.min(chiPixCmod), np.max(chiPixCmod))

    name = 'chi2plotMarginalAstro.png'
    if (name is None):
       	plt.show() 
    else:
        print 'saving plot to:', name
       	plt.savefig(name, bbox_inches='tight')

    return p_XcenP, p_CmodP


### COMPUTATIONS 


## make 2D chi2 image as a function of X position and counts 
def getChi2imageAstro(oneDpixels, image, muY, Bkgd, alpha, variance): 

    ### make chi2(Xcentroid, Cmod) image, find ML (min chi2) solution by brute force, 
    ### and return the best-fit model image and its parameters 

    ## make chi2 image
    # define the grid
    chiPixelsXcen = np.linspace(-1.0, 1.0, 1010)
    chiPixelsCmod = np.linspace(500, 1500, 1010)
    chi2 = lnLinit(chiPixelsCmod[:, None], chiPixelsXcen)  # lnLinit defined in tools2Dgauss.py
    # loop over all Xcen and Cmod
    chi2min = np.inf
    XcenML = -1.0
    CmodML = -1.0
    for i in range(0, chiPixelsXcen.size):
        Xcen = chiPixelsXcen[i]
        for j in range(0,chiPixelsCmod.size):
            Cmod = chiPixelsCmod[j]
            # the model image for current grid values of Xcen and Cmod
            model, src = gauss2Dastrom(Xcen, muY, alpha, Cmod, Bkgd, oneDpixels[:, np.newaxis], oneDpixels)
            thisChi2 = np.sum((image-model)**2/variance)
            chi2[j][i] = thisChi2   
            if (thisChi2 < chi2min): 
                chi2min = thisChi2
                XcenML = Xcen
                CmodML = Cmod

    # ML model
    bestModel, src = gauss2Dastrom(XcenML, muY, alpha, CmodML, Bkgd, oneDpixels[:, np.newaxis], oneDpixels)

    return chiPixelsXcen, chiPixelsCmod, chi2, bestModel, XcenML, CmodML, chi2min


def gauss2Dastrom(muX, muY, alpha, A, Bkgd, Xpixels, Ypixels):
    """2D circular gaussian + background"""
    r = np.sqrt((Xpixels-muX)**2 + (Ypixels-muY)**2)
    # make and set image to the background value
    image = np.empty(r.shape)
    image.fill(Bkgd)
    ## now add circular gaussian profile (area is normalized to A)    
    if (1):
        sourceImage = A*np.exp(-r**2/2/alpha**2) / (2*math.pi*alpha**2)
    else: 
        # double gaussian: 1:10 amplitude ratio and sigma2 = 2*sigma1
        sourceImage = 0.909*A*np.exp(-r**2/2/alpha**2) / (2*math.pi*alpha**2)
        alpha2 = alpha*2
        sourceImage += 0.091*A*np.exp(-r**2/2/alpha2**2) / (2*math.pi*alpha2**2)

    image += sourceImage
    return image, sourceImage


def addnoise(inimage, sigNoise, sourceImage, addsourcenoise=0): 
    # make a copy of input image and add gaussian noise
    image = np.copy(inimage)
    print 'addnoise: adding noise = ', sigNoise
    image += np.random.normal(0, sigNoise, image.shape)
    variance = 0*image + sigNoise**2 
    if (addsourcenoise):
        gain = 1.0
        sourceVariance = sourceImage/gain
        image += np.random.normal(0, np.sqrt(sourceVariance), image.shape)
        variance += sourceVariance
    return image, variance 
