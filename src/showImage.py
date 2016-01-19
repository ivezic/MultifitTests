import matplotlib.pyplot as plt
import numpy as np


def FourPanelStampPlot(oneDpixels, nonoise, psf, image, diffimage):

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.90, wspace=0.18, hspace=0.46)

    ## noiseless image
    ax = fig.add_subplot(221)
    ax.set_title('noiseless image', fontsize=14)
    plt.imshow(nonoise, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')
    plt.clim(np.min(nonoise), np.max(nonoise))
    plt.colorbar().set_label('')

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## psf image
    ax = fig.add_subplot(223)
    ax.set_title('source', fontsize=14)
    plt.imshow(psf, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')
    plt.clim(np.min(psf), np.max(psf))
    plt.colorbar().set_label('')

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## image with noise 
    ax = fig.add_subplot(222)
    ax.set_title('image (with noise)', fontsize=14)
    plt.imshow(image, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')

    plt.clim(np.min(image), np.max(image))
    plt.colorbar().set_label(r'$counts$', fontsize=14)

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## difference object - psf 
    ax = fig.add_subplot(224)
    ax.set_title('image - source', fontsize=14)
    plt.imshow(diffimage, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')

    plt.clim(np.min(diffimage), np.max(diffimage))
    plt.colorbar().set_label(r'$counts$', fontsize=14)

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    plt.show()
