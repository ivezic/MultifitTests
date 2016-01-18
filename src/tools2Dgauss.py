"""
Plot pixelized image of a 2D gaussian with noise
----------------------------------------
Modified code from astroML Book Figure 5.4
http://www.astroml.org/book_figures/chapter5/fig_likelihood_gaussian.html
"""
# Author: Jake VanderPlas, modified by Zeljko Ivezic
# License: BSD

import os
import numpy as np
import math
from matplotlib import pyplot as plt

# Hack to fix import issue in older versions of pymc
import scipy
import scipy.misc
scipy.derivative = scipy.misc.derivative
from scipy.special import gamma
from sklearn.neighbors import BallTree

# import pymc

from astroML.plotting.mcmc import plot_mcmc
from astroML.plotting.mcmc import convert_to_stdev
from astroML.decorators import pickle_results
from astroML import stats


import sys 
sys.path.append(os.path.abspath('/Users/ivezic/.ipython/profile_zi/startup/'))
# from zi import *

from plot2Dgauss import *
from figPlots import *



#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=8, usetex=True)


#----------------------------------------------------------------------
# Run the MCMC sampling for an arbitrary gaussian profile
def DoGaussFit(Xpixels, Ypixels, sigPSF, sigNoise, imageObs, PSF=0, seed=0):

    np.random.seed(seed)

    #------------------------------------------------------------------------
    # Set up MCMC sampling: this assumes "reasonable" values for input params
    # b0 = pymc.Uniform('b0', -20, 20, value=-20+50 * np.random.random())
    # strong prior: 
    b0 = pymc.Uniform('b0', -0.001, 0.001, value=-0.001+0.002 * np.random.random())
    A = pymc.Uniform('A', 500, 1500, value=1000 * np.random.random())
    # muG = pymc.Uniform('muG', -2, 2, value=-5+10 * np.random.random())
    # strong prior: 
    muG = pymc.Uniform('muG', -0.001, 0.001, value=-0.001+0.002 * np.random.random())
    if (PSF):
        log_sigma = pymc.Uniform('log_sigma', -0.001, 0.001, value=0)
    else:
        log_sigma = pymc.Uniform('log_sigma', -10, 10, value=0)

    # uniform prior on log(sigma)
    @pymc.deterministic
    def sigma(log_sigma=log_sigma):
        return np.exp(log_sigma)

    @pymc.deterministic
    def y_model(Xpixels=Xpixels, Ypixels=Ypixels, b0=b0, A=A, muG=muG, sigma=sigma):
        return gauss2D(muG, muG, sigma, A, b0, sigPSF, Xpixels, Ypixels)

    y = pymc.Normal('y', mu=y_model, tau=sigNoise ** -2, observed=True, value=imageObs)
    model = dict(b0=b0, A=A, muG=muG, log_sigma=log_sigma, sigma=sigma, y_model=y_model, y=y)

    @pickle_results('2DgaussFit.pkl')
    # small standard: 15000/5000
    def compute_MCMC_results(niter=15000, burn=5000):
        S = pymc.MCMC(model)
        S.sample(iter=niter, burn=burn)
        traces = [S.trace(s)[:] for s in ['b0', 'A', 'muG', 'sigma']]

        M = pymc.MAP(model)
        M.fit()
        fit_vals = (M.b0.value, M.A.value, M.muG.value, M.sigma.value)

        logp = get_logp(S, model)
        traces2 = np.vstack(traces)
        BF, dBF = estimate_bayes_factor(traces2, logp, r=0.02)
        # as r changes from 0.001 to 0.1, BF changes by 18 (relative to about -700), rms is ~1 

        return traces, fit_vals, BF, dBF

    traces, fit_vals, BF, dBF = compute_MCMC_results()
    return traces, fit_vals, BF, dBF


## copied from astroML book figure 5.24 (with a modification)
def estimate_bayes_factor(traces, logp, r=0.05, return_list=False):
    """Estimate the bayes factor using the local density of points"""

    # modified from the original 
    D = len(traces)
    N = traces[0].size

    # compute volume of a D-dimensional sphere of radius r
    Vr = np.pi ** (0.5 * D) / gamma(0.5 * D + 1) * (r ** D)

    # use neighbor count within r as a density estimator
    bt = BallTree(traces.T)
    count = bt.query_radius(traces.T, r=r, count_only=True)

    BF = logp + np.log(N) + np.log(Vr) - np.log(count)

    if return_list:
        return BF
    else:
        p25, p50, p75 = np.percentile(BF, [25, 50, 75])
        return p50, 0.7413 * (p75 - p25)


## copied from astroML book figure 5.24
def get_logp(S, model):
    """compute log(p) given a pyMC model"""
    M = pymc.MAP(model)
    traces = np.array([S.trace(s)[:] for s in S.stochastics])
    logp = np.zeros(traces.shape[1])
    for i in range(len(logp)):
        logp[i] = -M.func(traces[:, i])
    return logp




def DoGaussFit5(Xpixels, Ypixels, sigPSF, sigNoise, imageObs, PSF=0, seed=0):

    np.random.seed(seed)

    #------------------------------------------------------------------------
    # Set up MCMC sampling: this assumes "reasonable" values for input params
    b0 = pymc.Uniform('b0', -20, 20, value=-20+50 * np.random.random())
    # strong prior: 
    # b0 = pymc.Uniform('b0', -0.001, 0.001, value=-0.001+0.002 * np.random.random())
    # b0 = pymc.Uniform('b0', 0.495, 0.505, value=0.5+0.001 * np.random.random())
    A = pymc.Uniform('A', 500, 1500, value=1000 * np.random.random())
    muGx = pymc.Uniform('muGx', -2, 2, value=-5+10 * np.random.random())
    muGy = pymc.Uniform('muGy', -2, 2, value=-5+10 * np.random.random())
    # strong priors: 
    # muGx = pymc.Uniform('muGx', -0.001, 0.001, value=-0.001+0.002 * np.random.random())
    # muGy = pymc.Uniform('muGy', -0.001, 0.001, value=-0.001+0.002 * np.random.random())
    # muGy = pymc.Uniform('muGy', 0.495, 0.505, value=0.5+0.001 * np.random.random())
    if (PSF):
        log_sigma = pymc.Uniform('log_sigma', -0.001, 0.001, value=0)
    else:
        log_sigma = pymc.Uniform('log_sigma', -10, 10, value=0)

    # uniform prior on log(sigma)
    @pymc.deterministic
    def sigma(log_sigma=log_sigma):
        return np.exp(log_sigma)

    @pymc.deterministic
    def y_model(Xpixels=Xpixels, Ypixels=Ypixels, b0=b0, A=A, muGx=muGx, muGy=muGy, sigma=sigma):
        return gauss2D(muGx, muGy, sigma, A, b0, sigPSF, Xpixels, Ypixels)

    print 'sigNoise=', sigNoise
    y = pymc.Normal('y', mu=y_model, tau=sigNoise ** -2, observed=True, value=imageObs)
    model = dict(b0=b0, A=A, muGx=muGx, muGy=muGy, log_sigma=log_sigma, sigma=sigma, y_model=y_model, y=y)

    @pickle_results('2DgaussFit5.pkl')
    # small standard: 15000/5000
    # large standard: 50000/15000
    def compute_MCMC_results(niter=15000, burn=5000):
        S = pymc.MCMC(model)
        S.sample(iter=niter, burn=burn)
        traces = [S.trace(s)[:] for s in ['b0', 'A', 'muGx', 'muGy', 'sigma']]

        M = pymc.MAP(model)
        M.fit()
        fit_vals = (M.b0.value, M.A.value, M.muGx.value, M.muGy.value, M.sigma.value)

        logp = get_logp(S, model)
        traces2 = np.vstack(traces)
        BF, dBF = estimate_bayes_factor(traces2, logp, r=0.02)
        # as r changes from 0.001 to 0.1, BF changes by 18 (relative to about -700), rms is ~1 

        return traces, fit_vals, BF, dBF

    traces, fit_vals, BF, dBF = compute_MCMC_results()
    return traces, fit_vals, BF, dBF



def gauss2D(muX, muY, sig, A, B, sigPSF, Xpixels, Ypixels):
    """2D circular gaussian + background"""
    r = np.sqrt((Xpixels-muX)**2 + (Ypixels-muY)**2)
    # make and set image to the background value
    image = np.empty(r.shape)
    image.fill(B)
    ## now add circular gaussian profile (area is normalized to A)
    # source gauss convolved with single-gauss PSF  
    sigConvSquare = sig**2 + sigPSF**2
    image += A*np.exp(-r**2/2/sigConvSquare) / (2*math.pi*sigConvSquare)
    return image

def addnoise(inimage, sigNoise, addsourcenoise=0): 
    # make a copy of input image and add gaussian noise
    image = np.copy(inimage)
    print 'addnoise: adding noise = ', sigNoise
    image += np.random.normal(0, sigNoise, image.shape)
    if (addsourcenoise):
        gain = 1.0
        #image += np.random.normal(0, np.sqrt(inimage/gain), image.shape)
    return image


def doImageStats(oneDpixels, sigmaPSF, sigmaNoise, psf, nonoise, image, verbose=0, fitModel=1): 
    ## compute psf counts, model counts, and errors
    psfNorm = psf / np.sum(psf)
    wPSF = psfNorm / np.sum(psfNorm**2)
    CountPSF = np.sum(image*wPSF)
    # effective number of pixels
    neff = 1.0 / np.sum(psfNorm**2)
    # error for CountPSF, assuming that source noise is negligible 
    CountPSFerr = sigmaNoise * np.sqrt(neff) 
    SNRpsf = CountPSF/CountPSFerr
    # MCMC fit of gaussian
    if (fitModel):
        # fit single gauss + background to this image
        os.remove('2DgaussFit.pkl')
        print '       going to fit with sigmaNoise=', sigmaNoise
        traces, fit_vals, BF, dBF = DoGaussFit(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigmaNoise, image)
        b0Best = np.mean(traces[0])
        ABest = np.mean(traces[1])
        muGBest = np.mean(traces[2])
        sigmaBest = np.mean(traces[3])
        ABestRMS = np.std(traces[1])
        sigmaBestRMS = np.std(traces[3])
        BestModel = gauss2D(muGBest, muGBest, sigmaBest, ABest, b0Best, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        modelNorm = BestModel / np.sum(BestModel)
    else:
        # assume that the model is nonoise image itself
        modelNorm = nonoise / np.sum(nonoise)
        ABest = -1.0
        sigmaBest = -1.0 
        ABestRMS = -1.0
        sigmaBestRMS = -1.0 
        traces = 0 
        fit_vals = 0 

    # CountModel and its error
    wModel = modelNorm / np.sum(modelNorm**2)
    CountModel = np.sum(image*wModel)
    neffMod = 1.0 / np.sum(modelNorm**2)
    CountModErr = sigmaNoise * np.sqrt(neffMod) 
    SNRmod = CountModel/CountModErr
    # mpsf-mmod
    dmag = -2.5*np.log10(CountPSF/CountModel)

    if (verbose):
        print '   CountPSF = ', CountPSF
        print 'CountPSFerr = ', CountPSFerr
        print '     SNRpsf = ', SNRpsf 
        print ' CountModel = ', CountModel
        print 'CountModErr = ', CountModErr
        print '     SNRmod = ', SNRmod 
        print '       neff = ', neff 
        print '    neffMod = ', neffMod
        print '  mPSF-mMod = ', dmag
        if (fitModel):
            print '     Best-fit C = ', ABest, '+- ', ABestRMS
            print ' Best-fit sigma = ', sigmaBest, '+- ', sigmaBestRMS

    return neff, CountPSF, CountModel, neffMod, ABest, sigmaBest, ABestRMS, sigmaBestRMS, traces, fit_vals 


def doImageStats5(oneDpixels, sigmaPSF, sigmaNoise, psf, nonoise, image, verbose=0, fitModel=1): 
    ## compute psf counts, model counts, and errors
    psfNorm = psf / np.sum(psf)
    wPSF = psfNorm / np.sum(psfNorm**2)
    CountPSF = np.sum(image*wPSF)
    # effective number of pixels
    neff = 1.0 / np.sum(psfNorm**2)
    # error for CountPSF, assuming that source noise is negligible 
    CountPSFerr = sigmaNoise * np.sqrt(neff) 
    SNRpsf = CountPSF/CountPSFerr
    # MCMC fit of gaussian
    if (fitModel):
        # fit single gauss + background to this image
        os.remove('2DgaussFit.pkl')
        print '       going to fit with sigmaNoise=', sigmaNoise
        traces, fit_vals = DoGaussFit5(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigmaNoise, image)
        b0Best = np.mean(traces[0])
        b0BestRMS = np.std(traces[0])
        ABest = np.mean(traces[1])
        ABestRMS = np.std(traces[1])
        muGxBest = np.mean(traces[2])
        muGxBestRMS = np.std(traces[2])
        muGyBest = np.mean(traces[3])
        muGyBestRMS = np.std(traces[3])
        sigmaBest = np.mean(traces[4])
        sigmaBestRMS = np.std(traces[4])
        BestModel = gauss2D(muGxBest, muGyBest, sigmaBest, ABest, b0Best, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        modelNorm = BestModel / np.sum(BestModel)
    else:
        # assume that the model is nonoise image itself
        modelNorm = nonoise / np.sum(nonoise)
        ABest = -1.0
        sigmaBest = -1.0 
        ABestRMS = -1.0
        sigmaBestRMS = -1.0 
        traces = 0 
        fit_vals = 0 

    # CountModel and its error
    wModel = modelNorm / np.sum(modelNorm**2)
    CountModel = np.sum(image*wModel)
    neffMod = 1.0 / np.sum(modelNorm**2)
    CountModErr = sigmaNoise * np.sqrt(neffMod) 
    SNRmod = CountModel/CountModErr
    # mpsf-mmod
    dmag = -2.5*np.log10(CountPSF/CountModel)

    if (verbose):
        print '   CountPSF = ', CountPSF
        print 'CountPSFerr = ', CountPSFerr
        print '     SNRpsf = ', SNRpsf 
        print ' CountModel = ', CountModel
        print 'CountModErr = ', CountModErr
        print '     SNRmod = ', SNRmod 
        print '       neff = ', neff 
        print '    neffMod = ', neffMod
        print '  mPSF-mMod = ', dmag
        if (fitModel):
            print '     Best-fit B = ', b0Best, '+- ', b0BestRMS
            print '     Best-fit C = ', ABest, '+- ', ABestRMS
            print ' Best-fit sigma = ', sigmaBest, '+- ', sigmaBestRMS
            print '  Best-fit muGx = ', muGxBest, '+- ', muGxBestRMS
            print '  Best-fit muGy = ', muGyBest, '+- ', muGyBestRMS

    return neff, CountPSF, CountModel, neffMod, ABest, sigmaBest, ABestRMS, sigmaBestRMS, traces, fit_vals 



# wrappers of above procs
def make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigmaNoise, makeplot=1):

    # define the (square) grid
    oneDpixels = np.linspace(-7, 7, 15)

    ## make psf (sigtrue=0) 
    psf = gauss2D(muXtrue, muYtrue, 0, Atrue, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
    ## make noiseless image (convolved with psf, size given by 1Dpixels) 
    nonoise = gauss2D(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
    ## now add noise
    addsourcenoise = 0
    image = addnoise(nonoise, sigmaNoise, addsourcenoise) 

    ## difference object - psf
    diffimage = image - psf 

    # plot
    if (makeplot):
        if (makeplot == 1):
            FourPanelPlot(oneDpixels, nonoise, psf, image, diffimage)
        if (makeplot > 1):
            ## do computation
            chiPixSig, chiPixCmod, chi2im, bestModel, sigmaML, CmodML, chi2min = \
                getChi2image(oneDpixels, image, muXtrue, muYtrue, Btrue, sigmaPSF, sigmaNoise)
            print 'sigmaML = ', sigmaML
            print ' CmodML = ', CmodML
            print 'chi2min = ', chi2min
            # compute Cmod from bestModel and compare to CmodML
            modelNorm = bestModel / np.sum(bestModel)
            wModel = modelNorm / np.sum(modelNorm**2)
            CountModel = np.sum(image*wModel)
            print 'Cmod Direct = ', CountModel

            # chi2-based plots
            chi2plot(oneDpixels, image, bestModel, chiPixSig, chiPixCmod, chi2im, sigtrue, sigmaML, CmodML)
            chi2plotMarginal(chiPixSig, chiPixCmod, chi2im, sigtrue, sigmaML, CmodML)

    return oneDpixels, nonoise, psf, image, diffimage


def lnLinit(yPixels, xPixels):
    lnL = np.zeros(xPixels.shape, dtype=float) - 0 * np.log(yPixels)
    return lnL


def getChi2image(oneDpixels, image, muX, muY, B, sigmaPSF, sigmaNoise): 

    ### make chi2(sigma_g, Cmod) image, find ML solution, and return best-fit model image 

    ## make chi2 image
    # define the grid
    chiPixelsSigma = np.linspace(0, 2.5, 101)
    chiPixelsCmod = np.linspace(500, 1500, 101)
    # chiPixelsSigma = np.linspace(0.0, 5.00, 201)
    # chiPixelsCmod = np.linspace(0.0, 2000.0, 201)
    lnL = lnLinit(chiPixelsCmod[:, None], chiPixelsSigma)
    # loop over all sigma and Cmod
    chi2min = np.inf
    sigmaML = -1.0
    CmodML = -1.0
    for i in range(0, chiPixelsSigma.size):
        sigma = chiPixelsSigma[i]
        for j in range(0,chiPixelsCmod.size):
            Cmod = chiPixelsCmod[j]
            # model image
            model = gauss2D(muX, muY, sigma, Cmod, B, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
            ss =  np.sum((image-model)/sigmaNoise)
            thisChi2 = np.sum(((image-model)/sigmaNoise)**2)
            lnL[j][i] = thisChi2
            if (thisChi2 < chi2min): 
                chi2min = thisChi2
                sigmaML = sigma
                CmodML = Cmod
    # ML model
    bestModel = gauss2D(muX, muY, sigmaML, CmodML, B, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
    print 'getChi2image: chi2 computed with sigmaNoise=', sigmaNoise

    return chiPixelsSigma, chiPixelsCmod, lnL, bestModel, sigmaML, CmodML, chi2min




def analyzeManyImages(N=10, sigtrue=0.0, sigma_m=2.0, fitModel=1):

    ### test errors for psf and model counts, and their correlation 

    ## source profile parameters BEFORE convolution with the psf 
    # units are pixels 
    Btrue = 0.0        # background
    Atrue = 1000.0     # total number of counts, Image = A * profile, where profile integrates to 1
    muXtrue = 0.0      # X location of the maximum count
    muYtrue = 0.0      # Y location of the maximum count

    ## point spread function parameters (assuming single gaussian,  
    sigmaPSF = 1.5     # "psf width": corresponds to FWHM=2.355*sigmaPSF pixels, peak count = A * 0.07 for sigmaPSF=1.5

    # arrays for return
    mpsfArr = {}
    mmodArr = {}
    fitArr = {}
    for iter in range(0,N):
        print '====> doManyImages ITERATION: ', iter, '   for noise=', sigma_m, ' and sigtrue=', sigtrue 
        oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigma_m, 0)
        mpsf, mmod, fit, traces, fit_vals = analyzeImages(oneDpixels, sigmaPSF, sigma_m, psf, nonoise, image, fitModel=fitModel)
        mpsfArr[iter] = mpsf
        mmodArr[iter] = mmod
        fitArr[iter] = fit 
       
    return mpsfArr, mmodArr, fitArr



def analyzeImages(oneDpixels, sigmaPSF, sigmaNoise, psf, nonoise, image, fitModel=1): 

    ## compute psf counts and errors
    psfNorm = psf / np.sum(psf)
    wPSF = psfNorm / np.sum(psfNorm**2)
    CountPSF = np.sum(image*wPSF)
    # effective number of pixels
    neff = 1.0 / np.sum(psfNorm**2)
    # error for CountPSF, assuming that source noise is negligible 
    CountPSFerr = sigmaNoise * np.sqrt(neff) 
    SNRpsf = CountPSF/CountPSFerr

    ## compute model counts and errors
    if (fitModel):
        # fit single gauss + background to this image
        os.remove('2DgaussFit.pkl')
        print '       going to fit with sigmaNoise=', sigmaNoise
        # MCMC fit of gaussian
        traces, fit_vals, BF, dBF = DoGaussFit(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigmaNoise, image)
        b0Best = np.mean(traces[0])
        ABest = np.mean(traces[1])
        muGBest = np.mean(traces[2])
        sigmaBest = np.mean(traces[3])
        ABestRMS = np.std(traces[1])
        sigmaBestRMS = np.std(traces[3])
        BestModel = gauss2D(muGBest, muGBest, sigmaBest, ABest, b0Best, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        modelNorm = BestModel / np.sum(BestModel)
        ## direct chi2
        # for psf
        chi2PSF = np.sum(((image-CountPSF/1000*psfNorm)/sigmaNoise)**2)
        # for model, with true profile
        nonoiseNorm = nonoise / np.sum(nonoise)
        w = nonoiseNorm / np.sum(nonoiseNorm**2)
        CountModelTrue= np.sum(image*w)
        chi2modTrue = np.sum(((image-CountModelTrue/1000*nonoiseNorm)/sigmaNoise)**2)
        # for best-fit model
        wModel = modelNorm / np.sum(modelNorm**2)
        CountModelBest = np.sum(image*wModel)
        chi2modBest = np.sum(((image-CountModelBest/1000*modelNorm)/sigmaNoise)**2)
        ## and now find scatter of chi2modBest and return as chi2modBestRMS 
        chiAux = 0.0 + 0*traces[0]
        for i in range(0,traces[0].size):
            b0X = np.mean(traces[0])
            ABestX = np.mean(traces[1])
            muGX = np.mean(traces[2])
            sigmaBestX = np.mean(traces[3])
            BestModelX = gauss2D(muGX, muGX, sigmaBestX, ABestX, b0X, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
            modelNormX = BestModelX / np.sum(BestModelX)
            wModelX = modelNormX / np.sum(modelNormX**2)
            CountModelBestX = np.sum(image*wModelX)
            chiAux[i] = np.sum(((image-CountModelBestX/1000*modelNormX)/sigmaNoise)**2)

        chi2modBestRMS = np.std(chiAux)

    else:
        # for debugging: 
        # assume that the model is nonoise image itself
        modelNorm = nonoise / np.sum(nonoise)
        ABest = -1.0
        sigmaBest = -1.0 
        ABestRMS = -1.0
        sigmaBestRMS = -1.0 
        chi2PSF = -999
        chi2modTrue = -999
        chi2modBest = -999
        chi2modBestRMS = -999
        traces = 0 
        fit_vals = 0 

    # CountModel and its error
    wModel = modelNorm / np.sum(modelNorm**2)
    CountModel = np.sum(image*wModel)
    neffMod = 1.0 / np.sum(modelNorm**2)
    CountModErr = sigmaNoise * np.sqrt(neffMod) 
    SNRmod = CountModel/CountModErr
 
    # pack the results and return 
    mpsf = {}
    mmod = {}
    fit = {}

    # based on profiles
    mpsf['neff'] = neff
    mpsf['C'] = CountPSF
    mpsf['Cerr'] = CountPSFerr
    mmod['neff'] = neffMod
    mmod['C'] = CountModel
    mmod['Cerr'] = CountModErr

    # based on MCMC fit (amplitude and width)
    fit['A'] = ABest
    fit['Arms'] = ABestRMS
    fit['SIG'] = sigmaBest
    fit['SIGrms'] = sigmaBestRMS
    fit['chi2PSF'] = chi2PSF
    fit['chi2modTrue'] = chi2modTrue
    fit['chi2modBest'] = chi2modBest
    fit['chi2modBestRMS'] = chi2modBestRMS
    return mpsf, mmod, fit, traces, fit_vals 
 


def makeVectors(N, mpsf, mmod, fit):

    # arrays for return
    dmag = np.zeros(N)
    Cpsf = np.zeros(N)
    CpsfErr = np.zeros(N)
    neffPsf = np.zeros(N)
    Cmod = np.zeros(N)
    CmodErr = np.zeros(N)
    neffMod = np.zeros(N)
    sigBest = np.zeros(N)
    sigBestRMS = np.zeros(N)
    ABest = np.zeros(N)
    ABestRMS = np.zeros(N)
    chi2PSF = np.zeros(N)
    chi2modTrue = np.zeros(N)
    chi2modBest = np.zeros(N)
    chi2modBestRMS = np.zeros(N)

    for i in range(0,N):
        Cpsf[i] = mpsf[i]['C']
        CpsfErr[i] = mpsf[i]['Cerr']
        neffPsf[i] = mpsf[i]['neff']
        Cmod[i] = mmod[i]['C']
        CmodErr[i] = mmod[i]['Cerr']
        neffMod[i] = mmod[i]['neff']
        sigBest[i] = fit[i]['SIG']
        sigBestRMS[i] = fit[i]['SIGrms']
        ABest[i] = fit[i]['A']
        ABestRMS[i] = fit[i]['Arms']
        chi2PSF[i] = fit[i]['chi2PSF']
        chi2modTrue[i] = fit[i]['chi2modTrue']
        chi2modBest[i] = fit[i]['chi2modBest'] 
        chi2modBestRMS[i] = fit[i]['chi2modBestRMS'] 

    dmag = -2.5*np.log10(Cpsf/Cmod)
    aVec = []
    for vec in (dmag, Cpsf, CpsfErr, neffPsf, Cmod, CmodErr, neffMod, sigBest, sigBestRMS, ABest, ABestRMS, chi2PSF, chi2modTrue, chi2modBest, chi2modBestRMS):
        aVec.append(vec)

    return aVec




def doManyImages(N, sigtrue=0.0, sigma_m=22.0):

    ### test errors for psf and model counts, and their correlation 

    ## source profile parameters BEFORE convolution with the psf 
    # units are pixels 
    Btrue = 0.0        # background
    Atrue = 1000.0     # total number of counts, Image = A * profile, where profile integrates to 1
    muXtrue = 0.0      # X location of the maximum count
    muYtrue = 0.0      # Y location of the maximum count
    # sigtrue = 0.75     # for profile=gaussian, width of the gaussian BEFORE convolution

    ## point spread function parameters (assuming single gaussian,  
    sigmaPSF = 1.5     # "psf width": corresponds to FWHM=2.355*sigmaPSF pixels, peak count = A * 0.07 for sigmaPSF=1.5
    # measurement noise for counts (per pixel w/o source contribution) 
    # sigma_m = 2.2      # measurement uncertainty for counts (for A=1000, 38 is roughly SNR=5 for psf mag) 

    # arrays for return
    dmag = np.zeros(N)
    Cpsf = np.zeros(N)
    CpsfErr = np.zeros(N)
    Cmod = np.zeros(N)
    CmodErr = np.zeros(N)
    SNRpsf = np.zeros(N)
    SNRmod = np.zeros(N)
    Cbest = np.zeros(N)
    CbestRMS = np.zeros(N)
    sigBest = np.zeros(N)
    sigBestRMS = np.zeros(N)

    for iter in range(0,N):
        print '====> doManyImages ITERATION: ', iter, '   for noise=', sigma_m
        oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigma_m, 0)
        neff, Cpsf[iter], Cmod[iter], neffMod, Cbest[iter], sigBest[iter], CbestRMS[iter], sigBestRMS[iter], traces, fit_vals = doImageStats(oneDpixels, sigmaPSF, sigma_m, psf, nonoise, image)
        # error for CountPSF, assuming that source noise is negligible 
        CpsfErr[iter]  = sigma_m * np.sqrt(neff) 
        SNRpsf[iter]  = Cpsf[iter]/CpsfErr[iter] 
        CmodErr[iter] = sigma_m * np.sqrt(neffMod) 
        SNRmod[iter] = Cmod[iter]/CmodErr[iter] 
        dmag[iter] = -2.5*np.log10(Cpsf[iter]/Cmod[iter])

    return dmag, Cpsf, CpsfErr, SNRpsf, Cmod, CmodErr, SNRmod, Cbest, CbestRMS, sigBest, sigBestRMS



def doAnalytics(N, sigma_m=0.0):

    ### test errors for psf and model counts, and their correlation 

    ## source profile parameters BEFORE convolution with the psf 
    # units are pixels 
    Btrue = 0.0        # background
    Atrue = 1000.0     # total number of counts, Image = A * profile, where profile integrates to 1
    muXtrue = 0.0      # X location of the maximum count
    muYtrue = 0.0      # Y location of the maximum count
    sigmaPSF = 1.5     # "psf width": corresponds to FWHM=2.355*sigmaPSF pixels, peak count = A * 0.07 for sigmaPSF=1.5
    # measurement noise for counts (per pixel w/o source contribution) 
    # sigma_m = 2.2      # measurement uncertainty for counts (for A=1000, 38 is roughly SNR=5 for psf mag) 

    sigtrueVec = np.linspace(0.0, 2.0, N)

    # arrays for return
    dmag = np.zeros(N)
    Cpsf = np.zeros(N)
    CpsfErr = np.zeros(N)
    Cmod = np.zeros(N)
    CmodErr = np.zeros(N)
    SNRpsf = np.zeros(N)
    SNRmod = np.zeros(N)
    Cbest = np.zeros(N)
    CbestRMS = np.zeros(N)
    sigBest = np.zeros(N)
    sigBestRMS = np.zeros(N)
    neffMod = np.zeros(N)
    neffPst = np.zeros(N)
    neffXXX = np.zeros(N)

    for iter in range(0,N):
        sigtrue = sigtrueVec[iter]
        print '====> doManyImages ITERATION: ', iter, '   for noise=', sigma_m
        oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigma_m, 0)
        neff, Cpsf[iter], Cmod[iter], neffMod, Cbest[iter], sigBest[iter], CbestRMS[iter], sigBestRMS[iter], traces, fit_vals = doImageStats(oneDpixels, sigmaPSF, sigma_m, psf, nonoise, image)
        # error for CountPSF, assuming that source noise is negligible 
        CpsfErr[iter]  = sigma_m * np.sqrt(neff) 
        SNRpsf[iter]  = Cpsf[iter]/CpsfErr[iter] 
        CmodErr[iter] = sigma_m * np.sqrt(neffMod) 
        SNRmod[iter] = Cmod[iter]/CmodErr[iter] 
        dmag[iter] = -2.5*np.log10(Cpsf[iter]/Cmod[iter])

    return dmag, Cpsf, CpsfErr, SNRpsf, Cmod, CmodErr, SNRmod, Cbest, CbestRMS, sigBest, sigBestRMS





def analyzeSNRbehavior(sigtrue = 0.0): 
    # sigtrue is the intrinsic width of source profile (single gaussian) 

    # number of random samples at each SNR 
    Niter = 10
    # number of steps in SNR
    NSNR = 2 
    # grid of measurement noise for counts (per pixel w/o source contribution) 
    sigmaNoise = np.linspace(2, 10, NSNR)
        
    # arrays for return
    dmag = np.zeros(NSNR)
    dmagStd = np.zeros(NSNR)
    Cpsf = np.zeros(NSNR)
    CpsfErr = np.zeros(NSNR)
    Cmod = np.zeros(NSNR)
    CmodErr = np.zeros(NSNR)
    SNRpsf = np.zeros(NSNR)
    SNRmod = np.zeros(NSNR)
    Cbest = np.zeros(NSNR)
    CbestStd = np.zeros(NSNR)
    sigBest = np.zeros(NSNR)
    sigBestStd = np.zeros(NSNR)

    for i, sigma_m in zip(range(0,NSNR), sigmaNoise):   
        print 'working on sigmaNoise = ', sigma_m, 'iteration=',i
        dmag1, Cpsf1, CpsfErr1, SNRpsf1, Cmod1, CmodErr1, SNRmod1, Cbest1, sigBest1 = doManyImages(Niter,sigtrue,sigma_m)
        dmag[i] = np.median(dmag1)
        dmagStd[i] = np.std(dmag1)
        Cpsf[i] = np.median(Cpsf1)
        CpsfErr[i] = np.median(CpsfErr1)
        Cmod[i] = np.median(Cmod1)
        CmodErr[i] = np.median(CmodErr1)
        SNRpsf[i] = np.median(SNRpsf1)
        SNRmod[i] = np.median(SNRmod1)
        Cbest[i] = np.median(Cbest1)
        CbestStd[i] = np.std(Cbest1)
        sigBest[i] = np.median(sigBest1)
        sigBestStd[i] = np.std(sigBest1)


    return dmag, dmagStd, Cpsf, CpsfErr, SNRpsf, Cmod, CmodErr, SNRmod, Cbest, CbestStd, sigBest, sigBestStd


def plot_mcmcZI(traces, labels=None, limits=None, true_values=None,
              fig=None, contour=True, scatter=False,
              levels=[0.683, 0.955], bins=20,
              bounds=[0.08, 0.08, 0.95, 0.95], **kwargs):
    """Plot a grid of MCMC results

    Parameters
    ----------
    traces : array_like

        the MCMC chain traces.  shape is [Ndim, Nchain]
    labels : list of strings (optional)
        if specified, the label associated with each trace
    limits : list of tuples (optional)
        if specified, the axes limits for each trace
    true_values : list of floats (optional)
        if specified, the true value for each trace (will be indicated with
        an 'X' on the plot)
    fig : matplotlib.Figure (optional)
        the figure on which to draw the axes.  If not specified, a new one
        will be created.
    contour : bool (optional)
        if True, then draw contours in each subplot.  Default=True.
    scatter : bool (optional)
        if True, then scatter points in each subplot.  Default=False.
    levels : list of floats
        the list of percentile levels at which to plot contours.  Each
        entry should be between 0 and 1
    bins : int, tuple, array, or tuple of arrays
        the binning parameter passed to np.histogram2d.  It is assumed that
        the point density is constant on the scale of the bins
    bounds : list of floats
        the bounds of the set of axes used for plotting

    additional keyword arguments are passed to scatter() and contour()

    Returns
    -------
    axes_list : list of matplotlib.Axes instances
        the list of axes created by the routine
    """
    # Import here so that testing with Agg will work
    from matplotlib import pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(8, 8))

    if limits is None:
        limits = [(t.min(), t.max()) for t in traces]

    if labels is None:
        labels = ['' for t in traces]

    num_traces = len(traces)

    bins = [np.linspace(limits[i][0], limits[i][1], bins + 1)
            for i in range(num_traces)]

    xmin, xmax = bounds[0], bounds[2]
    ymin, ymax = bounds[1], bounds[3]

    dx = (xmax - xmin) * 1. / (num_traces - 1)
    dy = (ymax - ymin) * 1. / (num_traces - 1)

    axes_list = []

    for j in range(1, num_traces):
        for i in range(j):
            ax = fig.add_axes([xmin + i * dx,
                               ymin + (num_traces - 1 - j) * dy,
                               dx, dy])

            if scatter:
                plt.scatter(traces[i], traces[j], **kwargs)

            if contour:
                H, xbins, ybins = np.histogram2d(traces[i], traces[j],
                                                 bins=(bins[i], bins[j]))

                H[H == 0] = 1E-16
                Nsigma = convert_to_stdev(np.log(H))

                ax.contour(0.5 * (xbins[1:] + xbins[:-1]),
                           0.5 * (ybins[1:] + ybins[:-1]),
                           Nsigma.T, levels=levels, **kwargs)

            if i == 0:
                ax.set_ylabel(labels[j])
            else:
                ax.yaxis.set_major_formatter(plt.NullFormatter())

            if j == num_traces - 1:
                ax.set_xlabel(labels[i])
            else:
                ax.xaxis.set_major_formatter(plt.NullFormatter())

            if true_values is not None:
                ax.plot(limits[i], [true_values[j], true_values[j]],
                        ':k', lw=1)
                ax.plot([true_values[i], true_values[i]], limits[j],
                        ':k', lw=1)

            ax.set_xlim(limits[i])
            ax.set_ylim(limits[j])

            axes_list.append(ax)

    return axes_list



def plotSingleCase(b0_true, A_true, muG_true, sigma_true, sigma_m, sigmaPSF, traces, fit_vals):

    #------------------------------------------------------------
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.45, right=0.95, hspace=0.05, wspace=0.05)

    labels = ['$Background$', '$Amplitude$', '$centroid$', r'$\sigma$']
    limits = [(-2, 2), (500.0, 1500.0), (-0.45, 0.45), (0.0, 1.8)]   # for sigtrue
    # limits = [(-2, 2), (500.0, 1500.0), (-0.45, 0.45), (1.0, 2.8)]     # for sigobs
    # limits = [(-0.4, 0.4), (8.0, 12.0), (-0.2, 0.2), (0.2, 0.5)]
    true = [b0_true, A_true, muG_true, sigma_true]
    print 'true=', true
    # This function plots multiple panels with the traces (ZI version for debugging) 
    axes_list = plot_mcmcZI(traces, labels=labels, limits=limits, true_values=true, fig=fig, 
          levels=[0.683, 0.955, 0.997], bounds=[0.10, 0.08, 0.95, 0.95], bins=25, colors='k')

    # Plot the model fit
    ax = fig.add_axes([0.5, 0.7, 0.45, 0.25])

    plt.show()

    return traces, fit_vals



def plotSingleCase5(oneDpixels, image, b0_true, A_true, muGx_true, muGy_true, sigma_true, sigma_m, sigmaPSF, traces, fit_vals):

    #------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.45, right=0.95, hspace=0.05, wspace=0.05)

    labels = ['$Background$', '$Amplitude$', '$Xcentroid$', '$Ycentroid$', r'$\sigma$']
    limits = [(-4.8, 4.8), (400.0, 1600.0), (-0.99, 0.99), (-0.99, 0.99), (0.0, 1.99)]  
    true = [b0_true, A_true, muGx_true,  muGy_true, sigma_true]
    print 'true=', true
    # This function plots multiple panels with the traces (ZI version for debugging) 
    axes_list = plot_mcmcZI(traces, labels=labels, limits=limits, true_values=true, fig=fig, 
          levels=[0.683, 0.955, 0.997], bounds=[0.10, 0.08, 0.95, 0.95], bins=25, colors='k')

    # Plot the data image, best-fit model, and residual
    ax = fig.add_axes([0.5, 0.7, 0.45, 0.25])

    plt.show()

    return traces, fit_vals



def plot2chains(Xchain, Ychain, X0True, Y0true, Xlabel='X', Ylabel='Y', limits=0, bins=25):

    #------------------------------------------------------------
    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.45, right=0.95, hspace=0.05, wspace=0.05)

    if (not limits):
        limits = [(np.min(Xchain), np.max(Xchain)), (np.min(Ychain), np.max(Ychain))]
    # true = [X0True, Y0True]
    true = [1000, 0.75]
    traces = [Xchain, Ychain]
    labels = [Xlabel, Ylabel]

    # This function plots multiple panels with the traces (ZI version for debugging) 
    axes_list = plot_mcmcZI(traces, labels=labels, limits=limits, true_values=true, fig=fig, 
          levels=[0.683, 0.955, 0.997], bounds=[0.10, 0.08, 0.95, 0.95], bins=bins, colors='k')

    # Plot the model fit
    # ax = fig.add_axes([0.5, 0.7, 0.45, 0.25])

    plt.show()

    return



#------------------------------------------------------------
## source profile parameters BEFORE convolution with the psf 
# units are pixels 
Btrue = 0.0        # background
Atrue = 1000.0     # total number of counts, Image = A * profile, where profile integrates to 1
muXtrue = 0.0      # X location of the maximum count
muYtrue = 0.0      # Y location of the maximum count
sigtrue = 0.75     # for profile=gaussian, width of the gaussian BEFORE convolution
## point spread function parameters (assuming single gaussian,  
sigmaPSF = 1.5     # "psf width": corresponds to FWHM=2.355*sigmaPSF pixels, peak count = A * 0.07 for sigmaPSF=1.5
# measurement noise for counts (per pixel w/o source contribution) 
sigma_m = 10.0      # measurement uncertainty for counts (for A=1000, 38 is roughly SNR=5 for psf mag) 

# make images and plots 
# oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigma_m)

# only plotting:
# FourPanelPlot(oneDpixels, nonoise, psf, image, diffimage)

# basic stats 
# neff, CountPSF, CountModel, neffMod, Cbest, sigmaBest = doImageStats(oneDpixels, sigmaPSF, sigma_m, psf, nonoise, image, 1)

## fit 
# os.remove('2DgaussFit.pkl')
# traces, fit_vals = DoGaussFit(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigma_m, image)

#traces2, fit_vals2 = plotSingleCase(Btrue, Atrue, muXtrue, sigtrue, sigma_m, sigmaPSF, traces, fit_vals)

    

# neff, CountPSF, CountModel, neffMod, Cbest, sigmaBest, ABestRMS, sigmaBestRMS, traces, fit_vals  = doImageStats(oneDpixels, sigmaPSF, sigma_m, psf, nonoise, image, 1)
# traces2, fit_vals2 = plotSingleCase(Btrue, Atrue, muXtrue, sigtrue, sigma_m, sigmaPSF, traces, fit_vals)


#------------------------------------------------------------
# analyze many vs. SNR, and for different intrinsic size

if (0):
    dmag0, dmagStd0, Cpsf0, CpsfErr0, SNRpsf0, Cmod0, CmodErr0, SNRmod0, Cbest0, CbestStd0, sigBest0, sigBestStd0 = analyzeSNRbehavior(0.0)
    dmag1, dmagStd1, Cpsf1, CpsfErr1, SNRpsf1, Cmod1, CmodErr1, SNRmod1, Cbest1, CbestStd1, sigBest1, sigBestStd1 = analyzeSNRbehavior(0.375)
    dmag2, dmagStd2, Cpsf2, CpsfErr2, SNRpsf2, Cmod2, CmodErr2, SNRmod2, Cbest2, CbestStd2, sigBest2, sigBestStd2 = analyzeSNRbehavior(0.75)


   # ***** run above for 30 steps in SNR, 25 iterations ****
   # plot dmag plots, is Cbest/CbestStd similar to SNR?  what about SNR for sigmaBest vs. dmag? 
   # 

### plot vs. SNR 
# dmagPlot(SNRmod0, dmag0, dmagStd0, SNRmod1, dmag1, dmagStd1)

# dmagPlot(SNRmod1, dmag1, dmagStd1, SNRmod2, dmag2, dmagStd2)

# dmagPlot(SNRmod1, Cbest1, CbestStd1, SNRmod2, Cbest2, CbestStd2)
 
# dmagPlot(SNRmod0, Cbest0, CbestStd0, SNRmod1, Cbest1, CbestStd1)

# dmagPlot(SNRmod0, sigBest0, sigBestStd0, SNRmod1, sigBest1, sigBestStd1)

# dmagPlot(SNRmod1, sigBest1, sigBestStd1, SNRmod2, sigBest2, sigBestStd2)


## constant SNR 
#sigma_m = 15.0
#sigtrue = 0.75
#dmag1, Cpsf1, CpsfErr1, SNRpsf1, Cmod1, CmodErr1, SNRmod1, Cbest1, CbestRMS1, sigBest1, sigBestRMS1 = doManyImages(50,sigtrue,sigma_m)
#sigtrue = 0.50
#dmag2, Cpsf2, CpsfErr2, SNRpsf2, Cmod2, CmodErr2, SNRmod2, Cbest2, CbestRMS2, sigBest2, sigBestRMS2 = doManyImages(50,sigtrue,sigma_m)
#sigtrue = 0.25
#dmag3, Cpsf3, CpsfErr3, SNRpsf3, Cmod3, CmodErr3, SNRmod3, Cbest3, CbestRMS3, sigBest3, sigBestRMS3 = doManyImages(50,sigtrue,sigma_m)
#sigtrue = 0.0
#dmag4, Cpsf4, CpsfErr4, SNRpsf4, Cmod4, CmodErr4, SNRmod4, Cbest4, CbestRMS4, sigBest4, sigBestRMS4 = doManyImages(50,sigtrue,sigma_m)



### SCALING dmag vs. sigTrue for sigma_m=2.0, 10, 15  analysis below   ****
if (0):
    Nsteps = 8
    sigtrueVec = np.linspace(0, 1.5, Nsteps)
    dmag = 0*sigtrueVec
    dmagRMS = 0*sigtrueVec
    psfSNR = 0*sigtrueVec
    modSNR = 0*sigtrueVec
    Cmod = 0*sigtrueVec
    Cbest= 0*sigtrueVec
    NeffPsf = 0*sigtrueVec
    NeffMod = 0*sigtrueVec

    sigma_m = 2.0
    for sigiter in range(0,Nsteps):
        sigtrue = sigtrueVec[sigiter]
        print '|||||||| working on sigtrue =', sigtrue, '(sigma_m=10)'
        dmag1, Cpsf1, CpsfErr1, SNRpsf1, Cmod1, CmodErr1, SNRmod1, Cbest1, CbestRMS1, sigBest1, sigBestRMS1 = doManyImages(5,sigtrue,sigma_m)
        print 'returned from doManyImages'
        dmag[sigiter] = np.median(dmag1)
        dmagRMS[sigiter] = np.std(dmag1)
        psfSNR[sigiter] = np.median(SNRpsf1)
        modSNR[sigiter] = np.median(SNRmod1)
        NeffPsf[sigiter] = np.median((CpsfErr1/sigma_m)**2)
        NeffMod[sigiter] = np.median((CmodErr1/sigma_m)**2)
        Cmod[sigiter] = np.median(Cmod1)
        Cbest[sigiter] = np.median(Cbest1)



# %run '/Users/ivezic/.ipython/profile_zi/startup/zi.ipy'
# plotXYv(sigtrueVec, dmag, xMin=0, xMax=3.0, yMin = -0.2, yMax= 1.3, xLabel='sigtrue', yLabel='dmag')

# sConv = np.sqrt(1.5**2 +  sigtrueVec**2)
# sRatio = sConv/1.5
# fluxRat = 10**(0.4*dmag)
# dmagModelOld=1.086*(sRatio-1)
# dmagModel = 2.5*np.log10((1+sRatio**2)/2)
# dmagResid = dmag - dmagModel
# dmagdmagRMS = dmag/dmagRMS
# plotXYv(sRatio, dmag, xMin=1, xMax=2.3, yMin = 0.0, yMax= 1.3, xLabel='sigRatio', yLabel='dmag')
# plotXYv(sRatio, dmagResid, xMin=1, xMax=2.3, yMin = -0.07, yMax= 0.07, xLabel='sigRatio', yLabel='dmagResid')
# plotXYv(sRatio, fluxRat, xMin=1, xMax=2.3, yMin = 1.0, yMax= 3.0, xLabel='sigmaRat', yLabel='fluxRat')
# fluxRatModel = 0.5*(1+sRatio**2)
# fluxResid = fluxRat / fluxRatModel
# plotXYv(sRatio, fluxResid, xMin=1, xMax=2.3, yMin = 0.9, yMax= 1.1, xLabel='sigmaRat', yLabel='fluxRat')
# plotXYv(sRatio, dmagRMS, xMin=1, xMax=2.3, yMin = 0.0, yMax= 0.1, xLabel='sigmaRat', yLabel='dmagRMS')
# plotXYv(Cbest, Cmod, xMin=900, xMax=1100, yMin = 900, yMax= 1100, xLabel='Cbest', yLabel='Cmod')
# plotXYv(dmagdmagRMS, sRatio, xMin=0.0, xMax=5.0, yMin = 1.0, yMax= 2.3, xLabel='dmag/dmagRMS', yLabel='sigmaRat')


## flux ratio goes as 0.5*(1+sRatio**2)!!!    **** for sigRatio=2.0, dmag = 1.0 mag ****
## and dmag as dmag = 2.5*np.log10((1+sRatio**2)/2)
## with dmagRMS = 0.01 + 0.013 * sRatio    <== this is for sigma_m = 2.0  ** high SNR: <psf>=31, <mod>=42** 
#
##   for sigma_m = 10, interesting behavior of dmagRMS:     SNR <psf>= 12.6, <mod>=13.5 (+-3.4) 
##      sigRat < 1.1:  dmagRMS = sigRat-1.0  (for sigRat=1.1, dmagRMS=0.1)
##      sigRat > 1.1:  dmagRMS = 0.09 + 0.08*(sigRat-1.0)   
#    
##   for sigma_m = 15, interesting behavior of dmagRMS:     SNR <psf>= 8.2, <mod>=9.1 (+-2.2) 
##      sigRat < 1.3:  dmagRMS = 0.7*(sigRat-1.0)  (for sigRat=1.3, dmagRMS=0.2)
##      sigRat > 1.3:  dmagRMS = 0.18 + 0.14*(sigRat-1.0)   
##     for lowSNR, dmag is "stuck to 0", here, until sigRat = 1.2, then dmag = (sigRat-1) until sigRat~1.2
##         apart from being stuck around sigRat~1.2, model dmagModel = 2.5*np.log10((1+sRatio**2)/2) is OK at ~0.05 mag level
## 

###########  DES idea: convolved with a known profile???? vs. SNR  and for a range of sigRatio
## 5 methods: dmag with best fit, spread_model, Sebok ML solution, full Bayesian solution, fitting sigma
#########

##  sigRatio = 1.0
##    sigma_m     SNRpsf    dmag   dmagRMS   <sigBest>  sigStd  <sigRMS> 
##      2.0       94.3      0.000   0.004      0.03      0.05     0.06
##     10.0       18.8      0.002   0.015      0.07      0.11     0.13
##     15.0       12.3      0.006   0.011      0.09      0.09     0.18
##     20.0        9.4      0.012   0.073      0.02      0.35     0.04


##  sigRatio = 1.4
##    sigma_m     SNRpsf  SNRmod  dmag   dmagRMS 
##      2.0       31.0     42.0   0.44    0.03    
##     10.0       12.6     13.5   0.45    0.12      # 1 mag above 5sigma limit, still OK 
##     15.0        8.2      9.1   0.30    0.22 

##   for stars, dmagRMS smaller than psf mag errors from SNR: for high SNR, twice as small, for low SNR becoming equal 
##    for sigRatio=1.4, dmagRMS is about equal at high SNR, and becoming larger by a factor of low SNR



#  <dmag>    dmag_rms   <SNRpsf>   <SNRmod>  <Cmod>  CmodStd <CmodErr>  |||  <Cbest>   CbestStd   <CbestRMS> CbestRMSStd  <sigBest>  sigStd  <sigRMS> 
# sigtrue = 0.0, sigma_m=20.0, 500 iterations 
#   0.012     0.073       9.37      9.44     1033.7    124.9     106.4        1033.9      124.6       10.9        3.2        0.02     0.35     0.04
#
# sigtrue = 0.0, sigma_m=35.0,  50 iterations 
#   0.019     0.184       5.52      5.63     1083.9    179.4     186.9        1055.8      150.0      183.5       22.7        0.14     8.45     0.31
# sigtrue = 0.0, sigma_m=35.0,  ** 2000 ** iterations 
#   0.021     0.174       5.60      5.71     1098.8    176.8     187.0        1065.2      138.1      182.9       23.9        0.15    51.56     0.30
#  -> dmag is biased high because sigBest is biased high; does sigma=0.15 correspond to dmag=0.02 mag?   SCALING dmag vs. sigTrue for sigma_m=2.0 


### if we gave correct model, what would be dmag_rms?  Repeat the above with true model
##  running at the market: changed fitModel=0 in doImageStats and resubmitted above (top right window, left is for practicing plotting) 
#  <dmag>    dmag_rms   <SNRpsf>   <SNRmod>  <Cmod>  CmodStd <CmodErr>  |||  <Cbest>   CbestStd   <CbestRMS> CbestRMSStd  <sigBest>  sigStd  <sigRMS> 
#  0.000      0.000      5.35        5.35     995.9   185.5    186.1           -1.0      0.0       -1.0        0.0        -1.00     0.00     -1.00
##   exactly the same psf profile and model profile: of course dmag is always 0 ... 




##  #  MARKET  while at the market: TOP RIGHT WINDOW also repeat for answering 
# 1) why is Cbest biased for large sigtrue? 
# 
# sigma_m = 10.0
# sigtrue = 1.0
# dmag1, Cpsf1, CpsfErr1, SNRpsf1, Cmod1, CmodErr1, SNRmod1, Cbest1, CbestRMS1, sigBest1, sigBestRMS1 = doManyImages(2000,sigtrue,sigma_m)

#  <dmag>    dmag_rms   <SNRpsf>   <SNRmod>  <Cmod>  CmodStd <CmodErr>  |||  <Cbest>   CbestStd   <CbestRMS> CbestRMSStd  <sigBest>  sigStd  <sigRMS> 
# sigtrue = 1.0, sigma_m=10.0  *** 50 iterations ***
#   0.043     0.063     15.33      15.53     856.1     73.7     54.9           884.2       75.5      126.6        25.0        0.38     0.25     0.46
# sigtrue = 1.0, sigma_m=10.0  *** 2000 iterations ***
#   0.162     0.114     15.39      15.67     947.2    107.9     60.6           959.7      101.5       99.3        13.4        0.82     0.37     0.33
#    once more with 2000 iterations
#   0.156     0.112     15.39      15.67     942.2    110.0     60.3           960.2      103.6       99.5        13.4        0.80     0.36     0.34



# limits = [(500, 1500), (0,2)]
# plot2chains(Cbest1, sigBest1, 1000, 0.75, Xlabel='C', Ylabel='sigma', limits=limits, bins=25)


# individual case 
# sigtrue = 1.0 
# sigma_m = 10.0 
# oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigma_m)
# basic stats 
# neff, CountPSF, CountModel, neffMod, Cbest, sigmaBest, ABestRMS, sigmaBestRMS, traces, fit_vals  = doImageStats(oneDpixels, sigmaPSF, sigma_m, psf, nonoise, image, 1)
# traces2, fit_vals2 = plotSingleCase(Btrue, Atrue, muXtrue, sigtrue, sigma_m, sigmaPSF, traces, fit_vals)

# 1) why is Cbest biased for large sigtrue? 
#   sigma from best-fit model is biased low, and thus more similar to psf and underestimates counts...
# 




#######  50 iterations, 35000/5000 for MCMC, without taking source noise into account, and sigmaPSF = 1.5  
###  at i~25, the mode for galaxy distribution is sigtrue = 0.5 * sigmaPSF -> 0.75 
##                 sigtrue = 1.0  => sigconvolved = 1.2 *sigmaPSF 
# 
#  <dmag>    dmag_rms   <SNRpsf>   <SNRmod>  <Cmod>  CmodStd <CmodErr>  |||  <Cbest>   CbestStd   <CbestRMS> CbestRMSStd  <sigBest>  sigStd  <sigRMS> 
# sigtrue = 1.0, sigma_m=2.0
#   0.219     0.015     76.83      78.10     1000.0    17.4     12.8           1000.4      17.4       17.9        0.4        1.00     0.04     0.04
# sigtrue = 0.75, sigma_m=2.0
#   0.128     0.012     83.66      84.17     1002.2    15.6     11.9           1002.5      15.6       16.7        0.4        0.75     0.04     0.04
# sigtrue = 0.50, sigma_m=2.0
#   0.056     0.015     89.20      89.32      999.9    19.5     11.2           1000.6      18.9       16.1        0.5        0.49     0.08     0.06
# sigtrue = 0.25, sigma_m=2.0
#   0.001     0.008     92.97      92.99      991.4    12.1     10.6            995.0      12.8       12.3        2.6        0.07     0.09     0.10
# sigtrue = 0.00, sigma_m=2.0 
#   0.000     0.004     94.31      94.32     1004.0    11.1     10.6           1004.6      11.3       11.0        1.3        0.03     0.05     0.06
#
# =======
#  <dmag>    dmag_rms   <SNRpsf>   <SNRmod>  <Cmod>  CmodStd <CmodErr>  |||  <Cbest>   CbestStd   <CbestRMS> CbestRMSStd  <sigBest>  sigStd  <sigRMS> 
# sigtrue = 1.0, sigma_m=10.0
#   0.043     0.063     15.33      15.53     856.1     73.7     54.9           884.2      75.5      126.6        25.0        0.38     0.25     0.46
# sigtrue = 0.75, sigma_m=10.0
#   0.035     0.070     16.51      16.71     927.5     85.8     54.6           946.6      83.4       86.1        15.9        0.35     0.28     0.33
# sigtrue = 0.50, sigma_m=10.0
#   0.005     0.028     18.00      18.02     963.0     57.4     53.3           968.7      58.5       59.0        10.9        0.10     0.14     0.19
# sigtrue = 0.25, sigma_m=10.0
#   0.003     0.008     18.23      18.27     976.5     52.4     53.2           978.0      53.4       56.6         7.8        0.07     0.08     0.14
# sigtrue = 0.00, sigma_m=10.0
#   0.002     0.015     18.76      18.78    1001.1     50.5     53.2          1002.5      51.8       55.5        10.9        0.07     0.11     0.13
#
# =======
#  <dmag>    dmag_rms   <SNRpsf>   <SNRmod>  <Cmod>  CmodStd <CmodErr>  |||  <Cbest>   CbestStd   <CbestRMS> CbestRMSStd  <sigBest>  sigStd  <sigRMS> 
# sigtrue = 1.0, sigma_m=15.0
#  0.056     0.142     10.23      10.52     900.8   157.9     82.7            917.2      152.3       125.5        30.4        0.41     0.45     0.39
# sigtrue = 0.75, sigma_m=15.0
#  0.026     0.086     11.30      11.48     928.8    98.5     80.9            961.7      100.4       108.5        29.6        0.25     0.32     0.36
# sigtrue = 0.50, sigma_m=15.0 
#  0.007     0.040     11.82      11.92     967.6    96.1     79.9            968.3       98.6        85.4        20.0        0.10     0.18     0.20
# sigtrue = 0.25, sigma_m=15.0
#  0.007     0.014     12.49      12.61    1013.4    76.1     79.9           1015.2       78.4        83.2        13.4        0.08     0.09     0.17
# sigtrue = 0.00, sigma_m=15.0 
#  0.006     0.011     12.34      12.36     999.1    81.4     79.9           1006.1       81.8        83.6        13.0        0.09     0.09     0.18


# Questions:
# 1) why is Cbest biased for large sigtrue? 
# 2)  --> why does dmag_rms increase with sigtrue at constant SNR? <-- 
# 3) does dmag_rms increase with sigma_m at constant sigtrue? 


#s1 = np.median(dmag1)
#s2 = np.std(dmag1)
#s3 = np.median(SNRpsf1)
#s4 = np.median(SNRmod1)
#s5 = np.median(Cmod1)
#s6 = np.std(Cmod1)
#s7 = np.median(CmodErr1)
#c1 = np.median(Cbest1)
#c2 = np.std(Cbest1)
#c3 = np.median(CbestRMS1)
#c4 = np.std(CbestRMS1)
#c5 = np.median(sigBest1)
#c6 = np.std(sigBest1)
#c7 = np.median(sigBestRMS1)

#print '    %.3f     %.3f     %.2f      %.2f     %.1f    %.1f     %.1f           %.1f      %.1f       %.1f        %.1f        %.2f     %.2f     %.2f' % (s1, s2, s3, s4, s5, s6, s7, c1, c2, c3, c4, c5, c6, c7)





### FINAL ###
###  at i~25, the mode for galaxy distribution is at sigtrue = 0.5 * sigmaPSF -> 0.75 pixels (mode: FWHM=0.35 arcsec) 
##    sigtrue = 0.75  => sigconvolved = 1.68 pixels  ===> sigRatio = 1.12 
##  the median is at 0.4 arcsec: this is 0.57*sigmaPSF and  ===> sigRatio = 1.15 
##   at i~22, the median is 0.77 arcsec, roughly sigRatio = 1.41 


##    We want sigRatio = 1.0, 1.12, and 1.41  (sigtrue = 0.0, 0.75 and 1.5 pixel) 
##      SNR: sigma_m = 2.0, 10.0, 15.0  

###########  DES idea: convolved with a known profile???? vs. SNR  and for a range of sigRatio
## 5 methods: dmag with best fit, spread_model, Sebok ML solution, full Bayesian solution, fitting sigma
#########

##  sigRatio = 1.0    (sigtrue = 0.00) 
##    sigma_m     SNRpsf    dmag   dmagRMS   <sigBest>  sigStd  <sigRMS> 
##      2.0       94.3      0.000   0.004      0.03      0.05     0.06      
##     10.0       18.8      0.002   0.015      0.07      0.11     0.13
##     15.0       12.3      0.006   0.011      0.09      0.09     0.18

##  sigRatio = 1.12   (sigtrue = 0.75)   
##    sigma_m     SNRpsf  SNRmod   dmag    dmagRMS   <sigBest>  sigStd  <sigRMS> 
##      2.0        83.7    84.2   0.128     0.012      0.75      0.04     0.04
##     10.0        16.5    16.7   0.035     0.070      0.35      0.28     0.33
##     15.0        11.3    11.5   0.026     0.086      0.25      0.32     0.36

##  sigRatio = 1.4   (sigtrue = 1.5) 
##    sigma_m     SNRpsf  SNRmod  dmag   dmagRMS   <sigBest>  sigStd  <sigRMS> 
##      2.0        31.0    42.0   0.44    0.03    
##     10.0        12.6    13.5   0.45    0.12     
##     15.0         8.2     9.1   0.30    0.22 



# Given sigRatio, at what SNR is dmag = 2*dmagRMS? 
# Or, given SNR, at what sigRatio is dmag = 2*dmagRMS? 
#
# Generally, given a classifier, as a function of SNR, at what thetaObs/thetaPSF is 
#            the classifier different from its PSF value at k*sigma level?   


# e.g. to be resolved with k=2 sigma significance (using dmag or sigBest), sigmaRat must be at least
#                 dmag   sigBest
# sigma_m =  2    1.05    1.03                sigBest biased low for sigTrue < 0.4  (sigmaRatio=1.035)
# sigma_m = 10    1.25    1.20                sigBest biased low for sigTrue < 1.0  (sigmaRatio=1.202)
# sigma_m = 15    1.45    1.30                sigBest biased low for sigTrue < 1.5  (sigmaRatio=1.41)   <dmag>=0.5, dmagRMS=0.25 

## vs. stars: if sigRatio = 1.0 (sigtrue=0), what is the scatter in dmag and sigBest for stars? 
##    sigma_m     SNRpsf    dmag   dmagRMS   <sigBest>  sigStd  <sigRMS> 
##      2.0       94.3      0.000   0.004      0.03      0.05     0.06      
##     10.0       18.8      0.002   0.015      0.07      0.11     0.13
##     15.0       12.3      0.006   0.011      0.09      0.09     0.18

### it looks like we need 3 cases of star vs. resolved source comparison:
##  sigma_m = 2,  with sigtrue=0.4
##  sigma_m = 10, with sigtrue=1.0
##  sigma_m = 15, with sigtrue=1.5 



if (0):
    sRatio = np.sqrt(1.5**2 +  sigtrueVec**2)/1.5
    dmagdmagRMS = dmag/dmagRMS
    sigBsigBRMS = sigBest/sigBestRMS 
    sigBchi = sigBestStd/sigBestRMS    # is this ok? 

    # %run '/Users/ivezic/.ipython/profile_zi/startup/zi.ipy'
    plotXYv(dmagdmagRMS, sRatio, xMin=0.0, xMax=5.0, yMin = 1.0, yMax= 2.3, xLabel='dmag/dmagRMS', yLabel='sigmaRat')
    plotXYv(sigBsigBRMS, sRatio, xMin=0.0, xMax=5.0, yMin = 1.0, yMax= 2.3, xLabel='sigBest/sigBestRMS', yLabel='sigmaRat')

    plotXYv(sigtrueVec, sigBest, xMin=0.0, xMax=3.1, yMin = 0.0, yMax= 3.1, xLabel='sigTrue', yLabel='sigmaBest')

    plotXYv(dmagdmagRMS, dmag, xMin=0.0, xMax=5.0, yMin = 0.0, yMax= 1.2, xLabel='dmag/dmagRMS', yLabel='dmag')
    plotXYv(dmagdmagRMS, dmagRMS, xMin=0.0, xMax=5.0, yMin = 0.0, yMax= 0.3, xLabel='dmag/dmagRMS', yLabel='dmagRMS')

### compute for sigma_m= 2.0, 10, 15 
Ntrials = 100
if (0):
    sigma_m = 15.0
    #Nsteps = 31
    #sigtrueVec = np.linspace(0, 3.0, Nsteps)
    Nsteps = 8
    sigtrueVec = np.linspace(1, 2.6, Nsteps)
    dmag = 0*sigtrueVec
    dmagRMS = 0*sigtrueVec
    psfSNR = 0*sigtrueVec
    modSNR = 0*sigtrueVec
    Cmod = 0*sigtrueVec
    Cbest= 0*sigtrueVec
    sigBest = 0*sigtrueVec
    sigBestStd = 0*sigtrueVec
    sigBestRMS = 0*sigtrueVec
    sigBestRMSstd = 0*sigtrueVec
    for sigiter in range(0,Nsteps):
        sigtrue = sigtrueVec[sigiter]
        print '|||||||| working on sigtrue =', sigtrue, '(sigma_m=',sigma_m,')'
        dmag1, Cpsf1, CpsfErr1, SNRpsf1, Cmod1, CmodErr1, SNRmod1, Cbest1, CbestRMS1, sigBest1, sigBestRMS1 = doManyImages(Ntrials,sigtrue,sigma_m)
        print 'returned from doManyImages'
        dmag[sigiter] = np.median(dmag1)
        dmagRMS[sigiter] = np.std(dmag1)
        sigBest[sigiter] = np.median(sigBest1)
        sigBestStd[sigiter] = np.std(sigBest1)
        sigBestRMS[sigiter] = np.median(sigBestRMS1)
        sigBestRMSstd[sigiter] = np.std(sigBestRMS1)
        psfSNR[sigiter] = np.median(SNRpsf1)
        modSNR[sigiter] = np.median(SNRmod1)
        Cmod[sigiter] = np.median(Cmod1)
        Cbest[sigiter] = np.median(Cbest1)



def ThreeClassifiersDistributions(Ntrial, sigG, sigmaNoise, fitMCMC=0, p5=0):

        if (0): 
            ## CODE CHANGE FOR TESTING C_SPREAD
            if (sigG < 0):
                sigGin = -1.0*sigG
                sigG = 0.0
            else: 
                sigGin = sigG  # these values are for evaluating C_spread
                sigG = 1.0   # always with this width
                
            print 'computing ansatz with:', sigGin, sigG, sigmaNoise

        ## defaults everywhere
	muTx = 0
        ## for 4 param testing
        # muTx = 0.4999
        muTy = 0
        Btrue = 0.0
	Atrue = 1000.0 
	sigmaPSF = 1.5

        C1 = np.linspace(0, 1, Ntrial) 
        C2 = np.linspace(0, 1, Ntrial) 
        C3 = np.linspace(0, 1, Ntrial) 
        neffPSF = np.linspace(0, 1, Ntrial) 
        neffModel = np.linspace(0, 1, Ntrial) 
        CountModel = np.linspace(0, 1, Ntrial) 
        chi2PSF = np.linspace(0, 1, Ntrial) 
        chi2Model = np.linspace(0, 1, Ntrial) 
        bestAVec = np.linspace(0, 1, Ntrial) 
        bestARmsVec = np.linspace(0, 1, Ntrial) 
        bestSigmaVec = np.linspace(0, 1, Ntrial) 
        bestSigmaRmsVec = np.linspace(0, 1, Ntrial) 
        spreadModel = np.linspace(0, 1, Ntrial) 
        eta = np.linspace(0, 1, Ntrial) 


        for i in range(0,Ntrial):
            print 'i, Ntrial=', i, Ntrial
            ## generate new images
            oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muTx, muTy, sigG, Atrue, Btrue, sigmaPSF, sigmaNoise, makeplot=0)
                
            ## psf profile is known:
            psfModelNorm = psf / np.sum(psf)

            ## model profile: two options
            if (0):
                # FAST: for model we take the true profile; in reality sigma would be a bit different 
                modelNorm = nonoise / np.sum(nonoise)
                bestSigmaVec[i] = -1.0
                bestSigmaRmsVec[i] = -1.0 
            else: 
                ## SLOW: but correct(er)...
                if (fitMCMC):
                    ## full MCMC fit, which varies other parameters too 
                    modelNorm, bestA, bestARMS, bestSigma, bestSigmaRMS, BF = getBestModelFitMCMC(oneDpixels, image, sigmaPSF, sigmaNoise, p5)
                    print 'i=', i, '  best A+-rms: ', bestA, bestARMS, '  best sigma+-rms: ', bestSigma, bestSigmaRMS
                    bestAVec[i] = bestA
                    bestARmsVec[i] = bestARMS
                    bestSigmaVec[i] = bestSigma
                    bestSigmaRmsVec[i] = bestSigmaRMS
                else:
                    ## chi2 minimization to get best sigtrue value (faster than MCMC but all other params fixed)
                    modelNorm, bestSigma = getBestModelFit(oneDpixels, image, sigmaPSF, 0.0, 2.5, 1001)
                    ## THIS IS FOR C_SPREAD TESTING
                    ## INSTEAD OF SEARCHING FOR BEST SIGMA, FORCE sigGin
                    ## modelNorm, bestSigma = getBestModelFit(oneDpixels, image, sigmaPSF, sigGin, sigGin, 1)

                    # ----------------
                    bestAVec[i] = -1.0 
                    bestARmsVec[i] = -1.0 
                    bestSigmaVec[i] = bestSigma
                    bestSigmaRmsVec[i] = -1.0 
 
            ## compute some return quantities 
            # psf 
            neffPSF[i] = 1.0 / np.sum(psfModelNorm**2)
            CountPSF = neffPSF[i] * np.sum(image*psfModelNorm)
            chi2PSF[i] = np.sum(((image-CountPSF*psfModelNorm)/sigmaNoise)**2)
   
            # model
            neffModel[i]  = 1.0 / np.sum(modelNorm**2)
            CountModel[i] = neffModel[i] * np.sum(image*modelNorm)
            chi2Model[i] = np.sum(((image-CountModel[i]*modelNorm)/sigmaNoise)**2)
          
	    # classifiers 
            if (0):
                ## THIS IS SEBOK'S CLASSIFIER
                C1[i] = (CountModel[i] / CountPSF) * np.sqrt(neffPSF[i]/neffModel[i]) 
            else:
                ## BAYES FACTOR 
                if (fitMCMC):
                    a1, a2, a3, a4, a5, BFpsf = getBestModelFitMCMC(oneDpixels, image, sigmaPSF, sigmaNoise, p5, PSF=1)
                    # for BF see call to getBestModelFitMCMC above
                    print 'BF=', BF, 'BFpsf=', BFpsf
                    C1[i] = 2*(BFpsf-BF)            
                else:
                    # compute chi2 for 2D array of sigma and Cmod for image
                    chiPixSig, chiPixCmod, chi2im, bestModel, sigmaML, CmodML, chi2min = \
                        getChi2image(oneDpixels, image, muTx, muTy, Btrue, sigmaPSF, sigmaNoise)
                    # and now get Bayes factor for this chi2 image
                    maxLikeRat, BayesFactor = chi2image2BF(chiPixSig, chi2im) 
                    ## take ln and multiply by 2 to go to BIC 
                    C1[i] = 2*np.log(BayesFactor)

            C2[i] = CountModel[i] / CountPSF
            C3[i] = chi2PSF[i] - chi2Model[i]
            # eta is needed for spread_model from SExtractor
            eta[i] = np.sqrt(np.sum(psfModelNorm**2)*np.sum(modelNorm**2))/np.sum(psfModelNorm*modelNorm)
            # N.B. spreadModel[i] = eta*Csebok - 1

        ### return also two bestSigma values 
	return eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, CountModel, bestAVec, bestARmsVec, bestSigmaVec, bestSigmaRmsVec



# compute maxL ratio and Bayes Factor 
def chi2image2BF(chiPixSig, chi2im): 

    # get likelihood from chi2
    L = np.exp(-chi2im/2)
    # the sigma=0 likelihood column
    L0 = L[:,0]
    ## first maximum likelihood ratio (wrt sigma=0)
    maxLikeRat = np.max(chi2im)/np.max(L0)

    ## and now Bayes factor (assuming uniform priors)
    # step and range in sigma
    stepSig = chiPixSig[1]-chiPixSig[0]
    rangeSig = chiPixSig[-1]-chiPixSig[0]
    # and now get Bayes factor for this chi2 image
    BayesFactor = (np.sum(L) * stepSig / rangeSig) / np.sum(L0) 
                   
    return maxLikeRat, BayesFactor


def getMLparametersMCMC(sigmaPSF, sigmaNoise, oneDpixels, image, traces, params5 = 0, returnChi2array = 0):

    chi2min = np.inf
    if (returnChi2array):
        Chi2array = 0 * traces[0]
    for i in range(0,traces[0].size):
        # any number of model parameters
        B = traces[0][i]
        A = traces[1][i]
        muGx = traces[2][i]
        if (params5):
            # five model parameters
            muGy = traces[3][i]
            sigma = traces[4][i]
        else:
            # two model parameters
            muGy = muGx
            sigma = traces[3][i]

        model = gauss2D(muGx, muGy, sigma, A, B, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        chi2 = np.sum((image-model)**2/sigmaNoise**2)
        if (returnChi2array):
            Chi2array[i] = chi2
        if ((i == 0) or (chi2 < chi2min)): 
            chi2min = chi2
            iBest = i 
 
    if (returnChi2array):
        return chi2min, iBest, Chi2array
    else:
        return chi2min, iBest



def getBestModelFitMCMCBF(oneDpixels, image, sigmaPSF, sigmaNoise, params5, PSF=0): 

        if (params5):
            os.remove('2DgaussFit5.pkl')
            # 5 free parameters:
            traces, fit_vals, BF, dBF = DoGaussFit5(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigmaNoise, image, PSF)
        else:
            os.remove('2DgaussFit.pkl')        
            # 2 free parameters:
            traces, fit_vals, BF, dBF = DoGaussFit(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigmaNoise, image, PSF)
        
        print 'BF, dBF=', BF, dBF

        if (1):
            # find true maximum likelihood point
            chi2min, iBest = getMLparametersMCMC(sigmaPSF, sigmaNoise, oneDpixels, image, traces, params5)
            # print 'ML point at i=', iBest, ' with chi2=', chi2min
            b0Best = traces[0][iBest]
            ABest = traces[1][iBest]
            muGxBest = traces[2][iBest]
            if (params5):
                # five model parameters
                muGyBest = traces[3][iBest]
                sigmaBest = traces[4][iBest]
            else:
                # two model parameters
                muGyBest = muGxBest
                sigmaBest = traces[3][iBest]

            ABestRMS = np.std(traces[1])
            sigmaBestRMS = np.std(traces[3])
        else:
            print 'TESTING BF (in getBestModelFitMCMC) !!!'

        BestModel = gauss2D(muGxBest, muGyBest, sigmaBest, ABest, b0Best, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        modelNormMCMC = BestModel / np.sum(BestModel)

        return modelNormMCMC, ABest, ABestRMS, sigmaBest, sigmaBestRMS, chi2min, iBest, BF, dBF, traces, fit_vals 



def getBestModelFitMCMC(oneDpixels, image, sigmaPSF, sigmaNoise, params5, PSF=0): 

        if (params5):
            os.remove('2DgaussFit5.pkl')
            # 5 free parameters:
            traces, fit_vals, BF, dBF = DoGaussFit5(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigmaNoise, image, PSF)
        else:
            os.remove('2DgaussFit.pkl')        
            # 2 free parameters:
            traces, fit_vals, BF, dBF = DoGaussFit(oneDpixels[:, np.newaxis], oneDpixels, sigmaPSF, sigmaNoise, image, PSF)
        
        print 'BF, dBF=', BF, dBF

        if (1):
            # find true maximum likelihood point
            chi2min, iBest = getMLparametersMCMC(sigmaPSF, sigmaNoise, oneDpixels, image, traces, params5)
            # print 'ML point at i=', iBest, ' with chi2=', chi2min
            b0Best = traces[0][iBest]
            ABest = traces[1][iBest]
            muGxBest = traces[2][iBest]
            if (params5):
                # five model parameters
                muGyBest = traces[3][iBest]
                sigmaBest = traces[4][iBest]
            else:
                # two model parameters
                muGyBest = muGxBest
                sigmaBest = traces[3][iBest]

            ABestRMS = np.std(traces[1])
            sigmaBestRMS = np.std(traces[3])
        else:
            print 'YOU SHOULD NOT BE HERE (in getBestModelFitMCMC) !!!'
            b0Best = np.mean(traces[0])
            ABest = np.mean(traces[1])
            muGxBest = np.mean(traces[2])
            muGyBest = np.mean(traces[2])
            sigmaBest = np.mean(traces[3])
            ABestRMS = np.std(traces[1])
            sigmaBestRMS = np.std(traces[3])

        BestModel = gauss2D(muGxBest, muGyBest, sigmaBest, ABest, b0Best, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        modelNormMCMC = BestModel / np.sum(BestModel)

        return  modelNormMCMC, ABest, ABestRMS, sigmaBest, sigmaBestRMS, BF


def getBestModelFit(oneDpixels, image, sigmaPSF, sigMin, sigMax, Nstep, muT=0, Btrue=0):

    # grid of sigma true 
    sigTrueGrid = np.linspace(sigMin, sigMax, Nstep) 

    for i in range(0,sigTrueGrid.size):
        sTrue = sigTrueGrid[i]
        BestModel = gauss2D(muT, muT, sTrue, 1.0, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        modelNorm = BestModel / np.sum(BestModel)
        wModel = modelNorm / np.sum(modelNorm**2)
        CountsModel = np.sum(image*wModel)
        chi2 = np.sum((image-CountsModel*modelNorm)**2)
        if ((i == 0) or (chi2 < chi2Min)): 
            chi2Min = chi2
            bestSigma = sTrue
            bestModelNorm = modelNorm    

    return bestModelNorm, bestSigma




def ThreeClassifiers(sigG, sigma_m):

	muT = muT = Btrue = 0.0
	Atrue = 1000.0 
	sigmaPSF = 1.5

	oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muT, muT, sigG, Atrue, Btrue, sigmaPSF, sigma_m)

	# psf 
	psfModel = gauss2D(muT, muT, 0.0, 100000.0, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
	modelNorm = psfModel / np.sum(psfModel)
	wModel = modelNorm / np.sum(modelNorm**2)
        CountPSF = np.sum(image*wModel)
	neffPSF = 1/np.sum(wModel**2)
	chi2PSF = np.sum(((image-CountPSF*modelNorm)/sigma_m)**2)
        print 'noisey profile with model psf: CountPSF=',CountPSF
        print 'noisey profile with model psf: chi2PSF=',chi2PSF
        print 'sum(chi):', np.sum(((image-CountPSF*modelNorm)/sigma_m))

        CountPSF0 = np.sum(nonoise*wModel)
        print 'improved CountPSF using noiseless profile and model psf: CountPSF0=',CountPSF0
	chi2PSF0 = np.sum(((image-CountPSF0*modelNorm)/sigma_m)**2)
        print 'improved CountPSF with model psf: chi2PSF0=',chi2PSF0
        print 'sum(chi):', np.sum(((image-CountPSF0*modelNorm)/sigma_m))

        CountPSFtest = np.sum(psf*wModel)
        chi2TestPSF = np.sum(((psf-CountPSFtest*modelNorm)/sigma_m)**2)
        print ' true psf with model psf, chi2TestPSF=',chi2TestPSF  
        print 'sum(chi):', np.sum(((psf-CountPSFtest*modelNorm)/sigma_m))

        # test 
        chi2Test = np.sum(((image-nonoise)/sigma_m)**2)
        print 'noisey-noiseless prof: chi2Test=',chi2Test  
 

        # grid of sigma true 
        sigTrueGrid = np.linspace(0, 2*sigG, 41) 
	CountsModel = 0 + 0*sigTrueGrid
	neffModel = 0 + 0*sigTrueGrid
	chi2Model = 0 + 0*sigTrueGrid

	for i in range(0,sigTrueGrid.size):
		sTrue = sigTrueGrid[i]
                BestModel = gauss2D(muT, muT, sTrue, 1.0, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
                modelNorm = BestModel / np.sum(BestModel)
                wModel = modelNorm / np.sum(modelNorm**2)
		neffModel[i] = 1/np.sum(wModel**2)
                CountsModel[i] = np.sum(image*wModel)
                chi2Model[i] = np.sum(((image-CountsModel[i]*modelNorm)/sigma_m)**2)

	# classifiers 
	C1 = (CountsModel / CountPSF) * np.sqrt(neffPSF/neffModel) 
	C2 = CountsModel / CountPSF
	C3 = chi2PSF - chi2Model

        print 'MIN(chi2Model):', np.min(chi2Model),'  MAX(chi2Model):', np.max(chi2Model)
	return oneDpixels, nonoise, psf, image, diffimage, sigTrueGrid, C1, C2, C3, chi2PSF, chi2Model 






#######################
### plotting tools 



def plot2classHistograms(Cs, Cg, Xmin=0.0, Xmax=1.0, Ymin=0.0, Ymax=1.0, Xlabel='X', Ylabel='Y', bins=10, Climit=80):

    #------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.15, right=0.90, hspace=0.26, wspace=0.10)
    ax = fig.add_subplot(2, 1, 1)

    limits = [(Xmin, Xmax, Ymin, Ymax)]
    labels = [Xlabel, Ylabel]

    # plot a histogram
    ax.hist(Cs, bins=bins, normed=True, histtype='stepfilled', alpha=0.2)
    ax.hist(Cg, bins=bins, normed=True, histtype='stepfilled', alpha=0.2)
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    ax.set_xlabel(Xlabel, fontsize=14)
    ax.set_ylabel(Ylabel, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=15)
    #ax.fill(x, p1, ec='k', fc='#AAAAAA', alpha=0.5)
    #ax.fill(x, p2, '-k', fc='#AAAAAA', alpha=0.5)
    #ax.plot([120, 120], [0.0, 0.04], '--k')
    #ax.text(100, 0.036, r'$h_B(x)$', ha='center', va='bottom')


    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.25, right=0.80, hspace=0.26, wspace=0.10)
    ax = fig.add_subplot(2, 1, 2)
    ax.set_xlim(Climit, 100)
    ax.set_ylim(Climit, 100)
    ax.set_xlabel('completeness (\%)', fontsize=14)
    ax.set_ylabel('purity (\%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=15)

    Ngrid = 1000
    Cgrid = np.linspace(Xmin, Xmax, Ngrid)
    ComplS = 0*Cgrid
    PurityS = 0*Cgrid + 100.0
    ComplG = 0*Cgrid
    PurityG = 0*Cgrid + 100.0
    for i in range(0,Ngrid): 
        C = Cgrid[i]
        CsOK = Cs[Cs < C]
        CgOK = Cg[Cg < C]
        if (CsOK.size>0):
            ComplS[i] = 100.0*CsOK.size / Cs.size
            PurityS[i] = 100.0*CsOK.size / (CsOK.size + CgOK.size)
        CsOK2 = Cs[Cs > C]
        CgOK2 = Cg[Cg > C]
        if (CgOK2.size>0):
            ComplG[i] = 100.0*CgOK2.size / Cg.size
            PurityG[i] = 100.0*CgOK2.size / (CsOK2.size + CgOK2.size)

    ax.plot(ComplS, PurityS, '-b', lw=1)  
    ax.plot(ComplG, PurityG, '-g', lw=1)  
    ax.plot([90, 90], [0.0, 100], '--k')
    ax.plot([95, 95], [0.0, 100], '--k')
    ax.plot([0, 100], [90, 90], '--k')
    ax.plot([0, 100], [95, 95], '--k')

    plt.show()
    return






def analyzeC3(foutname):

    vNames = ['C1', 'C2', 'C3', 'neffPSF', 'neffModel', 'chi2PSF', 'chi2Model', 'bestSig', 'bestSigErr']
    vNames = vNames + ['sC1', 'sC2', 'sC3', 'SneffPSF', 'SneffModel', 'Schi2PSF', 'Schi2Model', 'SbestSig', 'SbestSigErr']
    v = np.loadtxt(foutname)

    xN = 'sC1'
    yN = 'sC2'
    plotXYv(v[vNames.index(xN)], v[vNames.index(yN)], xMin=0.95, xMax=1.25, yMin = 0.9, yMax= 2.2, xLabel=xN, yLabel=yN)

    xN = 'C1'
    yN = 'C2'
    plotXYv(v[vNames.index(xN)], v[vNames.index(yN)], xMin=0.95, xMax=1.25, yMin = 0.9, yMax= 2.2, xLabel=xN, yLabel=yN)

    xN = 'sC1'
    yN = 'sC3'
    plotXYv(v[vNames.index(xN)], v[vNames.index(yN)], xMin=0.95, xMax=1.25, yMin = -2, yMax= 50, xLabel=xN, yLabel=yN)

    xN = 'C1'
    yN = 'C3'
    plotXYv(v[vNames.index(xN)], v[vNames.index(yN)], xMin=0.95, xMax=1.25, yMin = -2, yMax= 50, xLabel=xN, yLabel=yN)


    xN = 'sC2'
    yN = 'sC3'
    plotXYv(v[vNames.index(xN)], v[vNames.index(yN)], xMin = 0.95, xMax= 1.5,  yMin=-2, yMax=50, xLabel=xN, yLabel=yN)

    xN = 'C2'
    yN = 'C3'
    plotXYv(v[vNames.index(xN)], v[vNames.index(yN)], xMin = 0.95, xMax= 1.5,  yMin=-2, yMax=50, xLabel=xN, yLabel=yN)


    # is dchi2 correlated with bestSig? 
    xN = 'SbestSig'
    yN = 'sC3'
    bSig = v[vNames.index(xN)]
    SbSig =  v[vNames.index(yN)]
    plotXYv(bSig, SbSig, xMin=-0.05, xMax=1.55, yMin = -2, yMax= 50, xLabel=xN, yLabel=yN)

    xN = 'bestSig'
    yN = 'C3'
    bSig = v[vNames.index(xN)]
    SbSig =  v[vNames.index(yN)]
    plotXYv(bSig, SbSig, xMin=-0.05, xMax=1.55, yMin = -2, yMax= 50, xLabel=xN, yLabel=yN)

    # and now plot 2 CSeb histograms
    Xlabel = 'C1 == Csebok'
    Ylabel = '$n / (N\Delta_{bin})$'
    sC1 = v[vNames.index('sC1')]
    C1 = v[vNames.index('C1')]
    Xmin = np.min(sC1) - 0.02
    Xmax = np.max(C1) 
    plot2classHistograms(sC1, C1, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=200.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=20, Climit=50)

    # and now plot 2 dmag histograms
    Xlabel = 'C2 == $m_{psf}-m_{mod}$'
    Ylabel = '$n / (N\Delta_{bin})$'
    sC2 = v[vNames.index('sC2')]
    C2 = v[vNames.index('C2')]
    Xmin = np.min(sC2) - 0.05
    Xmax = np.max(C2) 
    plot2classHistograms(sC2, C2, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=20.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=20, Climit=50)

    # and now plot delta chi2 histograms
    Xlabel = 'C3 == $\chi^2_{psf}-\chi^2_{mod}$'
    Ylabel = '$n / (N\Delta_{bin})$'
    sC3 = v[vNames.index('sC3')]
    C3 = v[vNames.index('C3')]
    Xmin = np.min(sC3) - 1
    Xmax = np.max(C3) 
    plot2classHistograms(sC3, C3, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=0.5, Xlabel=Xlabel, Ylabel=Ylabel, bins=20, Climit=50)

    # and now plot bestSigma histograms
    Xlabel = 'C4 == best-fit $\sigma$'
    Ylabel = '$n / (N\Delta_{bin})$'
    SbS = v[vNames.index('SbestSig')]
    bS = v[vNames.index('bestSig')]
    Xmin = -0.1
    Xmax = np.max(bS) 
    plot2classHistograms(SbS, bS, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=2.5, Xlabel=Xlabel, Ylabel=Ylabel, bins=20, Climit=50)


    print 'galaxy:'
    print 'C1:', np.min(C1), np.median(C1), np.max(C1)
    print 'C2:', np.min(C2), np.median(C2), np.max(C2)
    print 'C3:', np.min(C3), np.median(C3), np.max(C3)
    print 'C4:', np.min(bS), np.median(bS), np.max(bS)

    print 'star:'
    print 'C1:', np.min(sC1), np.median(sC1), np.max(sC1)
    print 'C2:', np.min(sC2), np.median(sC2), np.max(sC2)
    print 'C3:', np.min(sC3), np.median(sC3), np.max(sC3)
    print 'C4:', np.min(SbS), np.median(SbS), np.max(SbS)


    print 'galaxy C1-C4rms:', np.std(C1), np.std(C2), np.std(C3), np.std(bS)
    print '  star C1-C4rms:', np.std(sC1), np.std(sC2), np.std(sC3), np.std(SbS)

    bSigErr = v[vNames.index('bestSigErr')]
    SbSigErr = v[vNames.index('SbestSigErr')]
    print 'galaxy <C4err>:', np.median(bSigErr), '    and rms:', np.std(bSigErr) 
    print '  star <C4err>:', np.median(SbSigErr),'    and rms:', np.std(SbSigErr)







def plotChi2(sigtrue=1.0, sigmaNoise=5.0):

	muXtrue = 0.0
	muYtrue = 0.0
	Atrue = 1000.0
	Btrue = 0.0
	sigmaPSF = 1.5

	oDpix, nonoise, psf, image, diffim = make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigmaNoise, 2)

	return oDpix, nonoise, psf, image, diffim



def getAnalyticFunctions(sigG, sigmaPSF):

        ## defaults everywhere
	muTx = 0
        ## for 4 param testing
        # muTx = 0.4999
        muTy = 0
        Btrue = 0.0
	Atrue = 1000.0 

        # define the (square) grid
        oneDpixels = np.linspace(-7, 7, 15)

        # return vectors
        Sp2 = 0*sigG
        Sg2 = 0*sigG
        Spg = 0*sigG

        for i in range(0,sigG.size):
            ## make psf (sigtrue=0) 
            psf = gauss2D(muTx, muTy, 0, Atrue, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
            ## make noiseless image (convolved with psf, size given by 1Dpixels) 
            gimage = gauss2D(muTx, muTy, sigG[i], Atrue, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
            # compute sums psf^2, g^2 and psf*g
            psfNorm = psf / np.sum(psf)
            gNorm = gimage / np.sum(gimage)
            Sp2[i] = np.sum(psfNorm**2)
            Sg2[i] = np.sum(gNorm**2)
            Spg[i] = np.sum(psfNorm*gNorm)
 
	return Sp2, Sg2, Spg



# compute all classifiers 
def getAnalyticClassifiers(sigG, sigmaPSF): 

    Sp2, Sg2, Spg = getAnalyticFunctions(sigG, sigmaPSF)
    # aux quantities 
    CmodCpsf = Sp2/Spg
    NmodNpsf = Sp2/Sg2
    eta = np.sqrt(Sp2*Sg2)/Spg

    # classifiers
    Csdss = 2.5*np.log10(CmodCpsf)
    Csebok = CmodCpsf / np.sqrt(NmodNpsf)
    Cspread = eta*Csebok - 1
 
    return Csdss, Csebok, Cspread




# compute Cspread 
def getAnalyticCspread(sigG, sigmaPSF, sigmaSpread): 

        ## defaults everywhere
	muTx = 0
        ## for 4 param testing
        # muTx = 0.4999
        muTy = 0
        Btrue = 0.0
	Atrue = 1000.0 

        # define the (square) grid
        oneDpixels = np.linspace(-7, 7, 15)

        # return vectors
        SpgS = 0*sigmaSpread
        SggS = 0*sigmaSpread

        ## make psf (sigtrue=0) 
        psf = gauss2D(muTx, muTy, 0, Atrue, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        psfNorm = psf / np.sum(psf)
        ## make noiseless image for sigG (convolved with psf, size given by 1Dpixels) 
        gimage = gauss2D(muTx, muTy, sigG, Atrue, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
        gNorm = gimage / np.sum(gimage)
        Sp2 = np.sum(psfNorm**2)
        Spg = np.sum(psfNorm*gNorm)

        for i in range(0,sigmaSpread.size):
            ## make noiseless image (convolved with psf, size given by 1Dpixels) 
            gimageSpread = gauss2D(muTx, muTy, sigmaSpread[i], Atrue, Btrue, sigmaPSF, oneDpixels[:, np.newaxis], oneDpixels)
            gSpreadNorm = gimageSpread / np.sum(gimageSpread)
            # compute sums psf^2, g^2 and psf*g
            SpgS[i] = np.sum(psfNorm*gSpreadNorm)
            SggS[i] = np.sum(gNorm*gSpreadNorm)

        Cspread = Sp2/Spg * SggS/SpgS - 1
	return Cspread


### analyze problems with MCMC Bayes Factor
def testBF(sigma_m=15.0, sigtrue=1.0):

	muXtrue = 0.0
	muYtrue = 0.0
	Atrue = 1000.0
	Btrue = 0.0
	sigmaPSF = 1.5

         ## get images
	oneDpixels, nonoise, psf, image, diffimage = make4panelPlot(muXtrue, muYtrue, sigtrue, Atrue, Btrue, sigmaPSF, sigma_m)

        ## get chi2 image by brute force 
        chiPixSig, chiPixCmod, chi2im, bestModel, sigmaML, CmodML, chi2min = \
	    getChi2image(oneDpixels, image, muXtrue, muYtrue, Btrue, sigmaPSF, sigma_m)
	print 'DIRECT:'
	print ' sigmaML = ', sigmaML, 'chi2min = ', chi2min
        # chi2min =  254.471603091


	# compute Cmod from bestModel and compare to CmodML
	modelNorm = bestModel / np.sum(bestModel)
	wModel = modelNorm / np.sum(modelNorm**2)
	CountModel = np.sum(image*wModel)
	print ' CmodML = ', CmodML, ' Cmod from profile = ', CountModel

	# chi2-based plots
	chi2plot(oneDpixels, image, bestModel, chiPixSig, chiPixCmod, chi2im, sigtrue, sigmaML, CmodML)
	chi2plotMarginal(chiPixSig, chiPixCmod, chi2im, sigtrue, sigmaML, CmodML)

	# compute maxL ratio and Bayes Factor 
	maxLikeRat, BayesFactorDirect = chi2image2BF(chiPixSig, chi2im) 
	lnLambda = np.log(maxLikeRat)
	lnBF = 2*np.log(BayesFactorDirect)
	print ' lnLambda=', lnLambda,' ln(BayesFactor)=',lnBF
	# lnLambda= 558.694577378  ln(BayesFactor)= 836.482158919


	# now do MCMC
#	modelNorm, bestA, bestARMS, bestSigma, bestSigmaRMS, chi2min, iBest, BF, dBF, traces, fit_vals  = \
#	    getBestModelFitMCMCBF(oneDpixels, image, sigmaPSF, sigma_m, params5=0)
	# BF, dBF= -339.742435816 0.932741825725
        ## STAR
#	modelNorm, bestA, bestARMS, bestSigma, bestSigmaRMS, chi2min, iBest, BF, dBF, traces, fit_vals  = \
#	    getBestModelFitMCMCBF(oneDpixels, image, sigmaPSF, sigma_m, params5=0)
	# BF, dBF= -330.657492679 
#	print 'MCMC:'
#	print ' sigmaML = ', sigmaML, 'chi2min = ', chi2min

	return oneDpixels, image, sigmaPSF, sigma_m, muXtrue, muYtrue, Btrue



## this is modified Jake's code from
## https://gist.github.com/jakevdp/a7b3c47d605ab1d47af5
def estimateEvidence(traces, logp, r=0.05, return_list=False,
                          old_version=False, normalize_space=True):
    """Estimate the model evidence using the local density of points"""
    D, N = traces.shape
    
    if normalize_space:
        traces = traces.copy()
        for i in range(traces.shape[0]):
            #traces[i] /= traces[i].std()
            traces[i] /= sigmaG(traces[i])
 
    if old_version:
        # use neighbor count within r as a density estimator
        bt = BallTree(traces.T)
        count = bt.query_radius(traces.T, r=r, count_only=True)
 
        # compute volume of a D-dimensional sphere of radius r
        Vr = np.pi ** (0.5 * D) / gamma(0.5 * D + 1) * (r ** D)
 
        BF = logp + np.log(N) + np.log(Vr) - np.log(count)
    else:
        bt = BallTree(traces.T)
        log_density = bt.kernel_density(traces.T, r, return_log=True)
 
        BF = logp + np.log(N) - log_density
 
    if return_list:
        return BF
    else:
        p25, p50, p75 = np.percentile(BF, [25, 50, 75])
        return p50, 0.7413 * (p75 - p25)





##########################################################################
### joint tests of Bayes Factor and spread_model using direct method
def readCvsSNR(foutRoot, i1, i2, Cname, makePlots=0):
 
    Nsteps = i2-i1+1
    SNR = np.linspace(0, 1, Nsteps)
    SNRrms = 0*SNR 
    CEvec = 0*SNR 
    Cname = Cname - 1

    for i in range(i1, i2+1):
        Cvec = []
        sCvec = []
        datafile = foutRoot + str(i) + '.dat'
	v = np.loadtxt(datafile)
        print 'read', datafile
   	vNames = ['seta', 'sC1', 'sC2', 'sC3', 'SneffPSF', 'SneffModel', 'Schi2PSF', 'Schi2Model',  'sCmod', 'SbestA', \
			  'SbestARMS', 'SbestSig', 'SbestSigErr']
	vNames = vNames + ['eta', 'C1', 'C2', 'C3', 'neffPSF', 'neffModel', 'chi2PSF', 'chi2Model', 'Cmod', 'bestA', \
			  'bestARMS', 'bestSig', 'bestSigErr']
        #--------------------
	# names for classification quantities (order as in the paper) 
        # not needed here, but as a reminder...
	C1name = '$C_{SDSS}=m_{psf}-m_{mod}$'
	C2name = '$C_{Sebok}$'
	C3name = '$\Delta \chi^2 = \chi^2_{psf}-\chi^2_{mod}$'
	C4name = '$C_{Bayes}$'
	C5name = '$C_{spread}$'
	C6name = '$\sigma \,\, (pixel)$'

        # corresponding vectors (order not corresponding to the above for "historic" reasons)
        # model
        neffpsf = v[vNames.index('neffPSF')]
        neffmod = v[vNames.index('neffModel')]
        CmodCpsf = v[vNames.index('C2')]
        Cmod = v[vNames.index('Cmod')]
        Cpsf = Cmod / CmodCpsf
        SNRsigma = np.median(Cpsf / np.sqrt(neffpsf)) 
        SNRsigmaRMS = np.std(Cpsf / np.sqrt(neffpsf)) 
        # this is the product of SNR and used sigmaNoise 
        SNR[i-i1] = SNRsigma
        SNRrms[i-i1] = SNRsigmaRMS

        eta = v[vNames.index('eta')]
        Cvec.append(2.5*np.log10(v[vNames.index('C2')]))
        Cvec.append(v[vNames.index('C2')] * np.sqrt(neffpsf/neffmod))
        Cvec.append(v[vNames.index('C3')])
        Cvec.append(v[vNames.index('C1')])
        Cvec.append(eta*Cvec[1] - 1)
        Cvec.append(v[vNames.index('bestSig')])
        # psf
        Sneffpsf = v[vNames.index('SneffPSF')]
        Sneffmod = v[vNames.index('SneffModel')]
        seta = v[vNames.index('seta')]
        sCvec.append(2.5*np.log10(v[vNames.index('sC2')]))
        sCvec.append(v[vNames.index('sC2')] * np.sqrt(Sneffpsf/Sneffmod))
        sCvec.append(v[vNames.index('sC3')])
        sCvec.append(v[vNames.index('sC1')])
        sCvec.append(seta*sCvec[1] - 1)
        sCvec.append(v[vNames.index('SbestSig')])

        # given classification parameter requested via Cname, compute 
        # the value of completeness that equals purity for equal-sized samples 
        # by construction, it doesn't matter which subsample is used 
        CE = getCeqE(sCvec[Cname], Cvec[Cname])
        CEvec[i-i1] = CE
        print 'i=', i, SNRsigma, CE

    return SNR, SNRrms, CEvec, sCvec[Cname], Cvec[Cname]




def getCeqE(Cs, Cg): 

    Ngrid = 1000
    Xmin = np.min(Cs)
    Xmax = np.max(Cg)
    Cgrid = np.linspace(Xmin, Xmax, Ngrid)
    Cthreshold = -1.0
    for i in range(0,Ngrid): 
        C = Cgrid[i]
        CsOK = Cs[Cs > C]
        CgOK = Cg[Cg > C]
        if (CgOK.size>0):
            ComplG = 100.0*CgOK.size / Cg.size
            PurityG = 100.0*CgOK.size / (CsOK.size + CgOK.size)
            if ((Cthreshold < 0) & (ComplG < PurityG)):
                Cthreshold = ComplG

    return Cthreshold
 
