import matplotlib.pyplot as plt
import numpy as np
from astroML.stats import binned_statistic_2d
from astroML.plotting.mcmc import convert_to_stdev
from tools2Dgauss import *


def FourPanelPlot(oneDpixels, nonoise, psf, image, diffimage):

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.90, wspace=0.18, hspace=0.46)

    ## noiseless image
    ax = fig.add_subplot(221)
    ax.set_title('noiseless image', fontsize=14)
    plt.imshow(nonoise, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')
    plt.clim(-20, 100)
    plt.colorbar().set_label('')

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## psf image
    ax = fig.add_subplot(223)
    ax.set_title('psf image', fontsize=14)
    plt.imshow(psf, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')
    plt.clim(-20, 100)
    plt.colorbar().set_label('')

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## image with noise 
    ax = fig.add_subplot(222)
    ax.set_title('image with noise', fontsize=14)
    plt.imshow(image, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')

    plt.clim(-20, 100)
    plt.colorbar().set_label(r'$counts$', fontsize=14)

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## difference object - psf 
    ax = fig.add_subplot(224)
    ax.set_title('image - psf', fontsize=14)
    plt.imshow(diffimage, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')

    plt.clim(-50, 50)
    plt.colorbar().set_label(r'$counts$', fontsize=14)

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    plt.show()


def dmagPlot(SNRmod1, dmag1, dmagStd1, SNRmod2, dmag2, dmagStd2):

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.90, wspace=0.18, hspace=0.46)

    ## 
    ax = fig.add_subplot(111)
    ax.set_title('$(m_{psf}-m_{mod}) \pm \sigma$', fontsize=14)

    ax.plot(1/SNRmod1, dmag1+dmagStd1, '-r', label=r'$\theta_{in}=\theta_{psf}/4$')
    ax.plot(1/SNRmod1, dmag1-dmagStd1, '-r', label=r'$\theta_{in}=\theta_{psf}/4$')
    ax.plot(1/SNRmod1, dmag1, '--r', label=r'$\theta_{in}=\theta_{psf}/4$')
    ax.plot(1/SNRmod2, dmag2+dmagStd2, '-b', label=r'$\theta_{in}=\theta_{psf}/2$')
    ax.plot(1/SNRmod2, dmag2-dmagStd2, '-b', label=r'$\theta_{in}=\theta_{psf}/2$')
    ax.plot(1/SNRmod2, dmag2, '--b', label=r'$\theta_{in}=\theta_{psf}/2$')

    ax.set_xlabel(r'1/SNR', fontsize=12)
    ax.set_ylabel(r'$(m_{psf}-m_{mod}) \pm \sigma$', fontsize=12)

    plt.show()



# e.g.
# figCcomparison('SGsimpleMCMC_2params_m15sigt1_3000.dat')

def figCcomparison(datafile, name = None, title=''):

        #### read data    VOLATILE: assumes SGsimple*dat files with the vectors below
	v = np.loadtxt(datafile)
	vNames = ['sC1', 'sC2', 'sC3', 'SneffPSF', 'SneffModel', 'Schi2PSF', 'Schi2Model',  'sCmod', 'SbestA', \
			  'SbestARMS', 'SbestSig', 'SbestSigErr']
	vNames = vNames + ['C1', 'C2', 'C3', 'neffPSF', 'neffModel', 'chi2PSF', 'chi2Model', 'Cmod', 'bestA', \
			  'bestARMS', 'bestSig', 'bestSigErr']

	def plotPanel(sX, sY, gX, gY, xMin, xMax, yMin, yMax, xLabel, yLabel, title=''): 

                ## bin (astroML code) 
		# axes limits
		range = np.zeros((2,2))
		range[0,0]=xMin
		range[0,1]=xMax
		range[1,0]=yMin
		range[1,1]=yMax
                NS, xedgesS, yedgesS = binned_statistic_2d(sX, sY, sY,'count', bins=20, range=range)
                NG, xedgesG, yedgesG = binned_statistic_2d(gX, gY, gY,'count', bins=20, range=range)

		## plot

                ## galaxies are blue contours
		levels = np.linspace(0, np.log10(NG.max()), 7)[2:]
		# plt.contour(np.log10(NG.T), levels, colors='b', linewidths=2, extent=[xedgesG[0], xedgesG[-1], yedgesG[0], yedgesG[-1]])
		plt.scatter(gX, gY, color='blue', s=5, linewidths=1, alpha=0.2)


                # stars are copper continuous map 
		cmap = plt.cm.copper
		cmap.set_bad('w', 0.0)
		# plt.imshow(np.log10(NS.T), origin='lower', extent=[xedgesS[0], xedgesS[-1], yedgesS[0], yedgesS[-1]], aspect='auto', interpolation='nearest', cmap=cmap)
		plt.scatter(sX, sY, color='red', s=10, linewidths=1, alpha=0.25)

		plt.xlim(xMin, xMax)
		plt.ylim(yMin, yMax)
		plt.xlabel(xLabel, fontsize=16)
		plt.ylabel(yLabel, fontsize=16)

		xTitle = xMin + 0.05*(xMax-xMin)
		yTitle = yMax + 0.05*(yMax-yMin)
		ax.text(xTitle, yTitle, title)

 
        #--------------------
	# names for classification quantities
	C1name = '$C_{Sebok}$'
	C2name = '$m_{psf}-m_{mod}$'
	C3name = '$\chi^2_{psf}-\chi^2_{mod}$'
	C4name = '$\sigma \,\, (pixel)$'
	C1 = v[vNames.index('C1')]
	C2 = 2.5*np.log10(v[vNames.index('C2')])
	C3 = v[vNames.index('C3')]
	C4 = v[vNames.index('bestSig')]
	sC1 = v[vNames.index('sC1')]
	sC2 = 2.5*np.log10(v[vNames.index('sC2')])
	sC3 = v[vNames.index('sC3')]
	sC4 = v[vNames.index('SbestSig')]

        ## Create figure and subplots
	fig = plt.figure(figsize=(8, 8))
	fig.subplots_adjust(wspace=0.25, hspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

	C2min = -0.05
	C2max = 0.7
	# 
	ax = plt.subplot(321)
	xLabel = C2name
	yLabel = C1name
	xMin = C2min
	xMax= C2max
	yMin=0.99
	yMax=1.12
        yMin = np.min(sC1) 
        yMax = np.max(C1) 
        yMax = 20.0 
        yMin = -5.0 
	plotPanel(sC2, sC1, C2, C1, xMin, xMax, yMin, yMax, xLabel, yLabel, title=title)

	# 
	ax = plt.subplot(322)
	xLabel = C2name
	yLabel = C3name
	xMin = C2min
	xMax= C2max
	yMin=-2
	yMax=22
	plotPanel(sC2, sC3, C2, C3, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(323)
	xLabel = C2name
	yLabel = C4name
	xMin = C2min
	xMax= C2max
	yMin=-0.05
	yMax=1.95
	plotPanel(sC2, sC4, C2, C4, xMin, xMax, yMin, yMax, xLabel, yLabel)


	# 
	ax = plt.subplot(324)
	xLabel = C1name
	yLabel = C3name
	xMin = 0.99
	xMax= 1.12
        xMin = np.min(sC1) 
        xMax = np.max(C1) 
        xMin = -5.0
        xMax = 20.0 
	yMin=-2
	yMax=22
	plotPanel(sC1, sC3, C1, C3, xMin, xMax, yMin, yMax, xLabel, yLabel)


	if (name is None):
		plt.show() 
	else:
		print 'saving plot to:', name
		plt.savefig(name, bbox_inches='tight')



def figClassification(datafile, name = None, title=''):

        #### read data    VOLATILE: assumes SGsimple*dat files with the vectors below
	v = np.loadtxt(datafile)
   	vNames = ['sC1', 'sC2', 'sC3', 'SneffPSF', 'SneffModel', 'Schi2PSF', 'Schi2Model',  'sCmod', 'SbestA', \
			  'SbestARMS', 'SbestSig', 'SbestSigErr']
	vNames = vNames + ['C1', 'C2', 'C3', 'neffPSF', 'neffModel', 'chi2PSF', 'chi2Model', 'Cmod', 'bestA', \
			  'bestARMS', 'bestSig', 'bestSigErr']


	def plot2Chistograms(Cs, Cg, Xmin, Xmax, Ymin, Ymax, Xlabel, Ylabel, bins=20, title=''):
		limits = [(Xmin, Xmax, Ymin, Ymax)]
		labels = [Xlabel, Ylabel]
		ax.set_xlim(Xmin, Xmax)
		ax.set_ylim(Ymin, Ymax)
		ax.set_xlabel(Xlabel, fontsize=12)
		ax.set_ylabel(Ylabel, fontsize=12)
		plt.tick_params(axis='both', which='major', labelsize=15)
		xTitle = Xmin + 0.05*(Xmax-Xmin)
		yTitle = Ymax + 0.05*(Ymax-Ymin)
		ax.text(xTitle, yTitle, title)

		# plot a histogram
		ax.hist(Cs, bins=bins, normed=True, facecolor='red', histtype='stepfilled', alpha=0.4)
		ax.hist(Cg, bins=bins, normed=True, facecolor='blue', histtype='stepfilled', alpha=0.4)

  	def plotROC(Cs, Cg, Climit=50, title=''):
		ax.set_xlim(Climit, 100)
		ax.set_ylim(Climit, 100)
		ax.set_xlabel('completeness (\%)', fontsize=12)
		ax.set_ylabel('purity (\%)', fontsize=12)
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

		ax.plot(ComplS, PurityS, '-r', lw=2)  
		ax.plot(ComplG, PurityG, '-b', lw=2) 
		for CL in [50, 60, 70, 80, 90]:
			ax.plot([CL, CL], [0.0, 100], '-k', lw=1)
			ax.plot([0, 100], [CL, CL], '-k', lw=1)
			ax.plot([CL+5, CL+5], [0.0, 100], '--k', lw=1)
			ax.plot([0, 100], [CL+5, CL+5], '--k', lw=1)


        #--------------------
	# names for classification quantities
	C1name = '$C_{Sebok}$'
	C2name = '$m_{psf}-m_{mod}$'
	C3name = '$\chi^2_{psf}-\chi^2_{mod}$'
	C4name = '$\sigma \,\, (pixel)$'
	C1 = v[vNames.index('C1')]
	C2 = 2.5*np.log10(v[vNames.index('C2')])
	C3 = v[vNames.index('C3')]
	C4 = v[vNames.index('bestSig')]
	sC1 = v[vNames.index('sC1')]
	sC2 = 2.5*np.log10(v[vNames.index('sC2')])
	sC3 = v[vNames.index('sC3')]
	sC4 = v[vNames.index('SbestSig')]

        ## Create figure and subplots
	fig = plt.figure(figsize=(8, 9))
	fig.subplots_adjust(wspace=0.27, hspace=0.35, left=0.12, right=0.94, bottom=0.05, top=0.95)

        Climit = 66
        Climit = 46
        # Climit = 0
	bins = 40 

	## chi2 classification 
	# plot histograms
	ax = plt.subplot(421)
	Xlabel = C3name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = np.min(sC3) - 1
        Xmax = np.max(C3) 
        Xmax = 25.0  
        plot2Chistograms(sC3, C3, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=0.5, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title=title)
	# plot ROC curves
	ax = plt.subplot(422)
  	plotROC(sC3, C3, Climit=Climit, title='')


	## Csebok classification 
	# plot histograms
	ax = plt.subplot(423)
	Xlabel = C1name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = np.min(sC1) - 0.01
        Xmax = np.max(C1) 
	# Xmax = 1.17
        plot2Chistograms(sC1, C1, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=0.5, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
        # plot2Chistograms(sC1, C1, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=45.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(424)
  	plotROC(sC1, C1, Climit=Climit, title='')


	## Cmod/Cpsf classification 
	# plot histograms
	ax = plt.subplot(425)
	Xlabel = C2name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = np.min(sC2) - 0.02
        Xmax = np.max(C2) 
        Xmax = 0.7
        plot2Chistograms(sC2, C2, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=15.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(426)
  	plotROC(sC2, C2, Climit=Climit, title='')


	## best sigma classification 
	# plot histograms
	ax = plt.subplot(427)
	Xlabel = C4name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = np.min(sC4) - 0.2
        Xmax = np.max(C4) 
        Xmax = 2.2
        plot2Chistograms(sC4, C4, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=5.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(428)
  	plotROC(sC4, C4, Climit=Climit, title='')

	if (name is None):
		plt.show() 
	else:
		print 'saving plot to:', name
		plt.savefig(name, bbox_inches='tight')


def chi2plot(oneDpixels, image, bestModel, chiPixSig, chiPixCmod, chi2image, sigtrue, sigmaML, CmodML):

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.08, bottom=0.15, right=0.95, top=0.90, wspace=0.29, hspace=0.46)

    ## image with noise 
    ax = fig.add_subplot(221)
    ax.set_title('data image', fontsize=14)
    plt.imshow(image, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')

    plt.clim(-20, 100)
    plt.colorbar().set_label(r'$counts$', fontsize=14)

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## chi2 image 
    ax = fig.add_subplot(222)
    ax.set_title('ln($\chi^2_{dof}$) image', fontsize=14)
    Lchi2image = np.log(chi2image/image.size)
    # pretty color map
    plt.imshow(Lchi2image, origin='lower', 
           extent=(chiPixSig[0], chiPixSig[-1], chiPixCmod[0], chiPixCmod[-1]),
	   cmap=plt.cm.RdYlGn, aspect='auto')
    # mark true values     
    ax.plot(sigtrue, 1000.0, 'o', color='blue', alpha=0.75)
    # mark ML solution: (sigmaML, CmodML)
    ax.plot(sigmaML, CmodML, '+', color='blue', alpha=0.99)
    print 'chi2plot: sigtrue, sigmaML, CmodML=', sigtrue, sigmaML, CmodML
    # legend
    plt.clim(np.min(Lchi2image), np.max(Lchi2image))
    plt.colorbar().set_label(r'ln($\chi^2_{dof}$)', fontsize=14)
    # contours
    plt.contour(chiPixSig, chiPixCmod, convert_to_stdev(Lchi2image),
            levels=(0.683, 0.955, 0.997),
            colors='k')

    ax.set_xlabel(r'$\sigma$ (pixel)', fontsize=12)
    ax.set_ylabel(r'$C_{mod}$ (counts)', fontsize=12)



    ## best-fit model image 
    ax = fig.add_subplot(223)
    ax.set_title('best-fit model', fontsize=14)
    plt.imshow(bestModel, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')

    plt.clim(-20, 100)
    plt.colorbar().set_label(r'$counts$', fontsize=14)

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    ## residual difference image - best-fit model 
    ax = fig.add_subplot(224)
    ax.set_title('data - model residuals', fontsize=14)
    diffimage = image - bestModel
    plt.imshow(diffimage, origin='lower', interpolation='nearest',
           extent=(oneDpixels[0], oneDpixels[-1], oneDpixels[0], oneDpixels[-1]),
           cmap=plt.cm.binary, aspect='auto')

    plt.clim(-60, 60)
    plt.colorbar().set_label(r'$counts$', fontsize=14)

    ax.set_xlabel(r'x (pixels)', fontsize=12)
    ax.set_ylabel(r'y (pixels)', fontsize=12)

    name = None
    if (name is None):
       	plt.show() 
    else:
        print 'saving plot to:', name
       	plt.savefig(name, bbox_inches='tight')



def chi2plotMarginal(chiPixSig, chiPixCmod, chi2image, sigtrue, sigmaML, CmodML):

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
           extent=(chiPixSig[0], chiPixSig[-1], chiPixCmod[0], chiPixCmod[-1]),
	   cmap=plt.cm.RdYlGn, aspect='auto')

    # colorbar
    cax = plt.axes([0.82, 0.35, 0.02, 0.6])
    cb = plt.colorbar(cax=cax)
    cb.set_label(r'$lnL(\sigma, C_{mod})$', fontsize=14)
    plt.clim(np.min(lnL), np.max(lnL))

    # contours  WHY IS THIS NOT WORKING?? 
    plt.contour(chiPixSig, chiPixCmod, convert_to_stdev(lnL),
            levels=(0.683, 0.955, 0.997),
            colors='k')

    # mark true values     
    ax.plot(sigtrue, 1000.0, 'o', color='red', alpha=0.75)
    # mark ML solution: (sigmaML, CmodML)
    ax.plot(sigmaML, CmodML, 'x', color='white', alpha=0.99, lw=35)



    # compute marginal projections
    p_sigma = np.exp(lnL).sum(0)
    p_Cmod = np.exp(lnL).sum(1)
    # and p(C|sigma=0)
    L = np.exp(lnL)
    L0 = L[:,0]
    pCmod0 = L0 / np.max(L0) * np.max(p_Cmod)

    ax1 = fig.add_axes([0.35, 0.1, 0.45, 0.23], yticks=[])
    ax1.plot(chiPixSig, p_sigma, '-k')
    ax1.set_xlabel(r'$\sigma$ (pixel)', fontsize=12)
    ax1.set_ylabel(r'$p(\sigma)$', fontsize=12)
    ax1.set_xlim(np.min(chiPixSig), np.max(chiPixSig))

    ax2 = fig.add_axes([0.15, 0.35, 0.18, 0.6], xticks=[])
    ax2.plot(p_Cmod, chiPixCmod, '-k')
    ax2.plot(pCmod0, chiPixCmod, '--b')
    ax2.set_ylabel(r'$C_{mod}$ (counts)', fontsize=12)
    ax2.set_xlabel(r'$p(C_{mod})$', fontsize=12)
    ax2.set_xlim(ax2.get_xlim()[::-1])  # reverse x axis
    ax2.set_ylim(np.min(chiPixCmod), np.max(chiPixCmod))

    name = None
    if (name is None):
       	plt.show() 
    else:
        print 'saving plot to:', name
       	plt.savefig(name, bbox_inches='tight')





def figCcomparison2v0(datafile, name = None, title=''):

        #### read data    VOLATILE: assumes SGall*dat files with the vectors below
	v = np.loadtxt(datafile)
	vNames = ['seta', 'sC1', 'sC2', 'sC3', 'SneffPSF', 'SneffModel', 'Schi2PSF', 'Schi2Model',  'sCmod', 'SbestA', \
			  'SbestARMS', 'SbestSig', 'SbestSigErr']
	vNames = vNames + ['eta', 'C1', 'C2', 'C3', 'neffPSF', 'neffModel', 'chi2PSF', 'chi2Model', 'Cmod', 'bestA', \
			  'bestARMS', 'bestSig', 'bestSigErr']

	def plotPanel(sX, sY, gX, gY, xMin, xMax, yMin, yMax, xLabel, yLabel, title=''): 

                ## bin (astroML code) 
		# axes limits
		range = np.zeros((2,2))
		range[0,0]=xMin
		range[0,1]=xMax
		range[1,0]=yMin
		range[1,1]=yMax
                NS, xedgesS, yedgesS = binned_statistic_2d(sX, sY, sY,'count', bins=20, range=range)
                NG, xedgesG, yedgesG = binned_statistic_2d(gX, gY, gY,'count', bins=20, range=range)

		## plot

                ## galaxies are blue contours
		levels = np.linspace(0, np.log10(NG.max()), 7)[2:]
		# plt.contour(np.log10(NG.T), levels, colors='b', linewidths=2, extent=[xedgesG[0], xedgesG[-1], yedgesG[0], yedgesG[-1]])
		plt.scatter(gX, gY, color='blue', s=5, linewidths=1, alpha=0.2)


                # stars are copper continuous map 
		cmap = plt.cm.copper
		cmap.set_bad('w', 0.0)
		# plt.imshow(np.log10(NS.T), origin='lower', extent=[xedgesS[0], xedgesS[-1], yedgesS[0], yedgesS[-1]], aspect='auto', interpolation='nearest', cmap=cmap)
		plt.scatter(sX, sY, color='red', s=10, linewidths=1, alpha=0.25)

		plt.xlim(xMin, xMax)
		plt.ylim(yMin, yMax)
		plt.xlabel(xLabel, fontsize=16)
		plt.ylabel(yLabel, fontsize=16)

		xTitle = xMin + 0.05*(xMax-xMin)
		yTitle = yMax + 0.05*(yMax-yMin)
		ax.text(xTitle, yTitle, title)

        #--------------------
	# names for classification quantities (order as in the paper) 
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
        eta = v[vNames.index('eta')]
        C1 = 2.5*np.log10(v[vNames.index('C2')])
        C2 = v[vNames.index('C2')] * np.sqrt(neffpsf/neffmod)
        C3 = v[vNames.index('C3')]
        C4 = v[vNames.index('C1')]
        C5 = eta*C2 - 1
	C6 = v[vNames.index('bestSig')]
        # psf
        Sneffpsf = v[vNames.index('SneffPSF')]
        Sneffmod = v[vNames.index('SneffModel')]
        seta = v[vNames.index('seta')]
        sC1 = 2.5*np.log10(v[vNames.index('sC2')])
        sC2 = v[vNames.index('sC2')] * np.sqrt(Sneffpsf/Sneffmod)
        sC3 = v[vNames.index('sC3')]
        sC4 = v[vNames.index('sC1')]
        sC5 = seta*sC2 - 1
	sC6 = v[vNames.index('SbestSig')]
 
        print '<eta>+-rms, min/max = ', np.median(eta), np.std(eta), np.min(eta), np.max(eta)
        print '<Seta>+-rms, min/max = ', np.median(seta), np.std(seta), np.min(seta), np.max(seta)

        print 'CBayes min/max:', np.min(C4), np.max(C4)

        # axes limits
	C1min = -0.05
	C1max = 0.7
	C2min = 0.99
	C2max = 1.12 
	C3min = -2.0
	C3max = 22.0 
	C4min = -5.0 
	C4max = 25.0
	C5min = -0.01 
	C5max = 0.19
	C6min = -0.05
	C6max = 1.9

        ## Create figure and subplots
	fig = plt.figure(figsize=(8, 8))
	fig.subplots_adjust(wspace=0.25, hspace=0.25, left=0.1, right=0.95, bottom=0.12, top=0.95)

	# 
	ax = plt.subplot(321)
	xLabel = C1name
	yLabel = C4name
	xMin = C1min
	xMax= C1max
	yMin=C4min
	yMax=C4max
	plotPanel(sC1, sC4, C1, C4, xMin, xMax, yMin, yMax, xLabel, yLabel, title=title)

	# 
	ax = plt.subplot(322)
	xLabel = C1name
	yLabel = C3name
	xMin = C1min
	xMax= C1max
	yMin= C3min
	yMax= C3max
	plotPanel(sC1, sC3, C1, C3, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(323)
	xLabel = C1name
	yLabel = C6name
	xMin = C1min
	xMax= C1max
	yMin= C6min
	yMax= C6max
	plotPanel(sC1, sC6, C1, C6, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(324)
	xLabel = C4name
	yLabel = C3name
	xMin = C4min
	xMax= C4max
	yMin= C3min
	yMax= C3max 
	plotPanel(sC4, sC3, C4, C3, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(325)
	xLabel = C4name
	yLabel = C2name
	xMin = C4min
	xMax= C4max
	yMin= C2min
	yMax= C2max
	plotPanel(sC4, sC2, C4, C2, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(326)
	xLabel = C4name
	yLabel = C5name
	xMin = C4min
	xMax= C4max
	yMin= C5min
	yMax= C5max 
	plotPanel(sC4, sC5, C4, C5, xMin, xMax, yMin, yMax, xLabel, yLabel)

	if (name is None):
		plt.show() 
	else:
		print 'saving plot to:', name
		plt.savefig(name, bbox_inches='tight')
               
        if (0):
            Cmod = v[vNames.index('Cmod')]
            bestA = v[vNames.index('bestA')]
            sCmod = v[vNames.index('sCmod')]
            SbestA = v[vNames.index('SbestA')]

            return Cmod, bestA, sCmod, SbestA

       



def figCcomparison2(datafile, name = None, title=''):

        #### read data    VOLATILE: assumes SGall*dat files with the vectors below
	v = np.loadtxt(datafile)
	vNames = ['seta', 'sC1', 'sC2', 'sC3', 'SneffPSF', 'SneffModel', 'Schi2PSF', 'Schi2Model',  'sCmod', 'SbestA', \
			  'SbestARMS', 'SbestSig', 'SbestSigErr']
	vNames = vNames + ['eta', 'C1', 'C2', 'C3', 'neffPSF', 'neffModel', 'chi2PSF', 'chi2Model', 'Cmod', 'bestA', \
			  'bestARMS', 'bestSig', 'bestSigErr']

	def plotPanel(sX, sY, gX, gY, xMin, xMax, yMin, yMax, xLabel, yLabel, title=''): 

                ## bin (astroML code) 
		# axes limits
		range = np.zeros((2,2))
		range[0,0]=xMin
		range[0,1]=xMax
		range[1,0]=yMin
		range[1,1]=yMax
                NS, xedgesS, yedgesS = binned_statistic_2d(sX, sY, sY,'count', bins=20, range=range)
                NG, xedgesG, yedgesG = binned_statistic_2d(gX, gY, gY,'count', bins=20, range=range)

		## plot

                ## galaxies are blue contours
		levels = np.linspace(0, np.log10(NG.max()), 7)[2:]
		# plt.contour(np.log10(NG.T), levels, colors='b', linewidths=2, extent=[xedgesG[0], xedgesG[-1], yedgesG[0], yedgesG[-1]])
		plt.scatter(gX, gY, color='blue', s=5, linewidths=1, alpha=0.2)


                # stars are copper continuous map 
		cmap = plt.cm.copper
		cmap.set_bad('w', 0.0)
		# plt.imshow(np.log10(NS.T), origin='lower', extent=[xedgesS[0], xedgesS[-1], yedgesS[0], yedgesS[-1]], aspect='auto', interpolation='nearest', cmap=cmap)
		plt.scatter(sX, sY, color='red', s=10, linewidths=1, alpha=0.25)

		plt.xlim(xMin, xMax)
		plt.ylim(yMin, yMax)
		plt.xlabel(xLabel, fontsize=16)
		plt.ylabel(yLabel, fontsize=16)

		xTitle = xMin + 0.05*(xMax-xMin)
		yTitle = yMax + 0.05*(yMax-yMin)
		ax.text(xTitle, yTitle, title)

        #--------------------
	# names for classification quantities (order as in the paper) 
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
        eta = v[vNames.index('eta')]
        C1 = 2.5*np.log10(v[vNames.index('C2')])
        C2 = v[vNames.index('C2')] * np.sqrt(neffpsf/neffmod)
        C3 = v[vNames.index('C3')]
        C4 = v[vNames.index('C1')]
        C5 = eta*C2 - 1
	C6 = v[vNames.index('bestSig')]
        # psf
        Sneffpsf = v[vNames.index('SneffPSF')]
        Sneffmod = v[vNames.index('SneffModel')]
        seta = v[vNames.index('seta')]
        sC1 = 2.5*np.log10(v[vNames.index('sC2')])
        sC2 = v[vNames.index('sC2')] * np.sqrt(Sneffpsf/Sneffmod)
        sC3 = v[vNames.index('sC3')]
        sC4 = v[vNames.index('sC1')]
        sC5 = seta*sC2 - 1
	sC6 = v[vNames.index('SbestSig')]
 
        print '<eta>+-rms, min/max = ', np.median(eta), np.std(eta), np.min(eta), np.max(eta)
        print '<Seta>+-rms, min/max = ', np.median(seta), np.std(seta), np.min(seta), np.max(seta)

        print 'CBayes min/max:', np.min(C4), np.max(C4)

        # axes limits
	C1min = -0.05
	C1max = 0.7
	C2min = 0.99
	C2max = 1.12 
	C3min = -2.0
	C3max = 22.0 
	C4min = -5.0 
	C4max = 25.0
	C5min = -0.01 
	C5max = 0.19
	C6min = -0.05
	C6max = 1.9

        ## Create figure and subplots
	fig = plt.figure(figsize=(8, 8))
	fig.subplots_adjust(wspace=0.23, hspace=0.28, left=0.1, right=0.95, bottom=0.12, top=0.95)

	# 
	ax = plt.subplot(321)
	xLabel = C1name
	yLabel = C6name
	xMin = C1min
	xMax= C1max
	yMin=C6min
	yMax=C6max
	plotPanel(sC1, sC6, C1, C6, xMin, xMax, yMin, yMax, xLabel, yLabel, title=title)

	# 
	ax = plt.subplot(322)
	xLabel = C3name
	yLabel = C4name
	xMin = C3min
	xMax= C3max
	yMin= C4min
	yMax= C4max
	plotPanel(sC3, sC4, C3, C4, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(323)
	xLabel = C1name
	yLabel = C3name
	xMin = C1min
	xMax= C1max
	yMin= C3min
	yMax= C3max
	plotPanel(sC1, sC3, C1, C3, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(324)
	xLabel = C3name
	yLabel = C5name
	xMin = C3min
	xMax= C3max
	yMin= C5min
	yMax= C5max 
	plotPanel(sC3, sC5, C3, C5, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(325)
	xLabel = C1name
	yLabel = C5name
	xMin = C1min
	xMax= C1max
	yMin= C5min
	yMax= C5max
	plotPanel(sC1, sC5, C1, C5, xMin, xMax, yMin, yMax, xLabel, yLabel)

	# 
	ax = plt.subplot(326)
	xLabel = C2name
	yLabel = C5name
	xMin = C2min
	xMax= C2max
	yMin= C5min
	yMax= C5max 
	plotPanel(sC2, sC5, C2, C5, xMin, xMax, yMin, yMax, xLabel, yLabel)

	if (name is None):
		plt.show() 
	else:
		print 'saving plot to:', name
		plt.savefig(name, bbox_inches='tight')
               
        if (0):
            Cmod = v[vNames.index('Cmod')]
            bestA = v[vNames.index('bestA')]
            sCmod = v[vNames.index('sCmod')]
            SbestA = v[vNames.index('SbestA')]

            return Cmod, bestA, sCmod, SbestA

       

def figClassification2(datafile, name = None, title=''):

        #### read data    VOLATILE: assumes SGsimple*dat files with the vectors below
	v = np.loadtxt(datafile)
   	vNames = ['seta', 'sC1', 'sC2', 'sC3', 'SneffPSF', 'SneffModel', 'Schi2PSF', 'Schi2Model',  'sCmod', 'SbestA', \
			  'SbestARMS', 'SbestSig', 'SbestSigErr']
	vNames = vNames + ['eta', 'C1', 'C2', 'C3', 'neffPSF', 'neffModel', 'chi2PSF', 'chi2Model', 'Cmod', 'bestA', \
			  'bestARMS', 'bestSig', 'bestSigErr']


	def plot2Chistograms(Cs, Cg, Xmin, Xmax, Ymin, Ymax, Xlabel, Ylabel, bins=20, title=''):
		limits = [(Xmin, Xmax, Ymin, Ymax)]
		labels = [Xlabel, Ylabel]
		ax.set_xlim(Xmin, Xmax)
		ax.set_ylim(Ymin, Ymax)
		ax.set_xlabel(Xlabel, fontsize=12)
		ax.set_ylabel(Ylabel, fontsize=12)
		plt.tick_params(axis='both', which='major', labelsize=15)
		xTitle = Xmin + 0.05*(Xmax-Xmin)
		yTitle = Ymax + 0.05*(Ymax-Ymin)
		ax.text(xTitle, yTitle, title)

		# plot a histogram
		ax.hist(Cs, bins=bins, normed=True, facecolor='red', histtype='stepfilled', alpha=0.4)
		ax.hist(Cg, bins=bins, normed=True, facecolor='blue', histtype='stepfilled', alpha=0.4)

  	def plotROC(Cs, Cg, Climit=50, title=''):
		ax.set_xlim(Climit, 100)
		ax.set_ylim(Climit, 100)
		ax.set_xlabel('completeness (\%)', fontsize=12)
		ax.set_ylabel('purity (\%)', fontsize=12)
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

		ax.plot(ComplS, PurityS, '-r', lw=3)  
		ax.plot(ComplG, PurityG, '-b', lw=3) 
		for CL in [50, 60, 70, 80, 90]:
			ax.plot([CL, CL], [0.0, 100], '-k', lw=1, alpha=0.4)
			ax.plot([0, 100], [CL, CL], '-k', lw=1, alpha=0.4)
			ax.plot([CL+5, CL+5], [0.0, 100], '--k', lw=1, alpha=0.4)
			ax.plot([0, 100], [CL+5, CL+5], '--k', lw=1, alpha=0.4)


        #--------------------
	# names for classification quantities (order as in the paper) 
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
        eta = v[vNames.index('eta')]
        C1 = 2.5*np.log10(v[vNames.index('C2')])
        C2 = v[vNames.index('C2')] * np.sqrt(neffpsf/neffmod)
        C3 = v[vNames.index('C3')]
        C4 = v[vNames.index('C1')]
        C5 = eta*C2 - 1
	C6 = v[vNames.index('bestSig')]
        # psf
        Sneffpsf = v[vNames.index('SneffPSF')]
        Sneffmod = v[vNames.index('SneffModel')]
        seta = v[vNames.index('seta')]
        sC1 = 2.5*np.log10(v[vNames.index('sC2')])
        sC2 = v[vNames.index('sC2')] * np.sqrt(Sneffpsf/Sneffmod)
        sC3 = v[vNames.index('sC3')]
        sC4 = v[vNames.index('sC1')]
        sC5 = seta*sC2 - 1
	sC6 = v[vNames.index('SbestSig')]
 
        print '<eta>+-rms, min/max = ', np.median(eta), np.std(eta), np.min(eta), np.max(eta)
        print '<Seta>+-rms, min/max = ', np.median(seta), np.std(seta), np.min(seta), np.max(seta)

        print 'CBayes min/max:', np.min(C4), np.max(C4)

        # axes limits
	C1min = -0.05
	C1max = 0.7
	C2min = 0.99
	C2max = 1.12 
	C3min = -2.0
	C3max = 22.0 
	C4min = -5.0 
	C4max = 25.0
	C5min = -0.01 
	C5max = 0.19
	C6min = -0.05
	C6max = 1.9

        ## Create figure and subplots
	fig = plt.figure(figsize=(8, 10))
	fig.subplots_adjust(wspace=0.27, hspace=0.51, left=0.12, right=0.94, bottom=0.05, top=0.95)

        Climit = 46
	bins = 40 


	## Cmod/Cpsf classification 
	# plot histograms
	ax = plt.subplot(6,2,1)
	Xlabel = C1name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = C1min
        Xmax = C1max
        plot2Chistograms(sC1, C1, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=12.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title=title)
	# plot ROC curves
	ax = plt.subplot(6,2,2)
  	plotROC(sC1, C1, Climit=Climit, title='')

	## Csebok classification 
	# plot histograms
	ax = plt.subplot(6,2,3)
	Xlabel = C2name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = C2min
        Xmax = C2max
        plot2Chistograms(sC2, C2, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=51.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(6,2,4)
  	plotROC(sC2, C2, Climit=Climit, title='')

	## chi2 classification 
	# plot histograms
	ax = plt.subplot(6,2,5)
	Xlabel = C3name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = C3min
        Xmax = C3max
        plot2Chistograms(sC3, C3, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=0.5, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(6,2,6)
  	plotROC(sC3, C3, Climit=Climit, title='')

	## Bayes classification 
	# plot histograms
	ax = plt.subplot(6,2,7)
	Xlabel = C4name
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = C4min
        Xmax = C4max
        plot2Chistograms(sC4, C4, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=0.5, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(6,2,8)
  	plotROC(sC4, C4, Climit=Climit, title='')

	## spread_model classification 
	# plot histograms
	ax = plt.subplot(6,2,9)
	Xlabel = C5name
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = C5min
        Xmax = C5max
        plot2Chistograms(sC5, C5, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=50.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(6,2,10)
  	plotROC(sC5, C5, Climit=Climit, title='')

	## best sigma classification 
	# plot histograms
	ax = plt.subplot(6,2,11)
	Xlabel = C6name 
        Ylabel = '$n / (N\Delta_{bin})$'
        Xmin = C6min
        Xmax = C6max
        plot2Chistograms(sC6, C6, Xmin=Xmin, Xmax=Xmax, Ymin=0.0, Ymax=5.0, Xlabel=Xlabel, Ylabel=Ylabel, bins=bins, title='')
	# plot ROC curves
	ax = plt.subplot(6,2,12)
  	plotROC(sC6, C6, Climit=Climit, title='')


	if (name is None):
		plt.show() 
	else:
		print 'saving plot to:', name
		plt.savefig(name, bbox_inches='tight')




  

def plotSNRplot1(SNR, CE1, CE2, CE3, Xmin=0, Xmax=100, Ymin=40, Ymax=110, name = None, title=''): 

    if (1):
        # 2, 4, 3
	C1name = '$C_{SDSS}=m_{psf}-m_{mod}$'
	C2name = '$C_{Bayes}$'
	C3name = '$\Delta \chi^2 = \chi^2_{psf}-\chi^2_{mod}$'
    else:
	C1name = ' '
	C2name = ' '
	C3name = ' '

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.90, wspace=0.18, hspace=0.46)

    ## 
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=26)
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    ax.set_xlabel(r'SNR', fontsize=26)
    ax.set_ylabel(r'C=E (\%)', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=25)

    ax.plot(SNR, CE1, '-k', label=C1name, lw=4)
    ax.plot(SNR, CE2, '--b', label=C2name, lw=4)
    ax.plot(SNR, CE3, '--r', label=C3name, lw=4)
    
    for CL in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        ax.plot([CL, CL], [0.0, 110], '-k', lw=1, alpha=0.4)
        ax.plot([0, 100], [CL, CL], '-k', lw=1, alpha=0.4)
        ax.plot([CL-5, CL-5], [0.0, 110], '--k', lw=1, alpha=0.4)
        ax.plot([0, 100], [CL-5, CL-5], '--k', lw=1, alpha=0.4)


    if (name is None):
       	plt.show() 
    else:
       	print 'saving plot to:', name
       	plt.savefig(name, bbox_inches='tight')


def plotSNRplot2(SNR1, CE1, SNR2, CE2, SNR3, CE3, Xmin=0, Xmax=100, Ymin=40, Ymax=110, name = None, title=''): 

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.90, wspace=0.18, hspace=0.46)

    ## 
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=26)
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    ax.set_xlabel(r'SNR', fontsize=26)
    ax.set_ylabel(r'C=E (\%)', fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=25)

    ax.plot(SNR1, CE1, '-k', lw=4)
    ax.plot(SNR2, CE2, '--b', lw=4)
    ax.plot(SNR3, CE3, '--r', lw=4)
    
    for CL in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        ax.plot([CL, CL], [0.0, 110], '-k', lw=1, alpha=0.4)
        ax.plot([0, 100], [CL, CL], '-k', lw=1, alpha=0.4)
        ax.plot([CL-5, CL-5], [0.0, 110], '--k', lw=1, alpha=0.4)
        ax.plot([0, 100], [CL-5, CL-5], '--k', lw=1, alpha=0.4)


    if (name is None):
       	plt.show() 
    else:
       	print 'saving plot to:', name
       	plt.savefig(name, bbox_inches='tight')


