from astrometry import *

def demo(muTx, sigmaNoise, alpha):

        ## defaults everywhere
        muTy = 0
	Atrue = 1000.0 
        Btrue = 0.0
        # make 1D pixel array
        oneDpixels = np.linspace(-11, 11, 23)
        Npixels = oneDpixels.size**2
        # noiseless model image
        nonoise, source = gauss2Dastrom(muTx, muTy, alpha, Atrue, Btrue, oneDpixels[:, np.newaxis], oneDpixels)
        # now add noise
        addsourcenoise = 1
        image, variance = addnoise(nonoise, sigmaNoise, source, addsourcenoise) 
   
        # and now find ML best-fit 
        thisVariance = 0*variance + 1*sigmaNoise**2 
        chiPixXcen, chiPixCmod, chi2, bestM, Xcen, Cmod, chi2min = \
              getChi2imageAstro(oneDpixels, image, muTy, Btrue, alpha, thisVariance)

        # analytic best-fit counts using best-fit profile 
        modelNorm = (bestM-Btrue) / np.sum(bestM-Btrue)
        wModel = modelNorm / np.sum(modelNorm**2)
        CbestFit = np.sum((image-Btrue)*wModel)
        # chi2modBest = np.sum((image-Btrue-CbestFit*modelNorm)**2/variance)
        chi2modBest = np.sum((image-Btrue-CbestFit*modelNorm)**2/thisVariance)
 
        print 'chi2min_pdf=', chi2min/Npixels
        print 'Xcen =', Xcen 
        print 'Cmod (ML) =', Cmod
        print 'CbestFit (approx) =', CbestFit
        print 'chi2modBest_pdf=', chi2modBest/Npixels

        return oneDpixels, image, chiPixXcen, chiPixCmod, chi2, bestM, Xcen, Cmod, chi2min


def doMany(alpha, sigmaNoise, Ntrials): 

    XcenArr = np.linspace(0,0.1,Ntrials) 
    XcenExpArr = 0*XcenArr
    XcenStdArr = 0*XcenArr
    CmodArr = 0*XcenArr
    CmodExpArr = 0*XcenArr
    CmodStdArr = 0*XcenArr
    CastroArr = 0*XcenArr
    SnrArr = 0*XcenArr
    chi2minArr = 0*XcenArr

    muTx = 0.0 
    for i in range(0, XcenArr.size):
       print '-------------------------------' 
       print 'working on #', i
       oneDpix, image, chiPixXcen, chiPixCmod, chi2, bestM, Xcen, Cmod, chi2min = demo(muTx, sigmaNoise, alpha)
       CmodArr[i] = Cmod
       XcenArr[i] = Xcen
       chi2minArr[i] = chi2min
       if (0):
           p_Xcen, p_Cmod = chi2plotMarginalAstro(chiPixXcen, chiPixCmod, chi2, muTx, Xcen, Cmod)
       else:
           # lnL from chi2  (n.b. no cutoff of low values as in plotting code above!) 
           lnL1 = -0.5*chi2
           lnL1 -= lnL1.max()
           p_Xcen = np.exp(lnL1).sum(0)
           p_Cmod = np.exp(lnL1).sum(1)

       XcenExp = np.trapz(chiPixXcen*p_Xcen, chiPixXcen)/np.trapz(p_Xcen, chiPixXcen)
       XcenStd = np.sqrt(np.trapz((chiPixXcen-XcenExp)**2*p_Xcen, chiPixXcen)/np.trapz(p_Xcen, chiPixXcen))
       CmodExp = np.trapz(chiPixCmod*p_Cmod, chiPixCmod)/np.trapz(p_Cmod, chiPixCmod)
       CmodStd = np.sqrt(np.trapz((chiPixCmod-CmodExp)**2*p_Cmod, chiPixCmod)/np.trapz(p_Cmod, chiPixCmod))
       SNR = np.abs(CmodExp/CmodStd)
       Castro = XcenStd * SNR / (alpha*2.355) 
       XcenExpArr[i] = XcenExp
       XcenStdArr[i] = XcenStd 
       CmodExpArr[i] = CmodExp
       CmodStdArr[i] = CmodStd
       SnrArr[i] = SNR
       CastroArr[i] = Castro
       print 'again Cmod=', Cmod
       print 'CmodExp =', CmodExp

    return CmodArr, CmodExpArr, CmodStdArr, XcenArr, XcenExpArr, XcenStdArr, CastroArr, SnrArr, chi2minArr


# tests (check first existing output, corresponding to case 1) 
# Saturday: change Bkgd=0 and 23 pixel square (and added chi2min to outputs) 

#Ntrial = 1000
#alpha = 1.0 
#sigmaNoise = 10.0
#CmodArr1, CmodExpArr1, CmodStdArr1, XcenArr1, XcenExpArr1, XcenStdArr1, CastroArr1, SnrArr1, ch1 = doMany(alpha, sigmaNoise, Ntrial) 
#alpha = 2.0 
#sigmaNoise = 10.0
#CmodArr2, CmodExpArr2, CmodStdArr2, XcenArr2, XcenExpArr2, XcenStdArr2, CastroArr2, SnrArr2, ch2 = doMany(alpha, sigmaNoise, Ntrial) 
#alpha = 1.0 
#sigmaNoise = 5.0
#CmodArr3, CmodExpArr3, CmodStdArr3, XcenArr3, XcenExpArr3, XcenStdArr3, CastroArr3, SnrArr3, ch3 = doMany(alpha, sigmaNoise, Ntrial) 


# left panel, testing single Gauss, no source noise 
#Ntrial = 1000
#alpha = 1.0 
#sigmaNoise = 5.0
#CmodArr4, CmodExpArr4, CmodStdArr4, XcenArr4, XcenExpArr4, XcenStdArr4, CastroArr4, SnrArr4, ch4 = doMany(alpha, sigmaNoise, Ntrial) 


# middle panel, testing single Gauss, with source noise   ** approx ML off ~2% (homoscedastic ML) 
#Ntrial = 1000
#alpha = 1.0 
#sigmaNoise = 5.0
#CmodArr5, CmodExpArr5, CmodStdArr5, XcenArr5, XcenExpArr5, XcenStdArr5, CastroArr5, SnrArr5, ch5 = doMany(alpha, sigmaNoise, Ntrial) 
### Much lower SNR here, visible in the actual scatter and computed sigma for Cmod and Xcen ### 

# right panel, double Gauss, no source noise 
def demo1():
    Ntrial = 1000
    alpha = 1.0 
    sigmaNoise = 5.0
    CmodArr6, CmodExpArr6, CmodStdArr6, XcenArr6, XcenExpArr6, XcenStdArr6, CastroArr6, SnrArr6, ch6 = doMany(alpha, sigmaNoise, Ntrial) 

  
# right-most panel, like case 5, but ignoring the fact that source noise was added 
def demo2():
    Ntrial = 1000
    alpha = 1.0 
    sigmaNoise = 5.0
    CmodArr7, CmodExpArr7, CmodStdArr7, XcenArr7, XcenExpArr7, XcenStdArr7, CastroArr7, SnrArr7, ch7 = doMany(alpha, sigmaNoise, Ntrial) 
    return CmodArr7, CmodExpArr7, CmodStdArr7, XcenArr7, XcenExpArr7, XcenStdArr7, CastroArr7, SnrArr7, ch7


print 'calling demo2...'
CmodArr, CmodExpArr, CmodStdArr, XcenArr, XcenExpArr, XcenStdArr, CastroArr, SnrArr, chi2minArr = demo2()




### QUESTION: is the model count scatter ~39 (instead of 18)? and Xcen 0.044 (instead of 0.025) 
###              (because more noise was added into the image but not accounted for) 




# SNR as np.median(CmodArr1)/np.std(CmodArr1)
#             27.9, 14.5, 56.7  
# from array: 22.8, 14.0, 18.4 

#  *** if Castro computed directly as e.g. np.std(XcenArr2)*np.median(CmodArr2)/np.std(CmodArr2)/2.355/2
#  then Castro = 0.60, 0.54, 0.68, or about 7% variation from 0.60. But this is consistent with N=100
# therefore repeating with N=1000 above which should bring it down to about 2% variation (<0.02 around 0.600) 
#    with Ntrial=1000, Castro=0.60, 0.61, 0.62 -> OK 

# Castro medians: 0.70, 0.60, 0.83 ??  why varying? shouldn't always be same?
#     std Xcen    mean XcenErr
# 1    0.052          0.072 
# 2    0.225          0.205 
# 3    0.026          0.105    -> how can error prediction be 4 times as large ?? 

# std.dev. for Xcen:  0.052, 0.225, 0.026
# from 1 to 2: because of alpha twice as large, SNR is factor of two smaller so change in astrom err is factor 4 
# from 1 to 3: alpha same but SNR higher by a factor of 2 

# Cmod and CmodExp behave similarly
# CmodExpArr2 has a median of 991 (bcs alpha=2?) 
# CmodExpArr2 has twise as large scatter as CmodExp1, CmodExpArr3 more than 2 smaller, approx OK 

# CmodStdArr3 has a median of 54 while CmodExpArr3 has st.dev of 16: chi2 << 1 
#   CmodStd problems are related to problems with SNR: 
# SNR medians: 22.9, 13.9, 18.5 something's wrong! only 18.5/22.9 reduction for sigmaNoise droping by 2 
#   SNR calculation is totally bad: it should go down and up by a factor of 2 from case 1
# somehow CmodStd calculation is bad 




