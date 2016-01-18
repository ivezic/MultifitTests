
# %run tools2Dgauss.py

# %run plot2Dgauss.py

# %run figPlots.py

### wrappers for various tests
# %run tools2Dgauss.py

from tools2Dgauss import *
from figPlots import *

def test1():

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaTrue = [1.0, 1.0, 1.0, 0.5, 1.5]
	sigmaNoise = [15.0, 10.0, 20.0, 15.0, 15.0]

	for i in range(0,len(sigmaNoise)):
		sigNoise = sigmaNoise[i]
		sigTrue = sigmaTrue[i]
      
		# direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0)
		eta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig)
		print 'median chi2mod:', np.median(chi2Model)

		# save 
		foutname = 'SGbrute_test1_' + str(i) + '.dat'
		vectors = [sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)

		print 'computed as saved data for case', i

		if (1):
			# make standard plots
			Fname = 'SGbrute_test1_Comp_' + str(i)
			title = 'SGbrute test1 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test1_Class_' + str(i)
			title = 'SGbrute test1 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test1() from plot2Dgauss.py'





def test1makePlots():

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaTrue = [1.0, 1.0, 1.0, 0.5, 1.5]
	sigmaNoise = [15.0, 10.0, 20.0, 15.0, 15.0]

	for i in range(0,len(sigmaNoise)):
      
		foutname = 'SGbrute_test1_' + str(i) + '.dat'
		if (1):
			# make standard plots
			Fname = 'SGbrute_test1_Comp_' + str(i)
			title = 'SGbrute test1 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test1_Class_' + str(i)
			title = 'SGbrute test1 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test1() from plot2Dgauss.py'





def test2():

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaTrue = np.linspace(0.5, 1.5, 101)
	sigmaNoise = 15.0 + 0*sigmaTrue

	for i in range(0,len(sigmaNoise)):
		sigNoise = sigmaNoise[i]
		sigTrue = sigmaTrue[i]
      
		# direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0)
		eta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig)
		print 'median chi2mod:', np.median(chi2Model)

		# save 
		foutname = 'SGbrute_test2_' + str(i) + '.dat'
		vectors = [sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)

		print 'computed as saved data for case', i

		if (1):
			# make standard plots
			Fname = 'SGbrute_test2_Comp_' + str(i)
			title = 'SGbrute test2 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test2_Class_' + str(i)
			title = 'SGbrute test2 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test2() from plot2Dgauss.py'




def test3():

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaNoise = np.linspace(1, 20, 101)
	sigmaTrue = 1.0 + 0*sigmaNoise

	for i in range(0,len(sigmaNoise)):
		sigNoise = sigmaNoise[i]
		sigTrue = sigmaTrue[i]
      
		# direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0)
		eta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig)
		print 'median chi2mod:', np.median(chi2Model)

		# save 
		foutname = 'SGbrute_test3_' + str(i) + '.dat'
		vectors = [sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)

		print 'computed as saved data for case', i

		if (1):
			# make standard plots
			Fname = 'SGbrute_test3_Comp_' + str(i)
			title = 'SGbrute test3 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test3_Class_' + str(i)
			title = 'SGbrute test3 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test3() from plot2Dgauss.py'





def test4():

	### same as test1, except that here ** 2-parameter ** MCMC is used: it should be comparable to test1 results
	### ALSO, instead several combinations of sigma_m and theta_g, sigmaTrue is set to 1.0 (and 3 noises)
	### AND: 1,000 trials instead of 10,000
        Ntrial = 10000
	sigmaTrue = [1.0, 1.0, 1.0]
	sigmaNoise = [15.0, 10.0, 20.0]

	for i in range(2,len(sigmaNoise)):
		sigNoise = sigmaNoise[i]
		sigTrue = sigmaTrue[i]
      
		# MCMC minimization - 2 parameters 
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=1)
		eta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=1)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig)
		print 'median chi2mod:', np.median(chi2Model)

		# save 
		foutname = 'SGbrute_test4_' + str(i) + '.dat'
		vectors = [sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)

		print 'computed as saved data for case', i

		if (1):
			# make standard plots
			Fname = 'SGbrute_test4_Comp_' + str(i)
			title = 'SGbrute test4 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test4_Class_' + str(i)
			title = 'SGbrute test4 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test4() from plot2Dgauss.py'





def test5():

        ###  -- identical to test4,   except that *** 5-parameter *** fit is used (instead of 2 parameters) 
	### same as test1, except that here ** 2-parameter ** MCMC is used: it should be comparable to test1 results
	### ALSO, instead several combinations of sigma_m and theta_g, sigmaTrue is set to 1.0 (and 3 noises)
	### AND: 1,000 trials instead of 10,000
       # Ntrial = 10000
        Ntrial = 3000
	sigmaTrue = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	sigmaNoise = [15.0, 10.0, 20.0, 1.0, 2.0, 5.0]

	for i in range(3,len(sigmaNoise)):
		sigNoise = sigmaNoise[i]
		sigTrue = sigmaTrue[i]
      
		# MCMC minimization - 5 parameters 
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=1)
		eta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=1)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig)
		print 'median chi2mod:', np.median(chi2Model)

		# save 
		foutname = 'SGbrute_test5_' + str(i) + '.dat'
		vectors = [sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)

		print 'computed as saved data for case', i

		if (1):
			# make standard plots
			Fname = 'SGbrute_test5_Comp_' + str(i)
			title = 'SGbrute test5 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test5_Class_' + str(i)
			title = 'SGbrute test5 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test5() from plot2Dgauss.py'





def test6():

  ### THIS INCLUDES BAYES FACTORS TOO ###
    ##  C2 = CSebok is replaced by Bayes factor ##

     ## ** need more noise for sigTrue=1 case: 25, 30, 35, 40 ###

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaTrue = [1.0, 1.0, 1.0, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0]
	sigmaNoise = [15.0, 10.0, 20.0, 15.0, 15.0, 25.0, 30.0, 35.0, 40.0]
        # Ntrial = 3000
	# sigmaTrue = [1.0]
	# sigmaNoise = [15.0]

      ### NEED TO COMPARE Cmod AND bestA !!! from existing files - just plot! ###      

	for i in range(5,len(sigmaNoise)):
		sigNoise = sigmaNoise[i]
		sigTrue = sigmaTrue[i]
      
		# direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig)
		print 'median chi2mod:', np.median(chi2Model)

		# save 
		foutname = 'SGbrute_test6_' + str(i) + '.dat'
		vectors = [sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)

		print 'computed as saved data for case', i

		if (1):
			# make standard plots
			Fname = 'SGbrute_test6_Comp_' + str(i)
			title = 'SGbrute test6 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test6_Class_' + str(i)
			title = 'SGbrute test6 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test6() from plot2Dgauss.py'





def test6makePlots():

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
	sigmaNoise = [15.0]

	for i in range(0, 9):
		foutname = 'SGbrute_test6_' + str(i) + '.dat'
		if (1):
			# make standard plots
			Fname = 'SGbrute_test6_Comp_' + str(i)
			title = 'SGbrute test6 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test6_Class_' + str(i)
			title = 'SGbrute test6 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test6() from plot2Dgauss.py'




def test5makePlots():


	for i in range(0, 6):
		foutname = 'SGbrute_test5_' + str(i) + '.dat'
		if (1):
			# make standard plots
			Fname = 'SGbrute_test5_Comp_' + str(i)
			title = 'SGbrute test5 Comp ' + str(i)
			figCcomparison(foutname, Fname, title)
			Fname = 'SGbrute_test5_Class_' + str(i)
			title = 'SGbrute test5 Class ' + str(i)
			figClassification(foutname, Fname, title)
			print 'made standard plots for case', i

		print 'completed case', i

	print 'DONE with test5() from plot2Dgauss.py'






##########################################################################
### joint tests of Bayes Factor and spread_model 
def test7(do1=1, do2=1, do3=1, makePlots=0):

        #### need to submit do1=1 with sigNoise=1, 40, step of 1
	####  also sigmaNoise =15, and sigTrue = 0.1, 2.0 step of 0.1
 
	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaTrue = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	sigmaNoise = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        Ntrial = 3000
	#sigmaTrue = [1.0]
	#sigmaNoise = [15.0]

        ### NEED TO COMPARE Cmod AND bestA !!! from existing files - just plot! ###      

	for i in range(3,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do1): 	
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test7_1_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test7_Comp1_' + str(i)
			title = 'SGall test7 Comp1 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test7_Class1_' + str(i)
			title = 'SGall test7 Class1 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 1', i


	    if (do2): 	
		### 2) MCMC chi2 minimization for 2 parameters
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=1, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=1, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test7_2_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test7_Comp2_' + str(i)
			title = 'SGall test7 Comp2 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test7_Class2_' + str(i)
			title = 'SGall test7 Class2 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 2', i

	    if (do3): 	
		### 2) MCMC chi2 minimization for 5 parameters
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=1, p5=3)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=1, p5=3)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test7_3_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test7_Comp3_' + str(i)
			title = 'SGall test7 Comp3 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test7_Class3_' + str(i)
			title = 'SGall test7 Class3 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 2', i


		print '================= completed case', i, ' ===================='

	print 'DONE with test7() from plot2Dgauss.py'



def test7makePlots(do1=1, do2=1, do3=1, makePlots=0):

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaTrue = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	sigmaNoise = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        Ntrial = 3000
	sigmaTrue = [1.0]
	sigmaNoise = [15.0]

  
	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do1): 	
		foutname = 'SGall_test7_1_' + str(i) + '.dat'
		if (makePlots):
			# make standard plots
			Fname = 'SGall_test7_Comp1_' + str(i)
			title = 'SGall test7 Comp1 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test7_Class1_' + str(i)
			title = 'SGall test7 Class1 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 1', i


	    if (do2): 	
		foutname = 'SGall_test7_2_' + str(i) + '.dat'
		if (makePlots):
			# make standard plots
			Fname = 'SGall_test7_Comp2_' + str(i)
			title = 'SGall test7 Comp2 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test7_Class2_' + str(i)
			title = 'SGall test7 Class2 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 2', i

	    if (do3): 	
		foutname = 'SGall_test7_3_' + str(i) + '.dat'
		if (makePlots):
			# make standard plots
			Fname = 'SGall_test7_Comp3_' + str(i)
			title = 'SGall test7 Comp3 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test7_Class3_' + str(i)
			title = 'SGall test7 Class3 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 2', i


		print '================= completed case', i, ' ===================='

	print 'DONE with test7() from plot2Dgauss.py'




##########################################################################
### joint tests of Bayes Factor and spread_model 
def test8(do1=1, do2=1, do3=1, makePlots=1):

	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000
	sigmaTrue = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	sigmaNoise = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        Ntrial = 1000
	sigmaTrue = [1.0]
	sigmaNoise = [5.0]

	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do1): 	
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test8_1_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test8_Comp1_' + str(i)
			title = 'SGall test8 Comp1 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test8_Class1_' + str(i)
			title = 'SGall test8 Class1 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 1', i


	    if (do2): 	
		### 2) MCMC chi2 minimization for 2 parameters
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=1, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=1, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test8_2_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test8_Comp2_' + str(i)
			title = 'SGall test8 Comp2 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test8_Class2_' + str(i)
			title = 'SGall test8 Class2 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 2', i

	    if (do3): 	
		### 2) MCMC chi2 minimization for 5 parameters
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=1, p5=3)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=1, p5=3)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test8_3_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test8_Comp3_' + str(i)
			title = 'SGall test8 Comp3 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test8_Class3_' + str(i)
			title = 'SGall test8 Class3 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 3', i


		print '================= completed case', i, ' ===================='

	print 'DONE with test8() from plot2Dgauss.py'






##########################################################################
### joint tests of Bayes Factor and spread_model using direct method
def test9(do1=1, do2=1, do3=1, makePlots=0):
 
	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000

    ### after Cspread understood, submit jobs like below
    ### with sigmaTrue = 0.5 and 1.5 (and revisit noise range) 

        #### constant sigmaTrue, varying noise
	sigmaNoise = np.linspace(1.0, 40.0, 40)
	sigmaTrue = 1.0 + 0*sigmaNoise

	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do1): 	
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test9_1_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test9_Comp1_' + str(i)
			title = 'SGall test9 Comp1 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test9_Class1_' + str(i)
			title = 'SGall test9 Class1 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 1', i

		print '================= completed case 1', i, ' ===================='



        #### constant noise, varying sigmaTrue
	sigmaTrue = np.linspace(0.0, 2.0, 21)
	sigmaNoise = 15.0 + 0*sigmaTrue

	for i in range(14,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do2): 	
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test9_2_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 2, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test9_Comp2_' + str(i)
			title = 'SGall test9 Comp2 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test9_Class2_' + str(i)
			title = 'SGall test9 Class2 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 2', i

		print '================= completed case 2', i, ' ===================='


       #### this is sigmaTrue = 1.0, and for Cspread using sigmaTrue CODE CHANGE!!! 
	sigmaTrue = np.linspace(0.5, 1.5, 11)
	sigmaNoise = 15.0 + 0*sigmaTrue

	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do3==1): 	
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		# note that 0 was replaced by sigTrue
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, -1.0*sigTrue, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test9_3_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 3, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test9_Comp3_' + str(i)
			title = 'SGall test9 Comp3 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test9_Class3_' + str(i)
			title = 'SGall test9 Class3 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 3', i

		print '================= completed case 3', i, ' ===================='


       #### this is sigmaTrue = 1.0, and for Cspread using sigmaTrue CODE CHANGE!!! 
          #### ****** this tests that classification breaks down for sigmaTrue = 0 ******
	sigmaTrue = np.linspace(0.0, 0.5, 6)
	sigmaNoise = 15.0 + 0*sigmaTrue

	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do3==2): 	
                print 'entering with', do3, sigTrue, sigNoise
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		# note that 0 was replaced by sigTrue
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, -1.0*sigTrue, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test9_4_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 4, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test9_Comp4_' + str(i)
			title = 'SGall test9 Comp4 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test9_Class4_' + str(i)
			title = 'SGall test9 Class4 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 4', i

		print '================= completed case 4', i, ' ===================='

      #### this is sigmaTrue = 1.0, and for Cspread using sigmaTrue CODE CHANGE!!! 
          #### ****** this tests that classification breaks down for sigmaTrue = 0 ******
	sigmaTrue = np.linspace(0.01, 0.1, 10)
	sigmaNoise = 15.0 + 0*sigmaTrue

	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do3==3): 	
                print 'entering with', do3, sigTrue, sigNoise
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		# note that 0 was replaced by sigTrue
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, -1.0*sigTrue, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test9_5_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 5, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test9_Comp5_' + str(i)
			title = 'SGall test9 Comp5 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test9_Class5_' + str(i)
			title = 'SGall test9 Class5 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 5', i

		print '================= completed case 5', i, ' ===================='





	print 'DONE with test9() from plot2Dgauss.py'




##########################################################################
### SAME as test9 case1, except for different sigmaTrue=0.3, 0.5, 1.5
def test10(do1=1, do2=1, do3=1, makePlots=0):
 
	### brute-force chi2 minimization over sigma and Cmodel
	### for several combinations of sigma_m and theta_g
        Ntrial = 10000

        #### constant sigmaTrue, varying noise
	sigmaNoise = np.linspace(1.0, 36.0, 36)

	sigmaTrue = 0.3 + 0*sigmaNoise
	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do1): 	
		### 1) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test10_1_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 1, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test10_Comp1_' + str(i)
			title = 'SGall test10 Comp1 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test10_Class1_' + str(i)
			title = 'SGall test10 Class1 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 1', i

		print '================= completed case 1', i, ' ===================='



	sigmaTrue = 0.5 + 0*sigmaNoise
	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do2): 	
		### 2) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test10_2_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 2, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test10_Comp2_' + str(i)
			title = 'SGall test10 Comp2 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test10_Class2_' + str(i)
			title = 'SGall test10 Class2 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 2', i

		print '================= completed case 2', i, ' ===================='


	sigmaTrue = 1.5 + 0*sigmaNoise
	for i in range(0,len(sigmaNoise)):
            sigNoise = sigmaNoise[i]
	    sigTrue = sigmaTrue[i]

	    if (do3): 	
		### 2) direct chi2 minimization
		eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr = \
		    ThreeClassifiersDistributions(Ntrial, sigTrue, sigNoise, fitMCMC=0, p5=0)
		seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr = \
		    ThreeClassifiersDistributions(Ntrial, 0.0, sigNoise, fitMCMC=0, p5=0)

		print 'median best Sig:', np.median(bestSig), ' rms:', np.std(bestSig), '<sigErr>:', np.median(bestSigErr)
		print 'median chi2mod:', np.median(chi2Model)
		print 'median seta:', np.median(seta)
		print 'median eta:', np.median(eta)

		# save 
		foutname = 'SGall_test10_3_' + str(i) + '.dat'
		vectors = [seta, sC1, sC2, sC3, SneffPSF, SneffModel, Schi2PSF, Schi2Model, sCmod, SbestA, SbestARMS, SbSig, SbSigErr, \
                               eta, C1, C2, C3, neffPSF, neffModel, chi2PSF, chi2Model, Cmod, bestA, bestARMS, bestSig, bestSigErr]
		np.savetxt(foutname, vectors)
		print 'computed and saved data for case 3, iteration:', i

		if (makePlots):
			# make standard plots
			Fname = 'SGall_test10_Comp3_' + str(i)
			title = 'SGall test10 Comp3 ' + str(i)
			figCcomparison2(foutname, Fname, title)
			Fname = 'SGall_test10_Class3_' + str(i)
			title = 'SGall test10 Class3 ' + str(i)
			figClassification2(foutname, Fname, title)
			print 'made standard plots for case 3', i

		print '================= completed case 3', i, ' ===================='



	print 'DONE with test10() from plot2Dgauss.py'


