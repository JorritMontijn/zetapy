import scipy.io
import zetapy
import os
import numpy as np
import matplotlib.pyplot 

def runExampleZETA():
	"""Run example ZETA-test
	
	This code loads data from an example LP cell and performs a ZETA-test,
	makes a raster plot and calculates the instantaneous firing rate
	
	Version history:
	1.0 - 17 June 2020
	Created by Jorrit Montijn, translated to python by Alexander Heimel
	"""

	np.random.seed(1) # to match Matlab output

	# load data for example cell
	### sLoad = load('ExampleDataZETA.mat');
	modulepath = os.path.dirname(__file__)
	sLoad = scipy.io.loadmat(os.path.join(modulepath,'ExampleDataZETA.mat'))
	
	# some information about the neuron is stored in the sNeuron structure,
	# such as whether Kilosort2 thought it was an acceptable neuron
	### sNeuron = sLoad.sNeuron;
	sNeuron = sLoad['sNeuron']
		
	### if sNeuron.KilosortGood == 0 || sNeuron.NonStationarity > 0.5
	### 	error([mfilename ':BadUnit'],'This unit is non-stationary, noise-like, or contaminated');
	### end
	
	if sNeuron['KilosortGood'] == 0 or sNeuron['NonStationarity'] > 0.5:
		raise(Exception('BadUnit: This unit is non-stationary, noise-like, or contaminated'))

	# retrieve the spike times as a vector from the field in sNeuron
	### vecSpikeTimes = sNeuron.SpikeTimes;
	vecSpikeTimes = sNeuron['SpikeTimes'][0][0]

	## load stimulation information
	### sStim = sLoad.sStim;
	sStim = sLoad['sStim']

	### vecStimulusStartTimes = sStim.StimOnTime(:); %use (:) to ensure it's a column vector
	### vecStimulusStopTimes = sStim.StimOffTime(:);
	vecStimulusStartTimes = sStim['StimOnTime'][0][0][0]  # unpacking Matlab array
	vecStimulusStopTimes = sStim['StimOffTime'][0][0][0]  # unpacking Matlab array

	## put stimulus start and stop times together into a [T x 2] matrix
	# matEventTimes = cat(2,vecStimulusStartTimes,vecStimulusStopTimes);
	matEventTimes = np.transpose(np.array([vecStimulusStartTimes,vecStimulusStopTimes]))
	
	## run the ZETA-test with default parameters
	# if we simply want to know if the neuron responds, no hassle, we can
	# use this simple syntax with default parameters:
	(dblZetaP, vecLatencies, sZETA, sRate) = zetapy.getZeta(vecSpikeTimes,matEventTimes)


	print('Default parameters')
	print('Matlab: ZetaP from default parameters: 1.4974e-05')
	print('Python: ZetaP from default parameters: ', dblZetaP)
	print('Matlab: vecLatencies = [0.0710 0.2730]')
	print('Python: vecLatencies = ', vecLatencies)

	# (vecMSD,sMSD) = zetapy.getIFR(vecSpikeTimes,matEventTimes)


	## run the ZETA-test with specified parameters
	# however, we can also specify the parameters ourselves
	                                 
	dblUseMaxDur = np.median(np.diff(vecStimulusStartTimes)) # median of trial-to-trial durations
	intResampNum = 50 #50 random resamplings should give us a good enough idea if this cell is responsive. If it's close to 0.05, we should increase this #.
	intPlot = 3 # what do we want to plot?(0=nothing, 1=inst. rate only, 2=traces only, 3=raster plot as well, 4=adds latencies in raster plot)
	intLatencyPeaks = 4 # how many latencies do we want? 1=ZETA, 2=-ZETA, 3=peak, 4=first crossing of peak half-height
	vecRestrictRange = (0,float('inf')) #do we want to restrict the peak detection to for example the time during stimulus? Then put [0 1] here.
	boolVerbose = True # displays the progress if it takes >5 seconds
	
	# then run ZETA with those parameters
	(dblZetaP, vecLatencies, sZETA, sRate) = zetapy.getZeta(vecSpikeTimes,matEventTimes,dblUseMaxDur,intResampNum,intPlot,intLatencyPeaks,vecRestrictRange,boolVerbose)
	
	print('Specified parameters')
	print("Matlab: ZetaP from specified parameters: 2.0095e-06")
	print('ZetaP from specified parameters: ', dblZetaP)
	print('Matlab: vecLatencies = [0.0710 0.2730 0.0548 0.0507]')
	print('Python: vecLatencies = ', vecLatencies)

	return dblZetaP, vecLatencies, sZETA, sRate

runExampleZETA()