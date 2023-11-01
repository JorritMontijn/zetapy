import scipy.io
from zetapy import getZeta
import os
import numpy as np
import matplotlib.pyplot

"""Run example ZETA-test

This code loads data from an example LP cell and performs a ZETA-test,
makes a raster plot and calculates the instantaneous firing rate

Version history:
1.0 - 17 June 2020
1.1 - 23 Feb 2021 Refactoring of python code by Guido Meijer
Created by Jorrit Montijn, translated to python by Alexander Heimel
"""

# set random seed
np.random.seed(1)

# load data for example cell
strDataFile = os.path.join(os.path.dirname(__file__), 'ExampleDataZETA.mat')
dLoad = scipy.io.loadmat(strDataFile)

# some information about the neuron is stored in the dNeuron structure,
# such as whether Kilosort2 thought it was an acceptable neuron
dNeuron = dLoad['sNeuron']
if dNeuron['KilosortGood'] == 0 or dNeuron['NonStationarity'] > 0.5:
    raise(Exception('BadUnit: This unit is non-stationary, noise-like, or contaminated'))

# retrieve the spike times as an array from the field in dNeuron
arrSpikeTimes = dNeuron['SpikeTimes'][0][0]

# load stimulation information
sStim = dLoad['sStim']
arrStimulusStartTimes = sStim['StimOnTime'][0][0][0]  # unpacking Matlab array
arrStimulusStopTimes = sStim['StimOffTime'][0][0][0]  # unpacking Matlab array

# put stimulus start and stop times together into a [T x 2] matrix
arrEventTimes = np.transpose(np.array([arrStimulusStartTimes,arrStimulusStopTimes]))

# run the ZETA-test with default parameters
dblZetaP, vecLatencies = getZeta(arrSpikeTimes, arrEventTimes)

print(f'\nDefault parameters\np-value: {dblZetaP}\nlatencies: {vecLatencies}')

## run the ZETA-test with specified parameters

# median of trial-to-trial durations
dblUseMaxDur = np.median(np.diff(arrStimulusStartTimes))

# 50 random resamplings should give us a good enough idea if this cell is responsive.
# If it's close to 0.05, we should increase this number.
intResampNum = 50

# what do we want to plot?(0=nothing, 1=inst. rate only, 2=traces only, 3=raster plot as well,
# 4=adds latencies in raster plot)
intPlot = 1

# how many latencies do we want? 1=ZETA, 2=-ZETA, 3=peak, 4=first crossing of peak half-height
intLatencyPeaks = 4

# do we want to restrict the peak detection to for example the time during stimulus?
# Then put (0 1) here.
tplRestrictRange = (0, np.inf)

# then run ZETA with those parameters
dblZetaP, vecLatencies = getZeta(arrSpikeTimes, arrEventTimes,
                                 dblUseMaxDur=dblUseMaxDur,
                                 intResampNum=intResampNum,
                                 intPlot=intPlot,
                                 intLatencyPeaks=intLatencyPeaks,
                                 tplRestrictRange=tplRestrictRange)


print(f'\nSpecified parameters\np-value: {dblZetaP}\nlatencies: {vecLatencies}')

