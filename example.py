"""Run example ZETA-test.

This code loads data from an example cell and performs some analyses as a tutorial/

Please note that the MATLAB version is much faster than zetapy due to the amount of
vectorized operations in the zeta-test that are computed very efficiently in MATLAB,
but less so in numpy. If anyone has any suggestions to improve compution times in python,
please contact me by e-mail or on the github repository: https://github.com/JorritMontijn/zetapy

Don't hesitate to leave any questions, bug reports or comments there either!

Version history:
1.0 - 17 June 2020 Created by Jorrit Montijn, translated to python by Alexander Heimel
1.1 - 23 Feb 2021 Refactoring of python code by Guido Meijer
2.0 - 29 Aug 2023 New translation to python by Jorrit Montijn

"""

import scipy.io
from zetapy import ifr, zetatest, zetatstest
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import convolve
import time

# %% note on performance
'''
The code uses vectorization as much as possible, but since the ZETA-test is based on bootstraps, it
might still take a couple of seconds to compute. Parallel processing in python is not yet implemented,
so if an expert in parallel computing in python reads this who is willing to contribute: please contact me!

In the examples below, I recorded the following processing times on my PC with default parameters:

MATLAB with parallel processing:
    zeta-test: 0.56 s
    time-series zeta-test: 2.77 s

MATLAB without parallel processing:
    zeta-test: 1.09 s
    time-series zeta-test: 19.88 s

Python without parallel processing:
    zeta-test: 1.41 s
    time-series zeta-test: 32.24 s
'''

# %% load and prepare some example data
# load data for example cell
strDataFile = os.path.join(os.path.dirname(__file__), 'ExampleDataZetaTest.mat')
dLoad = scipy.io.loadmat(strDataFile)

# retrieve the spike times as an array from the field in dNeuron
vecSpikeTimes1 = dLoad['sNeuron']['SpikeTimes'][0][0]
vecSpikeTimes2 = dLoad['sNeuron']['SpikeTimes'][0][1]

# load stimulation information
sStim = dLoad['sStim']
vecStimulusStartTimes = sStim['StimOnTime'][0][0][0]  # unpacking Matlab array
vecStimulusStopTimes = sStim['StimOffTime'][0][0][0]  # unpacking Matlab array

# %% calculate instantaneous firing rate without performing the ZETA-test
# if we simply want to plot the neuron's response, we can use:
vecTime, vecRate, dIFR = ifr(vecSpikeTimes1, vecStimulusStartTimes)

# plot results
f, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(vecTime, vecRate)
ax.set(xlabel='Time after event (s)', ylabel='Instantaneous spiking rate (Hz)')
ax.set(title="A simple plot of the neuron's rate using ifr()")

# %% run the ZETA-test with default parameters
# set random seed
np.random.seed(1)

# run test
t = time.time()
dblZetaP = zetatest(vecSpikeTimes1, vecStimulusStartTimes)[0]  # use [0] to return only the p-value
dblElapsedT = time.time() - t

print(f'\nDefault parameters (elapsed time: {dblElapsedT:.2f} s):\np-value: {dblZetaP}')

# %% run the ZETA-test with specified parameters
# set random seed
np.random.seed(1)

# use minimum of trial-to-trial durations as analysis window size
dblUseMaxDur = np.min(np.diff(vecStimulusStartTimes))

# 50 random resamplings should give us a good enough idea if this cell is responsive.
# If the p-value is close to 0.05, we should increase this number.
intResampNum = 50

# what size of jittering do we want? (multiple of dblUseMaxDur; default is 2.0)
dblJitterSize = 2.0

# Do we want to plot the results (0=no, 1=yes)
intPlot = 1

# How many latencies do we want to compute? 1=ZETA, 2=-ZETA, 3=peak, 4=first crossing of peak half-height
intLatencyPeaks = 4

# do we want to restrict the peak detection to for example the time during stimulus?
# Then put (0 1) here.
tplRestrictRange = (0, np.inf)

# do we want to compute the instantaneous firing rate?
boolReturnRate = True

# create a T by 2 array with stimulus onsets and offsets so we can also compute the t-test
arrEventTimes = np.transpose(np.array([vecStimulusStartTimes, vecStimulusStopTimes]))

# then run ZETA with those parameters
t = time.time()
dblZetaP, dZETA, dRate, vecLatencies = zetatest(vecSpikeTimes1, arrEventTimes,
                                                dblUseMaxDur=dblUseMaxDur,
                                                intResampNum=intResampNum,
                                                dblJitterSize=dblJitterSize,
                                                intPlot=intPlot,
                                                intLatencyPeaks=intLatencyPeaks,
                                                tplRestrictRange=tplRestrictRange,
                                                boolReturnRate=boolReturnRate)

dblElapsedT2 = time.time() - t
print(f"\nSpecified parameters (elapsed time: {dblElapsedT2:.2f} s): \
      \nzeta-test p-value: {dblZetaP}\nt-test p-value:{dZETA['dblMeanP']}")

# Note on the latencies: while the peaks of ZETA and -ZETA can be useful for diagnostic purposes,
# they are difficult to interpret, so we suggest sticking to the peak time (vecLatencies[2]),
# which is more easily interpretable. Please read the paper for more information.

# %% run the time-series zeta-test
# take subselection of data
intUseTrialNum = 480
vecStimulusStartTimesTs = vecStimulusStartTimes[0:intUseTrialNum]
vecStimulusStopTimesTs = vecStimulusStopTimes[0:intUseTrialNum]
arrEventTimesTs = np.transpose(np.array([vecStimulusStartTimesTs, vecStimulusStopTimesTs]))

# first transform the data to time-series
print('\nRunning time-series zeta-test; This will take around 40 seconds\n')
dblStartT = 0
dblEndT = vecStimulusStopTimesTs[-1] + dblUseMaxDur*5
dblSamplingRate = 50.0  # simulate acquisition rate
dblSampleDur = 1/dblSamplingRate
vecTimestamps = np.arange(dblStartT, dblEndT+dblSampleDur, dblSampleDur)
vecSpikesBinned = np.histogram(vecSpikeTimes1, bins=vecTimestamps)[0]
vecTimestamps = vecTimestamps[0:-1]
dblSmoothSd = 1.0
intSmoothRange = 2*np.ceil(dblSmoothSd).astype(int)
vecFilt = norm.pdf(range(-intSmoothRange, intSmoothRange+1), 0, dblSmoothSd)
vecFilt = vecFilt / sum(vecFilt)

# pad array
intPadSize = np.floor(len(vecFilt)/2).astype(int)
vecData = np.pad(vecSpikesBinned, ((intPadSize, intPadSize)), 'edge')

# filter
vecData = convolve(vecData, vecFilt, 'valid')

# set random seed
np.random.seed(1)

# time-series zeta-test with default parameters
t = time.time()
dblTsZetaP = zetatstest(vecTimestamps, vecData, vecStimulusStartTimesTs)[0]
dblElapsedT3 = time.time() - t
print(f"\nDefault parameters (elapsed time: {dblElapsedT3:.2f} s):\ntime-series zeta-test p-value: {dblTsZetaP}\n")

# %% run time-series zeta-test with specified parameters
# set random seed
np.random.seed(1)
t = time.time()
# run test
print('\nRunning time-series zeta-test with specified parameters; This will take around 40 seconds\n')
dblTsZetaP2, dZetaTs = zetatstest(vecTimestamps, vecData, arrEventTimesTs,
                                  dblUseMaxDur=None, intResampNum=100, intPlot=1,
                                  dblJitterSize=2.0, boolDirectQuantile=False)

dblElapsedT4 = time.time() - t
print(f"\nSpecified parameters (elapsed time: {dblElapsedT4:.2f} s): \
      \ntime-series zeta-test p-value: {dblTsZetaP2}\nt-test p-value:{dZetaTs['dblMeanP']}")
