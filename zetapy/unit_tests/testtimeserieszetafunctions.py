# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:21:36 2023

@author: Jorrit
"""
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve
import matplotlib.pyplot as plt

from zetapy import zetatstest
from zetapy.ts_dependencies import getPseudoTimeSeries,getTsRefT,getInterpolatedTimeSeries,calcTsZetaOne,getTimeseriesOffsetOne

# %% set random seed
#passes, same as matlab
intSeed = 0
np.random.seed(intSeed)
dblEndT = 12.9
dblStartT = -2.9
dblTotDur = dblEndT-dblStartT
dblWindowDur = 1.0
dblStimDur = 0.5
dblSamplingRate = 25.0 #Hz
dblSampleDur = 1/dblSamplingRate

vecSpikeTimes = dblTotDur*np.sort(np.random.rand(1000,1),axis=0) + dblStartT #the same
vecEventTimes = np.arange(0,10,dblWindowDur) #the same

#add stimulus spikes
vecSpikeTimesOn = dblTotDur*np.sort(np.random.rand(1000,1),axis=0) + dblStartT #the same
indKeepSpikesOn = np.full(vecSpikeTimesOn.shape, False)
vecEventTimesOff = vecEventTimes + dblStimDur
for intTrial,dblTrialStartT in enumerate(vecEventTimes):
    dblTrialStopT = vecEventTimesOff[intTrial]
    indKeepSpikesOn[np.logical_and(vecSpikeTimesOn > dblTrialStartT,vecSpikeTimesOn < dblTrialStopT)] = True

vecSpikeTimesOn = vecSpikeTimesOn[indKeepSpikesOn]
vecSpikeTimes = np.sort(np.concatenate((vecSpikeTimes[:,0],vecSpikeTimesOn)))

# %% transform to time-series
np.random.seed(intSeed)
vecTimestamps =  np.arange(dblStartT,dblEndT+dblSampleDur,dblSampleDur)
vecSpikesBinned = np.histogram(vecSpikeTimes, bins=vecTimestamps)[0]
vecTimestamps = vecTimestamps[0:-1]
dblSmoothSd = 2.0
intSmoothRange = 2*np.ceil(dblSmoothSd).astype(int)
vecFilt = norm.pdf(range(-intSmoothRange, intSmoothRange+1), 0, dblSmoothSd)
vecFilt = vecFilt / sum(vecFilt)

# pad array
intPadSize = np.floor(len(vecFilt)/2).astype(int)
vecData = np.pad(vecSpikesBinned, ((intPadSize, intPadSize)), 'edge')

# filter
vecData = convolve(vecData, vecFilt, 'valid')

# %% test getPseudoSpikeVectors
#passes, same as matlab
np.random.seed(intSeed)
vecPseudoTime, vecPseudoData, vecPseudoEventTime = getPseudoTimeSeries(vecTimestamps, vecData, vecEventTimes, dblWindowDur)


# %% test getTsRefT
np.random.seed(intSeed)
vecTime = getTsRefT(vecPseudoTime,vecPseudoTime,dblWindowDur)

# %% getInterpolatedTimeSeries
np.random.seed(intSeed)
vecUsedTime,matDataPerTrial = getInterpolatedTimeSeries(vecTimestamps,vecData,vecEventTimes,vecTime)

# %% getTimeseriesOffsetOne
np.random.seed(intSeed)
vecRealDeviation, vecRealFrac, vecRealFracLinear, vecRealTime = getTimeseriesOffsetOne(vecPseudoTime,vecPseudoData, vecPseudoEventTime, dblWindowDur)

# %% calctszetaone
np.random.seed(intSeed)
dblUseMaxDur=dblWindowDur
intResampNum=100
dblJitterSize=2.0
boolDirectQuantile=False
dZETA = calcTsZetaOne(vecTimestamps,vecData, vecEventTimes, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize)

# %% zetatstest
#not exactly the same, but close enough - getRefT uses uniquetol which does not exist in python
np.random.seed(intSeed)
pZeta,dZeta = zetatstest(vecTimestamps, vecData, vecEventTimes,
                         dblUseMaxDur=None, intResampNum=100, intPlot=0, dblJitterSize=2.0, boolDirectQuantile=False)

# %% zetatstest
#with ttest
np.random.seed(intSeed)
arrEventTimes = np.concatenate((vecEventTimes.reshape(-1,1),vecEventTimes.reshape(-1,1)+dblStimDur),axis=1)
pZeta,dZeta = zetatstest(vecTimestamps, vecData, arrEventTimes,
                         dblUseMaxDur=None, intResampNum=100, intPlot=1, dblJitterSize=2.0, boolDirectQuantile=False)
