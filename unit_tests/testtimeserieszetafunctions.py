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
np.random.seed(100)
dblEndT = 12.9
dblStartT = -2.9
dblTotDur = dblEndT-dblStartT
dblWindowDur = 1.0;
dblSamplingRate = 25.0 #Hz
dblSampleDur = 1/dblSamplingRate

vecSpikeTimes = dblTotDur*np.sort(np.random.rand(1000,1),axis=0) + dblStartT #the same
vecEventTimes = np.arange(0,10,dblWindowDur) #the same

#transform to time-series
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

# # %% test getPseudoSpikeVectors
# #passes, same as matlab

# vecPseudoTime, vecPseudoData, vecPseudoEventTime = getPseudoTimeSeries(vecTimestamps, vecData, vecEventTimes, dblWindowDur)


# # %% test getTsRefT
# vecTime = getTsRefT(vecPseudoTime,vecPseudoTime,dblWindowDur)

# # %% getInterpolatedTimeSeries
# vecUsedTime,matDataPerTrial = getInterpolatedTimeSeries(vecTimestamps,vecData,vecEventTimes,dblWindowDur,vecTime)

# # %% getTimeseriesOffsetOne
# vecRealDeviation, vecRealFrac, vecRealFracLinear, vecRealTime = getTimeseriesOffsetOne(vecPseudoTime,vecPseudoData, vecPseudoEventTime, dblWindowDur)

# # %% calctszetaone
# dblUseMaxDur=dblWindowDur
# intResampNum=100
# dblJitterSize=2.0
# boolDirectQuantile=False
# dZETA = calcTsZetaOne(vecTimestamps,vecData, vecEventTimes, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize)

# %% zetatstest
#not exactly the same, probably because getRefT uses uniquetol which does not exist in python
pZeta,dZeta = zetatstest(vecTimestamps, vecData, vecEventTimes,
                         dblUseMaxDur=None, intResampNum=1000, intPlot=0, dblJitterSize=2.0, boolDirectQuantile=False)
