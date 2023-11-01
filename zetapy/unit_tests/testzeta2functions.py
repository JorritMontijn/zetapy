# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:09:38 2023

@author: Jorrit
"""
import numpy as np
import logging
import scipy
from scipy import stats
from math import pi, sqrt, exp
from collections.abc import Iterable
import matplotlib.pyplot as plt

from zetapy import zetatest,ifr,getZeta
from zetapy.dependencies import calcZetaTwo,getZetaP,getTempOffsetTwo,getSpikesInTrial,flatten,getUniqueSpikes

# %% set random seed
#passes, same as matlab
np.random.seed(1)
dblEndT = 12.9
dblStartT = -2.9
dblTotDur = dblEndT-dblStartT
dblWindowDur = 1.0
dblStimDur = 0.5
dblSamplingRate = 25.0 #Hz
dblSampleDur = 1/dblSamplingRate
dblBaseSpikingRate1 = 1.0
dblStimSpikingRate1 = 10.0
dblBaseSpikingRate2 = 1.0
dblStimSpikingRate2 = 5.0
vecEventTimes = np.arange(0,10,dblWindowDur) #the same

#neuron 1
vecSpikeTimes1 = dblTotDur*np.sort(np.random.rand(int(dblBaseSpikingRate1*dblTotDur),1),axis=0) + dblStartT #the same
vecSpikeTimesOn = dblTotDur*np.sort(np.random.rand(int(dblStimSpikingRate1*dblTotDur),1),axis=0) + dblStartT #the same
indKeepSpikesOn = np.full(vecSpikeTimesOn.shape, False)
vecEventTimesOff = vecEventTimes + dblStimDur
for intTrial,dblTrialStartT in enumerate(vecEventTimes):
    dblTrialStopT = vecEventTimesOff[intTrial]
    indKeepSpikesOn[np.logical_and(vecSpikeTimesOn > dblTrialStartT,vecSpikeTimesOn < dblTrialStopT)] = True

vecSpikeTimesOn = vecSpikeTimesOn[indKeepSpikesOn]
vecSpikeTimes1 = np.sort(np.concatenate((vecSpikeTimes1[:,0],vecSpikeTimesOn)))

#neuron 2
vecSpikeTimes2 = dblTotDur*np.sort(np.random.rand(int(dblBaseSpikingRate2*dblTotDur),1),axis=0) + dblStartT #the same
vecSpikeTimesOn = dblTotDur*np.sort(np.random.rand(int(dblStimSpikingRate2*dblTotDur),1),axis=0) + dblStartT #the same
indKeepSpikesOn = np.full(vecSpikeTimesOn.shape, False)
vecEventTimesOff = vecEventTimes + dblStimDur
for intTrial,dblTrialStartT in enumerate(vecEventTimes):
    dblTrialStopT = vecEventTimesOff[intTrial]
    indKeepSpikesOn[np.logical_and(vecSpikeTimesOn > dblTrialStartT,vecSpikeTimesOn < dblTrialStopT)] = True

vecSpikeTimesOn = vecSpikeTimesOn[indKeepSpikesOn]
vecSpikeTimes2 = np.sort(np.concatenate((vecSpikeTimes2[:,0],vecSpikeTimesOn)))

# %% test getTempOffsetTwo
#passes, same as matlab
np.random.seed(1)
dblUseMaxDur = 1.0
intResampNum = 250
boolDirectQuantile = False;
arrEventTimes = np.concatenate((vecEventTimes.reshape(-1,1),vecEventTimes.reshape(-1,1)+dblStimDur),axis=1)

# %% get spikes per trial
vecEventT1 = arrEventTimes[:, 0]
vecEventT2 = arrEventTimes[:, 0]
cellTrialPerSpike1, cellTimePerSpike1 = getSpikesInTrial(vecSpikeTimes1, vecEventT1, dblUseMaxDur)
cellTrialPerSpike2, cellTimePerSpike2 = getSpikesInTrial(vecSpikeTimes2, vecEventT2, dblUseMaxDur)

# %% getTempOffsetTwo
# get difference
vecSpikeT, vecRealDiff, vecRealFrac1, vecThisSpikeTimes1, vecRealFrac2, vecThisSpikeTimes2 = \
    getTempOffsetTwo(cellTimePerSpike1, cellTimePerSpike2, dblUseMaxDur)
    
# %% calcZetaTwo
dZETA_Two = calcZetaTwo(vecSpikeTimes1, arrEventTimes, vecSpikeTimes2,
                        arrEventTimes, dblUseMaxDur, intResampNum, boolDirectQuantile)
