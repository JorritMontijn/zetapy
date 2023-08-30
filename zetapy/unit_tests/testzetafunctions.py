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
from zetapy.dependencies import calcZetaOne,getZetaP,getGumbel,getTempOffsetOne,getSpikeT,getPseudoSpikeVectors,flatten

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
dblBaseSpikingRate = 1.0
dblStimSpikingRate = 10.0

vecSpikeTimes = dblTotDur*np.sort(np.random.rand(int(dblBaseSpikingRate*dblTotDur),1),axis=0) + dblStartT #the same
vecEventTimes = np.arange(0,10,dblWindowDur) #the same

#add stimulus spikes
vecSpikeTimesOn = dblTotDur*np.sort(np.random.rand(int(dblStimSpikingRate*dblTotDur),1),axis=0) + dblStartT #the same
indKeepSpikesOn = np.full(vecSpikeTimesOn.shape, False)
vecEventTimesOff = vecEventTimes + dblStimDur
for intTrial,dblTrialStartT in enumerate(vecEventTimes):
    dblTrialStopT = vecEventTimesOff[intTrial]
    indKeepSpikesOn[np.logical_and(vecSpikeTimesOn > dblTrialStartT,vecSpikeTimesOn < dblTrialStopT)] = True

vecSpikeTimesOn = vecSpikeTimesOn[indKeepSpikesOn]
vecSpikeTimes = np.sort(np.concatenate((vecSpikeTimes[:,0],vecSpikeTimesOn)))

# %% test getPseudoSpikeVectors
#passes, same as matlab
np.random.seed(1)
dblWindowDur = 1.0;
boolDiscardEdges = False;

vecPseudoSpikeTimes,vecPseudoEventT = getPseudoSpikeVectors(vecSpikeTimes,vecEventTimes,dblWindowDur,boolDiscardEdges)

# %% test gettempoffsetone
#passes, same as matlab
np.random.seed(1)
vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT = getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblWindowDur)

# %% test getzetaone
#passes, same as matlab (if the jitter matrix is set to be identical!)
np.random.seed(1)
intResampNum = 100
boolDirectQuantile =False
dblJitterSize = 2.0
boolStitch = True
dZETA_One = calcZetaOne(vecSpikeTimes, vecEventTimes, dblWindowDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch,False)
vecSpikeT_cZO = np.reshape(np.array(dZETA_One['vecSpikeT']),(-1,1)) # first value is missing
vecSpikeT_cZO = dZETA_One['vecSpikeT']
vecRealDeviation_cZO = dZETA_One['vecRealDeviation']
#plt.plot(vecSpikeT_cZO,vecRealDeviation_cZO)

# %% test zetatest
#passes, same as matlab
np.random.seed(1)
p,dZETA,dRate,vecLatencies = zetatest(vecSpikeTimes,vecEventTimes)

# %% zetatest
#with ttest
arrEventTimes = np.concatenate((vecEventTimes.reshape(-1,1),vecEventTimes.reshape(-1,1)+dblStimDur),axis=1)
p,dZETA,dRate,vecLatencies =  zetatest(vecSpikeTimes, arrEventTimes,
             dblUseMaxDur=dblWindowDur, intResampNum=intResampNum, intPlot=1, dblJitterSize=2.0, boolDirectQuantile=False, boolReturnRate=True)

# %% ifr
vecTime,vecRate,dIFR = ifr(vecSpikeTimes, arrEventTimes,
        dblUseMaxDur=dblWindowDur, dblSmoothSd=2.0, dblMinScale=None, dblBase=1.5)

# %% legacy
pZeta0,vecLatencies0,dZeta0,dRate0 = getZeta(vecSpikeTimes, arrEventTimes, 
                                             dblUseMaxDur=dblWindowDur, intResampNum=intResampNum, intPlot=1,
            intLatencyPeaks=2, tplRestrictRange=(-np.inf, np.inf),
            boolReturnRate=True, boolReturnZETA=True)