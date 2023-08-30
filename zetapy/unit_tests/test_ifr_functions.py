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

from zetapy import zetatest,ifr
from zetapy.dependencies import calcZetaOne,getZetaP,getGumbel,getTempOffsetOne,getSpikeT,getPseudoSpikeVectors,flatten
from zetapy.ifr_dependencies import getMultiScaleDeriv,getPeak,getOnset

# %% set random seed
#passes, same as matlab
np.random.seed(1)
dblEndT = 10.0
dblUseMaxDur = 1.0
intSpikeNum = 100
vecSpikeTimes = dblEndT*np.sort(np.random.rand(intSpikeNum,1),axis=0) #the same
vecEventTimes = np.arange(0,dblEndT,dblUseMaxDur) #the same

# %% get deviation vector
dblWindowDur = 1.0;
boolDiscardEdges = False;
vecPseudoSpikeTimes,vecPseudoEventT = getPseudoSpikeVectors(vecSpikeTimes,vecEventTimes,dblWindowDur,boolDiscardEdges)
vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT = getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblWindowDur)

# %% test getmultiscalederiv
vecRate,dMSD = getMultiScaleDeriv(vecSpikeT, vecRealDeviation)

#with smoothing
dblSmoothSd = 1.0
dblMinScale = -20.0
dblBase = 1.3
dblMeanRate = 1.0
dblUseMaxDur = 1.0
vecRate2,dMSD2 = getMultiScaleDeriv(vecSpikeT,vecRealDeviation,
                                    dblSmoothSd=dblSmoothSd,dblMinScale=dblMinScale,dblBase=dblBase,dblMeanRate=dblMeanRate,dblUseMaxDur=dblUseMaxDur)

# %% test peak
#passes
dPeak = getPeak(vecRate2, dMSD2['vecT'])
dblPeakRate = dPeak['dblPeakValue']
dblPeakTime = dPeak['dblPeakTime']
dblPeakWidth = dPeak['dblPeakWidth']
print(dblPeakRate)
print(dblPeakTime)
print(dblPeakWidth)

# %% test onset
#passes
dOnset = getOnset(vecRate2,dMSD2['vecT'],dblPeakTime=dblPeakTime)
dblOnset = dOnset['dblOnset']
dblValue = dOnset['dblValue']
dblBaseValue = dOnset['dblBaseValue']
dblPeakTime = dOnset['dblPeakTime']
dblPeakValue = dOnset['dblPeakValue']

# %% test ifr
#passes
vecTime, vecRate,dIFR = ifr(vecSpikeTimes, vecEventTimes)

plt.plot(vecTime,vecRate)

# %% test plotting
#passes
np.random.seed(1)
intResampNum = 100
intPlot = 4
intLatencyPeaks = 4
tplRestrictRange = (-np.inf,np.inf);
boolDirectQuantile = False
dblJitterSize = 2
boolStitch = True
arrEventTimes = np.vstack((vecEventTimes, vecEventTimes+0.5)).T
dblZetaP,dZETA,dRate,vecLatencies = zetatest(vecSpikeTimes,arrEventTimes,
    dblUseMaxDur=dblUseMaxDur, intResampNum=intResampNum, intPlot=intPlot, dblJitterSize=dblJitterSize,
    intLatencyPeaks=intLatencyPeaks, tplRestrictRange=tplRestrictRange,
    boolStitch=boolStitch, boolDirectQuantile=boolDirectQuantile,
    boolReturnRate=True)