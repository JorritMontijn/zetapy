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

from zetapy import zetatest
from zetapy.dependencies import calcZetaOne,getZetaP,getGumbel,getTempOffsetOne,getSpikeT,getPseudoSpikeVectors,flatten

# %% set random seed
#passes, same as matlab
np.random.seed(1)
dblEndT = 10.0
vecSpikeTimes = dblEndT*np.sort(np.random.rand(100,1),axis=0) #the same
vecEventTimes = np.arange(0,dblEndT,1.0) #the same

# %% test getPseudoSpikeVectors
#passes, same as matlab
dblWindowDur = 1.0;
boolDiscardEdges = False;

vecPseudoSpikeTimes,vecPseudoEventT = getPseudoSpikeVectors(vecSpikeTimes,vecEventTimes,dblWindowDur,boolDiscardEdges)

# %% test gettempoffsetone
#passes, same as matlab
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
plt.plot(vecSpikeT_cZO,vecRealDeviation_cZO)

# %% test zetatest
#passes, same as matlab
np.random.seed(1)
p,dZETA,dRate,vecLatencies = zetatest(vecSpikeTimes,vecEventTimes)

print(p) #not the same