# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:01:24 2023

@author: Jorrit
"""
import numpy as np
from zetapy.dependencies import getPseudoSpikeVectors


vecSpikeTimes = np.arange(0,10,0.123)
vecEventT = np.arange(0,10,1.0)
dblWindowDur = 1.0;
boolDiscardEdges = False;

vecPseudoSpikeTimes,vecPseudoStartT = getPseudoSpikeVectors(vecSpikeTimes,vecEventT,dblWindowDur,boolDiscardEdges)

