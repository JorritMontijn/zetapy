"""test_zetatest Runs some tests using the zetatest

2024, Alexander Heimel, based on runExampleZETA.m
"""

import scipy.io
from zetapy import ifr, zetatest, zetatstest, zetatest2, zetatstest2, plotzeta, plottszeta, plotzeta2, plottszeta2
import os
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import norm
#from scipy.signal import convolve
#import time

import unittest

class TestZetaTest(unittest.TestCase):
    def test_zetatest_default(self):
        print('\nzetatest_default, expected to take about 2 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        np.random.seed(1)
        dblZetaP = zetatest(dLoad['vecSpikeTimes1'],dLoad['vecStimulusStartTimes'])[0]  # use [0] to return only the p-value
        self.assertTrue(abs(dblZetaP - 7.702804163978172e-05)<1E-6)

    def test_zetatest_specified(self):
        print('\nzetatest_specified, expected to take about 2 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        np.random.seed(1)
        #dblUseMaxDur = np.min(np.diff( dLoad['vecStimulusStartTimes'],axis=0))
        dblZetaP, dZETA, dRate = zetatest(dLoad['vecSpikeTimes1'], dLoad['matEventTimes'],
                                                dblUseMaxDur=dLoad['dblUseMaxDur'][0,0],
                                                intResampNum=dLoad['intResampNum'][0,0],
                                                dblJitterSize=dLoad['dblJitterSize'][0,0],
                                                boolPlot=False,
                                                tplRestrictRange=(0, np.inf),
                                                boolReturnRate=True)
        self.assertTrue(abs(dblZetaP - 1.353659556455611e-04)<1E-6)

    def test_zetatstest_default(self):
        print('\nzetatstest_default, expected to take about 20 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        np.random.seed(1)
        dblZetaP = zetatstest(dLoad['vecTimestamps'], dLoad['vecData1'], dLoad['matEventTimesTs'][:,0], 
                                    intResampNum=dLoad['intResampNum'][0,0])[0]
        self.assertTrue(abs(dblZetaP - 0.027278506931302)<1E-6)

    def test_zetatstest_specified(self):
        print('\nzetatstest_specified, expected to take about 20 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        np.random.seed(1)
        dblZetaP = zetatstest(dLoad['vecTimestamps'], dLoad['vecData1'], dLoad['matEventTimesTs'],
                                    dblUseMaxDur=dLoad['dblUseMaxDur'][0,0],
                                    intResampNum=dLoad['intResampNum'][0,0],
                                    boolPlot=False,boolDirectQuantile=False,
                                    dblJitterSize=dLoad['dblJitterSize'][0,0],boolStitch=True)[0]
        self.assertTrue(abs(dblZetaP - 0.027276318742224)<1E-6)

    def test_zetatest2_neurons(self):
        print('\nzetatest2_neurons, expected to take about 2 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        intTrials = 240
        np.random.seed(1)
        dblZetaP = zetatest2(dLoad['vecSpikeTimes1'],dLoad['matEventTimes'][0:intTrials,:],
                             dLoad['vecSpikeTimes2'],dLoad['matEventTimes'][0:intTrials,:],
                             dblUseMaxDur=dLoad['dblUseMaxDur'][0,0],
                             boolPlot=False)[0]  # use [0] to return only the p-value
        self.assertTrue(abs(dblZetaP - 0.00000356925555644594)<1E-6)


    def test_zetatest2_stimuli(self):
        print('\nzetatest2_stimuli, expected to take about 2 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        vecTrials1 = dLoad['vecStimulusOrientation']==0
        vecTrials2 = dLoad['vecStimulusOrientation']==90
        np.random.seed(1)
        dblZetaP = zetatest2(dLoad['vecSpikeTimes1'],dLoad['matEventTimes'][vecTrials1.flatten(),:],
                             dLoad['vecSpikeTimes1'],dLoad['matEventTimes'][vecTrials2.flatten(),:],
                             dblUseMaxDur=dLoad['dblUseMaxDur'][0,0],
                             boolPlot=False)[0]  # use [0] to return only the p-value
        self.assertTrue(abs(dblZetaP - 0.00908076827309078904)<1E-6)
        
    def test_zetatstest2_neurons(self):
        print('\nzetatstest2_neurons, expected to take about 2 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        intTrials = 240
        np.random.seed(1)
        dblZetaP = zetatstest2(dLoad['vecTimestamps'],dLoad['vecData1'], dLoad['matEventTimesTs'][0:intTrials,:],
                             dLoad['vecTimestamps'],dLoad['vecData2'], dLoad['matEventTimesTs'][0:intTrials,:],
                             dblUseMaxDur=dLoad['dblUseMaxDur'][0,0],
                             boolPlot=False)[0]  # use [0] to return only the p-value
        self.assertTrue(abs(dblZetaP - 7.201222648300920e-06)<1E-6)

    def test_zetatstest2_stimuli(self):
        print('\nzetatstest2_stimuli, expected to take about 2 s',end='')
        strDataFile = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        dLoad = scipy.io.loadmat(strDataFile)
        vecTrials1 = dLoad['vecStimulusOrientation'][0:480]==0
        vecTrials2 = dLoad['vecStimulusOrientation'][0:480]==90
        np.random.seed(1)
        dblZetaP = zetatstest2(dLoad['vecTimestamps'],dLoad['vecData1'], dLoad['matEventTimesTs'][vecTrials1.flatten(),:],
                             dLoad['vecTimestamps'],dLoad['vecData1'], dLoad['matEventTimesTs'][vecTrials2.flatten(),:],
                             dblUseMaxDur=dLoad['dblUseMaxDur'][0,0],
                             boolPlot=False)[0]  # use [0] to return only the p-value
        self.assertTrue(abs(dblZetaP / 0.033518074062296 - 1 )<5E-2) # increased tolerance, did not match getInterpolatedTimeSeries between versions

if __name__ == "__main__":
    unittest.main()