"""test_zetatest Runs some tests using the zetatest

2024, Alexander Heimel, based on runExampleZETA.m
"""

import scipy.io
from zetapy import zetatest, zetatstest, zetatest2, zetatstest2, plotzeta, plottszeta, plotzeta2, plottszeta2
import os
import numpy as np
import time

import unittest

class TestZetaTest(unittest.TestCase):
    def test_zetatest_default(self):
        print('\nzetatest_default ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        np.random.seed(1)
        zeta_p = zetatest(loaded_data['vecSpikeTimes1'], loaded_data['vecStimulusStartTimes'])[0]  # use [0] to return only the p-value
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p - 7.702804163978172e-05) < 1E-6)

    def test_zetatest_specified(self):
        print('\nzetatest_specified ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        np.random.seed(1)
        # max_dur = np.min(np.diff(loaded_data['vecStimulusStartTimes'], axis=0))
        zeta_p, zeta, rate = zetatest(loaded_data['vecSpikeTimes1'], loaded_data['matEventTimes'],
                                      max_duration=loaded_data['dblUseMaxDur'][0, 0],
                                      resampling_number=loaded_data['intResampNum'][0, 0],
                                      jitter_size=loaded_data['dblJitterSize'][0, 0],
                                      plot_enabled=False,
                                      restrict_range=(0, np.inf),
                                      return_rate=True)
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p - 1.353659556455611e-04) < 1E-6)

    def test_zetatstest_default(self):
        print('\nzetatstest_default ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        np.random.seed(1)
        zeta_p = zetatstest(loaded_data['vecTimestamps'], loaded_data['vecData1'], loaded_data['matEventTimesTs'][:, 0],
                            resampling_number=loaded_data['intResampNum'][0, 0])[0]
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p - 0.027278506931302) < 0.001)

    def test_zetatstest_specified(self):
        print('\nzetatstest_specified ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        np.random.seed(1)
        zeta_p = zetatstest(loaded_data['vecTimestamps'], loaded_data['vecData1'], loaded_data['matEventTimesTs'],
                            max_duration=loaded_data['dblUseMaxDur'][0, 0],
                            resampling_number=loaded_data['intResampNum'][0, 0],
                            plot_enabled=False,
                            direct_quantile=False,
                            jitter_size=loaded_data['dblJitterSize'][0, 0],
                            stitch_enabled=True)[0]
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p - 0.027276318742224) < 0.001)

    def test_zetatest2_neurons(self):
        print('\nzetatest2_neurons ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        trials = 240
        np.random.seed(1)
        zeta_p = zetatest2(loaded_data['vecSpikeTimes1'], loaded_data['matEventTimes'][0:trials, :],
                           loaded_data['vecSpikeTimes2'], loaded_data['matEventTimes'][0:trials, :],
                           max_duration=loaded_data['dblUseMaxDur'][0, 0],
                           plot_enabled=False)[0]  # use [0] to return only the p-value
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p - 0.00000356925555644594) < 1E-6)


    def test_zetatest2_stimuli(self):
        print('\nzetatest2_stimuli ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        trials_1 = loaded_data['vecStimulusOrientation'] == 0
        trials_2 = loaded_data['vecStimulusOrientation'] == 90
        np.random.seed(1)
        zeta_p = zetatest2(loaded_data['vecSpikeTimes1'], loaded_data['matEventTimes'][trials_1.flatten(), :],
                           loaded_data['vecSpikeTimes1'], loaded_data['matEventTimes'][trials_2.flatten(), :],
                           max_duration=loaded_data['dblUseMaxDur'][0, 0],
                           plot_enabled=False)[0]  # use [0] to return only the p-value
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p - 0.00908076827309078904) < 1E-6)
        
    def test_zetatstest2_neurons(self):
        print('\nzetatstest2_neurons ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        trials = 240
        np.random.seed(1)
        zeta_p = zetatstest2(loaded_data['vecTimestamps'], loaded_data['vecData1'], loaded_data['matEventTimesTs'][0:trials, :],
                             loaded_data['vecTimestamps'], loaded_data['vecData2'], loaded_data['matEventTimesTs'][0:trials, :],
                             max_duration=loaded_data['dblUseMaxDur'][0, 0],
                             plot_enabled=False)[0]  # use [0] to return only the p-value
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p - 7.201222648300920e-06) < 1E-6)

    def test_zetatstest2_stimuli(self):
        print('\nzetatstest2_stimuli ',end='')
        time_start = time.time()
        data_file = os.path.join(os.path.dirname(__file__), 'testZetaTestData.mat')
        loaded_data = scipy.io.loadmat(data_file)
        trials_1 = loaded_data['vecStimulusOrientation'][0:480] == 0
        trials_2 = loaded_data['vecStimulusOrientation'][0:480] == 90
        np.random.seed(1)
        zeta_p = zetatstest2(loaded_data['vecTimestamps'], loaded_data['vecData1'], loaded_data['matEventTimesTs'][trials_1.flatten(), :],
                             loaded_data['vecTimestamps'], loaded_data['vecData1'], loaded_data['matEventTimesTs'][trials_2.flatten(), :],
                             max_duration=loaded_data['dblUseMaxDur'][0, 0],
                             plot_enabled=False)[0]  # use [0] to return only the p-value
        print(f' took {np.round(time.time()-time_start, 1)} s', end='')
        self.assertTrue(abs(zeta_p / 0.033518074062296 - 1) < 5E-2) # increased tolerance, did not match getInterpolatedTimeSeries between versions

if __name__ == "__main__":
    unittest.main()