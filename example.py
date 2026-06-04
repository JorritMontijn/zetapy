"""
Run example ZETA-test. This code loads data from an example cell and performs some analyses as a tutorial.

"""

import scipy.io
from zetapy import ifr, zetatest, zetatstest, zetatest2, zetatstest2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import convolve
from pathlib import Path
import time

# %% load and prepare some example data
# load data for example cell
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path(os.getcwd())
data_file = current_dir / 'zetapy' / 'ExampleDataZetaTest.mat'
data = scipy.io.loadmat(data_file)

# retrieve the spike times as an array from the field in dNeuron
spike_times_1 = data['sNeuron']['SpikeTimes'][0][0]
spike_times_2 = data['sNeuron']['SpikeTimes'][0][1]

# load stimulation information
stim = data['sStim']
stimulus_start_times = stim['StimOnTime'][0][0][0]  # unpacking Matlab array
stimulus_stop_times = stim['StimOffTime'][0][0][0]  # unpacking Matlab array
orientation = stim['Orientation'][0][0][0]  # unpacking Matlab array

# %% calculate instantaneous firing rate without performing the ZETA-test
# if we simply want to plot the neuron's response, we can use:
times, rate, ifr_results = ifr(spike_times_1, stimulus_start_times)

# plot results
f, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(times, rate)
ax.set(xlabel='Time after event (s)', ylabel='Instantaneous spiking rate (Hz)')
ax.set(title="A simple plot of the neuron's rate using ifr()")

# %% run the ZETA-test with default parameters
# set random seed
np.random.seed(1)

# run test
t = time.time()
zeta_p = zetatest(spike_times_1, stimulus_start_times)[0]  # use [0] to return only the p-value
elapsed_t = time.time() - t

print(f'\nDefault parameters (elapsed time: {elapsed_t:.2f} s):\np-value: {zeta_p}')

# %% run the ZETA-test with specified parameters
# set random seed
np.random.seed(1)

# use minimum of trial-to-trial durations as analysis window size
use_max_dur = np.min(np.diff(stimulus_start_times))

# 50 random resamplings should give us a good enough idea if this cell is responsive.
# If the p-value is close to 0.05, we should increase this number.
resamp_num = 50

# what size of jittering do we want? (multiple of dblUseMaxDur; default is 2.0)
jitter_size = 2.0

# Do we want to plot the results?
plot_enabled = True

# do we want to restrict the peak detection to for example the time during stimulus?
# Then put (0 1) here.
restrict_range = (0, np.inf)

# do we want to compute the instantaneous firing rate?
return_rate = True

# create a T by 2 array with stimulus onsets and offsets so we can also compute the t-test
event_times = np.transpose(np.array([stimulus_start_times, stimulus_stop_times]))

# then run ZETA with those parameters
t = time.time()
zeta_p, zeta_results, rate_results = zetatest(
    spike_times_1,
    event_times,
    max_duration=use_max_dur,
    resampling_number=resamp_num,
    plot_enabled=plot_enabled,
    jitter_size=jitter_size,
    restrict_range=restrict_range,
    return_rate=return_rate)

elapsed_t_2 = time.time() - t
print(f"\nSpecified parameters (elapsed time: {elapsed_t_2:.2f} s): \
      \nzeta-test p-value: {zeta_p}\nt-test p-value:{zeta_results['ttest_p_value']}")


# %% run the time-series zeta-test
# take subselection of data
use_trial_num = 480
stimulus_start_times_ts = stimulus_start_times[0:use_trial_num]
stimulus_stop_times_ts = stimulus_stop_times[0:use_trial_num]
orientation_ts = orientation[0:use_trial_num]
event_times_ts = np.transpose(np.array([stimulus_start_times_ts, stimulus_stop_times_ts]))

# first transform the data to time-series
print('\nRunning time-series zeta-test; This will take around 5 seconds\n')
start_t = 0
end_t = stimulus_stop_times_ts[-1] + use_max_dur*5
sampling_rate = 50.0  # simulate acquisition rate
sample_dur = 1/sampling_rate
timestamps = np.arange(start_t, end_t+sample_dur, sample_dur)
spikes_binned = np.histogram(spike_times_1, bins=timestamps)[0]
timestamps = timestamps[0:-1]
smooth_sd = 1.0
smooth_range = 2*np.ceil(smooth_sd).astype(int)
filt = norm.pdf(range(-smooth_range, smooth_range+1), 0, smooth_sd)
filt = filt / sum(filt)

# pad array
pad_size = np.floor(len(filt)/2).astype(int)
data_1 = np.pad(spikes_binned, ((pad_size, pad_size)), 'edge')

# filter
data_1 = convolve(data_1, filt, 'valid')

# set random seed
np.random.seed(1)

# time-series zeta-test with default parameters
t = time.time()
ts_zeta_p = zetatstest(timestamps, data_1, stimulus_start_times_ts)[0]
elapsed_t_3 = time.time() - t
print(f"\nDefault parameters (elapsed time: {elapsed_t_3:.2f} s):\ntime-series zeta-test p-value: {ts_zeta_p}\n")

# %% run time-series zeta-test with specified parameters
# set random seed
np.random.seed(1)
t = time.time()

# run test
print('\nRunning time-series zeta-test with specified parameters; This will take around 5 seconds\n')
ts_zeta_p_2, zeta_ts = zetatstest(timestamps, data_1, event_times_ts,
                                  max_duration=None, resampling_number=100, plot_enabled=True,
                                  jitter_size=2.0, direct_quantile=False)

elapsed_t_4 = time.time() - t
print(f"\nSpecified parameters (elapsed time: {elapsed_t_4:.2f} s): \
      \ntime-series zeta-test p-value: {ts_zeta_p_2}\nt-test p-value:{zeta_ts['ttest_p_value']}")


# %% run the two-sample ZETA-test
#case 1: are neurons 1 & 2 responding differently to a set of visual stimuli?
np.random.seed(1)
print('\nRunning two-sample zeta-test on two neurons, same stimuli\n')
t = time.time()
trials = 240 #that's already more than enough
resamp_num = 500 #the two-sample test is more variable, as it depends on differences, so it requires more resamplings
zeta_two_sample, zeta_2 = zetatest2(
    spike_times_1, event_times[0:trials,:], spike_times_2, event_times[0:trials,:],
    use_max_dur, resamp_num, plot_enabled=True)
elapsed_t_5 = time.time() - t
print(f"\nAre two neurons responding differently? (elapsed time: {elapsed_t_5:.2f} s): \
      \ntwo-sample zeta-test p-value: {zeta_two_sample}\nt-test p-value:{zeta_2['ttest_p_value']}")


# case 2a: is neuron 1 responding differently to gratings oriented at 0 and 90 degrees?
trials_1 = orientation == 0
trials_2 = orientation == 90
print('\nRunning two-sample zeta-test on one neuron, different stimuli\n')
t = time.time()
zeta_two_sample_2a, zeta_2a = zetatest2(
    spike_times_1, event_times[trials_1,:], spike_times_1, event_times[trials_2,:],
    use_max_dur, resamp_num, plot_enabled=True)
elapsed_t_6 = time.time() - t
print(f"\nIs neuron 1 responding differently to 0 and 90 degree stimuli? (elapsed time: {elapsed_t_6:.2f} s): \
      \ntwo-sample zeta-test p-value: {zeta_two_sample_2a}\nt-test p-value:{zeta_2a['ttest_p_value']}")

#case 2b: is neuron 2 responding differently to gratings oriented at 0 and 90 degrees?
t = time.time()
resamp_num = 1000 #this difference is close to our threshold p=0.05, so we're increasing the number of bootstraps
zeta_two_sample_2b, zeta_2b = zetatest2(
    spike_times_2, event_times[trials_1,:], spike_times_2, event_times[trials_2,:],
    use_max_dur, resamp_num, plot_enabled=True)
elapsed_t_7 = time.time() - t
print(f"\nIs neuron 2 responding differently to 0 and 90 degree stimuli? (elapsed time: {elapsed_t_7:.2f} s): \
      \ntwo-sample zeta-test p-value: {zeta_two_sample_2b}\nt-test p-value:{zeta_2b['ttest_p_value']}")


# %% finally, the two-sample time-series ZETA test
#get trials
trials_1 = orientation_ts==0
trials_2 = orientation_ts==90

#get data for neuron 2
start_t = 0
end_t = stimulus_stop_times_ts[-1] + use_max_dur*5
sampling_rate = 50.0  # simulate acquisition rate
sample_dur = 1/sampling_rate
timestamps = np.arange(start_t, end_t+sample_dur, sample_dur)
spikes_binned = np.histogram(spike_times_2, bins=timestamps)[0]
timestamps = timestamps[0:-1]
smooth_sd = 1.0
smooth_range = 2*np.ceil(smooth_sd).astype(int)
filt = norm.pdf(range(-smooth_range, smooth_range+1), 0, smooth_sd)
filt = filt / sum(filt)

# pad array
pad_size = np.floor(len(filt)/2).astype(int)
data_2 = np.pad(spikes_binned, ((pad_size, pad_size)), 'edge')

# filter
data_2 = convolve(data_2, filt, 'valid')

#set parameters
resamp_num = 250
plot_enabled = True
direct_quantile = False
super_res_factor = 100
#case 1: are neurons 1 & 2 responding differently to a set of visual stimuli?
np.random.seed(1)
print('\nRunning two-sample time-series zeta-test on two neurons, same stimuli\n')
t = time.time()
ts_zeta_two_sample, ts_zeta_2 = zetatstest2(
    timestamps, data_1, event_times_ts, timestamps, data_2, event_times_ts,
    use_max_dur, resamp_num, plot_enabled, direct_quantile, super_res_factor)

elapsed_t_8 = time.time() - t
print(f"\nAre two neurons responding differently? (elapsed time: {elapsed_t_8:.2f} s): \
      \ntwo-sample time-series zeta-test p-value: {ts_zeta_two_sample}\nt-test p-value:{ts_zeta_2['ttest_p_value']}")


#case 2: is neuron 1 responding differently to gratings oriented at 0 and 90 degrees?
print('\nRunning two-sample time-series zeta-test on one neuron, different stimuli\n')
t = time.time()
ts_zeta_two_sample_b, ts_zeta_2b = zetatstest2(
    timestamps, data_1, event_times_ts[trials_1,:], timestamps, data_1, event_times_ts[trials_2,:],
    use_max_dur, resamp_num, plot_enabled, direct_quantile, super_res_factor)
elapsed_t_9 =  time.time() - t
print(f"\nIs neuron 1 responding differently to 0 and 90 degree stimuli? (elapsed time: {elapsed_t_9:.2f} s): \
      \ntwo-sample time-series zeta-test p-value: {ts_zeta_two_sample_b}\nt-test p-value:{ts_zeta_2b['ttest_p_value']}")
