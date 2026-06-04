# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:08:44 2023

@author: Jorrit
"""
import numpy as np
import time
import logging
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from zetapy.ts_dependencies import get_interpolated_time_series

# %% plot_ts_zeta_two
def plottszeta2(zeta_data, plot_random_samples=50):
    '''
    Creates figure for two-sample time-series ZETA-test

    Syntax:
    plot_ts_zeta_two(zeta_data, plot_random_samples=50)
    '''
    
    # unpack zeta_data
    try:
        zeta_p = zeta_data['zeta_p_value']
        zeta_score = zeta_data['zeta_score']
        zeta_deviation = zeta_data['zeta_deviation']
        zeta_time = zeta_data['zeta_time']
        mean_z_score = zeta_data['ttest_z_score']
        mean_p_value = zeta_data['ttest_p_value']
        zeta_deviation_inv_sign = zeta_data['zeta_deviation_inv_sign']
        zeta_time_inv_sign = zeta_data['zeta_time_inv_sign']
        reference_time = zeta_data['reference_time']
        real_difference = zeta_data['real_difference']
        random_differences = zeta_data['random_differences']
        real_fraction1 = zeta_data['real_fraction1']
        real_fraction2 = zeta_data['real_fraction2']
        trace_per_trial1 = zeta_data['trace_per_trial1']
        trace_per_trial2 = zeta_data['trace_per_trial2']
    except KeyError as e:
        raise Exception(
            f"plot_ts_zeta_two error: information is missing from zeta_data dictionary: {e}")

    # %% plot
    # Plot maximally 50 traces (or however many are requested)
    plot_random_samples = np.min([random_differences.shape[0], plot_random_samples])

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=300)
    
    # top left: heat map 1
    x_start = reference_time[1]
    x_end = reference_time[-1]
    x_width = x_end - x_start
    num_trials1 = trace_per_trial1.shape[0]
    y_height = num_trials1 - 1
    img_plot = ax1.imshow(trace_per_trial1, interpolation='none', extent=[x_start, x_end, 1, num_trials1])
    ax1.set_aspect((x_width / y_height) / 2)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial number',
            title='Cond1; Color indicates data value')
    fig.colorbar(img_plot, ax=ax1)

    # bottom left: heat map 2
    x_start = reference_time[1]
    x_end = reference_time[-1]
    x_width = x_end - x_start
    num_trials2 = trace_per_trial2.shape[0]
    y_height = num_trials2 - 1
    img_plot = ax2.imshow(trace_per_trial2, interpolation='none', extent=[x_start, x_end, 1, num_trials2])
    ax2.set_aspect((x_width / y_height) / 2)
    ax2.set(xlabel='Time after event (s)', ylabel='Trial number',
            title='Cond2; Color indicates data value')
    fig.colorbar(img_plot, ax=ax2)
    
    # top right: cumulative sums
    ax3.plot(reference_time, real_fraction1)
    ax3.plot(reference_time, real_fraction2)
    ax3.set(xlabel='Time after event (s)', ylabel='Scaled cumulative data (s)')

    # bottom right: deviation with random bootstraps
    for i in range(plot_random_samples):
        ax4.plot(reference_time, random_differences[i,:], color=[0.7, 0.7, 0.7])
    ax4.plot(reference_time, real_difference)
    ax4.plot(zeta_time, zeta_deviation, 'bx')
    ax4.plot(zeta_time_inv_sign, zeta_deviation_inv_sign, 'b*')
    ax4.set(xlabel='Time after event (s)', ylabel='Difference in cumulative density')
    if mean_z_score is not None:
        ax4.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f}), z(Hz)={mean_z_score:.3f} (p={mean_p_value:.3f})')
    else:
        ax4.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f})')

    fig.tight_layout()
    plt.show()

# %% plot_zeta_two
def plotzeta2(spike_times1, event_times1, spike_times2, event_times2, zeta_data,
              plot_random_samples=50, plot_spike_number=10000):
    '''
    Creates figure for two-sample ZETA-test

    Syntax:
    plot_zeta_two(spike_times1, event_times1, spike_times2, event_times2, zeta_data,
              plot_random_samples=50, plot_spike_number=10000)

    Parameters
    ----------
    spike_times1 : 1D array (float)
        spike times (in seconds) for condition 1.
    event_times1 : 1D or 2D array (float)
        event on times (s) for condition 1, or [T x 2] including event off times to calculate mean-rate difference.
    spike_times2 : 1D array (float)
        spike times (in seconds) for condition 2.
    event_times2 : 1D or 2D array (float)
        event on times (s) for condition 2, or [T x 2] including event off times to calculate mean-rate difference.
    zeta_data : dict
        Output of zetatest2.
    plot_random_samples : int, optional
        Maximum number of random resampling to plot. The default is 50.
    plot_spike_number : int, optional
        Maximum number of spikes to plot. The default is 10000.


    Code by Jorrit Montijn

    Version history:
    1.0 - 25 October 2023 Created by Jorrit Montijn
    '''
    
    # %% check input
    
    # spike_times1 must be [S by 1] array
    assert (len(spike_times1.shape) == 1 or spike_times1.shape[1] == 1) and issubclass(
        spike_times1.dtype.type, np.floating), "Input spike_times1 is not a 1D float np.array with >2 spike times"
    spike_times1 = np.sort(spike_times1.flatten(), axis=0)

    # spike_times2 must be [S by 1] array
    assert (len(spike_times2.shape) == 1 or spike_times2.shape[1] == 1) and issubclass(
        spike_times2.dtype.type, np.floating), "Input spike_times2 is not a 1D float np.array with >2 spike times"
    spike_times2 = np.sort(spike_times2.flatten(), axis=0)

    # ensure orientation and assert that event_times1 is a 1D or N-by-2 array of floats
    assert len(event_times1.shape) < 3 and issubclass(
        event_times1.dtype.type, np.floating), "Input event_times1 is not a 1D or 2D float np.array"
    if len(event_times1.shape) > 1:
        if event_times1.shape[1] < 3:
            pass
        elif event_times1.shape[0] < 3:
            event_times1 = event_times1.T
        else:
            raise Exception(
                "Input error: event_times1 must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        event_times1 = np.reshape(event_times1, (-1, 1))
    # define event starts
    event_starts1 = event_times1[:, 0]

    # ensure orientation and assert that event_times2 is a 1D or N-by-2 array of floats
    assert len(event_times2.shape) < 3 and issubclass(
        event_times2.dtype.type, np.floating), "Input event_times2 is not a 1D or 2D float np.array"
    if len(event_times2.shape) > 1:
        if event_times2.shape[1] < 3:
            pass
        elif event_times2.shape[0] < 3:
            event_times2 = event_times2.T
        else:
            raise Exception(
                "Input error: event_times2 must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        event_times2 = np.reshape(event_times2, (-1, 1))
    # define event starts
    event_starts2 = event_times2[:, 0]

    # unpack zeta_data
    try:
        max_duration = zeta_data['max_duration']
        zeta_score = zeta_data['zeta_score']
        zeta_p = zeta_data['zeta_p_value']
        zeta_deviation = zeta_data['zeta_deviation']
        zeta_time = zeta_data['zeta_time']
        zeta_index = zeta_data['zeta_index']
        
        deviation_inv_sign = zeta_data['deviation_inv_sign']
        zeta_time_inv_sign = zeta_data['zeta_time_inv_sign']
        zeta_index_inv_sign = zeta_data['zeta_index_inv_sign']

        mean_z_score = zeta_data['ttest_z_score']
        mean_p_value = zeta_data['ttest_p_value']

        spike_time_vector = zeta_data['spike_time_vector']
        real_difference = zeta_data['real_difference']
        real_fraction1 = zeta_data['real_fraction1']
        real_fraction2 = zeta_data['real_fraction2']
        random_times = zeta_data['random_times']
        random_differences = zeta_data['random_differences']

    except KeyError as e:
        raise Exception(
            f"plot_zeta_two error: information is missing from zeta_data dictionary: {e}")

    # %% plot
    # Plot maximally 50 traces (or however many are requested)
    plot_random_samples = np.min([len(random_times), plot_random_samples])

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=300)

    # reduce spikes
    if spike_times1.size > plot_spike_number or spike_times2.size > plot_spike_number:
        reduce_spikes_by = min(spike_times1.size / plot_spike_number, spike_times2.size / plot_spike_number)
        plot_spike_number1 = np.round(reduce_spikes_by * spike_times1.size).astype(int)
        plot_spike_number2 = np.round(reduce_spikes_by * spike_times2.size).astype(int)
        spike_times1_reduced = spike_times1[np.round(np.linspace(0, spike_times1.size-1, plot_spike_number1)).astype(int)]
        spike_times2_reduced = spike_times2[np.round(np.linspace(0, spike_times2.size-1, plot_spike_number2)).astype(int)]
    else:
        spike_times1_reduced = spike_times1
        spike_times2_reduced = spike_times2

    # top left: raster 1
    for i, t in enumerate(event_starts1):
        indices = np.bitwise_and(spike_times1_reduced >= t, spike_times1_reduced <= t + max_duration)
        event_spikes = spike_times1_reduced[indices]
        ax1.vlines(event_spikes - t, i + 1, i, color='k', lw=0.3)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial #', title='Spike raster plot 1')

    # bottom left: raster 2
    for i, t in enumerate(event_starts2):
        indices = np.bitwise_and(spike_times2_reduced >= t, spike_times2_reduced <= t + max_duration)
        event_spikes = spike_times2_reduced[indices]
        ax3.vlines(event_spikes - t, i + 1, i, color='k', lw=0.3)
    ax3.set(xlabel='Time after event (s)', ylabel='Trial #', title='Spike raster plot 2')

    
    # top right: cumulative sums
    ax2.plot(spike_time_vector, real_fraction1)
    ax2.plot(spike_time_vector, real_fraction2)
    ax2.set(xlabel='Time after event (s)', ylabel='Scaled cumulative spiking density (s)')

    # bottom right: deviation with random jitters
    for i in range(plot_random_samples):
        ax4.plot(random_times[i], random_differences[i], color=[0.7, 0.7, 0.7])
    ax4.plot(spike_time_vector, real_difference)
    ax4.plot(zeta_time, zeta_deviation, 'bx')
    ax4.plot(zeta_time_inv_sign, deviation_inv_sign, 'b*')
    ax4.set(xlabel='Time after event (s)', ylabel='Difference in cumulative density (s)')
    if mean_z_score is not None:
        ax4.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f}), z(Hz)={mean_z_score:.3f} (p={mean_p_value:.3f})')
    else:
        ax4.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f})')

    fig.tight_layout()
    plt.show()

# %% plot_zeta
def plotzeta(spike_times, event_times, zeta_data, rate_data,
             plot_random_samples=50, plot_spike_number=10000):
    """
    Creates figure for ZETA-test analysis

    Syntax:
    plot_zeta(spike_times, event_times, zeta_data, rate_data, plot_random_samples=50, plot_spike_number=10000)

    Parameters
    ----------
    spike_times : 1D array (float)
        spike times (in seconds).
    event_times : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate on/off difference.
    zeta_data : dict
        Output of zetatest.
    rate_data : dict
        Output of zetatest.
    plot_random_samples : int, optional
        Maximum number of random resampling to plot. The default is 50.
    plot_spike_number : int, optional
        Maximum number of spikes to plot. The default is 10000.


    Code by Jorrit Montijn

    Version history:
    1.0 - 07 September 2023 Created by Jorrit Montijn
    """

    # %% check input
    # spike_times must be [S by 1] array
    assert (len(spike_times.shape) == 1 or spike_times.shape[1] == 1) and issubclass(
        spike_times.dtype.type, np.floating), "Input spike_times is not a 1D float np.array with >2 spike times"
    spike_times = np.sort(spike_times.flatten(), axis=0)

    # ensure orientation and assert that event_times is a 1D or N-by-2 array of floats
    assert len(event_times.shape) < 3 and issubclass(
        event_times.dtype.type, np.floating), "Input event_times is not a 1D or 2D float np.array"
    if len(event_times.shape) > 1:
        if event_times.shape[1] < 3:
            pass
        elif event_times.shape[0] < 3:
            event_times = event_times.T
        else:
            raise Exception(
                "Input error: event_times must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        event_times = np.reshape(event_times, (-1, 1))
    # define event starts
    event_starts = event_times[:, 0]

    # unpack zeta_data
    try:
        max_duration = zeta_data['max_duration']
        zeta_score = zeta_data['zeta_score']
        zeta_p = zeta_data['zeta_p_value']
        zeta_deviation = zeta_data['zeta_deviation']
        latency_zeta = zeta_data['latency_zeta']
        deviation_inv_sign = zeta_data['deviation_inv_sign']
        latency_inv_zeta = zeta_data['latency_inv_zeta']
        mean_z_score = zeta_data['ttest_z_score']
        mean_p_value = zeta_data['ttest_p_value']
        spike_time_vector = zeta_data['spike_time_vector']
        real_deviation = zeta_data['real_deviation']
        random_times = zeta_data['random_times']
        random_deviations = zeta_data['random_deviations']

    except KeyError as e:
        raise Exception(
            f"plot_zeta error: information is missing from zeta_data dictionary: {e}")

    # unpack rate_data
    try:
        rate_vector = rate_data['rate_vector']
        rate_timestamps = rate_data['timestamps']
    except KeyError as e:
        raise Exception(
            f"plot_zeta error: information is missing from rate_data dictionary: {e}")

    # %% plot
    # Plot maximally 50 traces (or however many are requested)
    plot_random_samples = np.min([len(random_times), plot_random_samples])

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=300)

    # top left: raster
    if spike_times.size > plot_spike_number:
        spike_times_reduced = spike_times[np.round(np.linspace(0, spike_times.size-1, plot_spike_number)).astype(int)]
    else:
        spike_times_reduced = spike_times

    for i, t in enumerate(event_starts):
        indices = np.bitwise_and(spike_times_reduced >= t, spike_times_reduced <= t + max_duration)
        event_spikes = spike_times_reduced[indices]
        ax1.vlines(event_spikes - t, i + 1, i, color='k', lw=0.3)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial #', title='Spike raster plot')

    # top right: psth
    peth, binned_spikes = calculate_peths(spike_times, np.ones(spike_times.shape), [1],
                                          event_starts, pre_time=0, post_time=max_duration,
                                          bin_size=max_duration/25, smoothing=0)
    ax2.errorbar(peth['tscale'], peth['means'][0, :], yerr=peth['sems'])
    ax2.set(xlabel='Time after event (s)', ylabel='spks/s',
            title='Mean spiking over trials')

    # bottom left: deviation with random jitters
    for i in range(plot_random_samples):
        ax3.plot(random_times[i], random_deviations[i], color=[0.7, 0.7, 0.7])
    ax3.plot(spike_time_vector, real_deviation)
    ax3.plot(latency_zeta, zeta_deviation, 'bx')
    ax3.plot(latency_inv_zeta, deviation_inv_sign, 'b*')
    ax3.set(xlabel='Time after event (s)', ylabel='Spiking density anomaly (s)')
    if mean_z_score is not None:
        ax3.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f}), z(Hz)={mean_z_score:.3f} (p={mean_p_value:.3f})')
    else:
        ax3.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f})')

    # bottom right: ifr
    if len(rate_timestamps) > 1000:
        subset_indices = np.round(np.linspace(0, len(rate_timestamps)-1, 1000)).astype(int)
        ax4.plot(rate_timestamps[subset_indices], rate_vector[subset_indices])
    else:
        ax4.plot(rate_timestamps, rate_vector)
    ax4.set(xlabel='Time after event (s)', ylabel='Spiking rate (Hz)', title='IFR (instantaneous firing rate)')

    fig.tight_layout()
    plt.show()

# %% plot_ts_zeta


def plottszeta(time, data, event_times, zeta_data, plot_random_samples=50):
    """
    Parameters
    ----------
    time [N x 1]: 1D array (float)
        timestamps in seconds corresponding to entries in data.
    data [N x 1] : 1D array (float)
        values (e.g., dF/F0 activity).
    event_times : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate on/off difference.
    zeta_data : dict
        Output of zetatstest.
    plot_random_samples : int, optional
        Maximum number of random resampling to plot. The default is 50.

    """

    # %% prep data and assert inputs are correct

    # time and data must be [N by 1] arrays
    assert len(time.shape) == len(
        data.shape) and time.shape == data.shape, "time and data have different shapes"
    assert (len(time.shape) == 1 or time.shape[1] == 1) and issubclass(
        time.dtype.type, np.floating), "Input time is not a 1D float np.array with >2 spike times"
    time = time.flatten()
    data = data.flatten()
    reorder_indices = np.argsort(time, axis=0)
    time = time[reorder_indices]
    data = data[reorder_indices]

    # ensure orientation and assert that event_times is a 1D or N-by-2 array of floats
    assert len(event_times.shape) < 3 and issubclass(
        event_times.dtype.type, np.floating), "Input event_times is not a 1D or 2D float np.array"
    if len(event_times.shape) > 1:
        if event_times.shape[1] < 3:
            pass
        elif event_times.shape[0] < 3:
            event_times = event_times.T
        else:
            raise Exception(
                "Input error: event_times must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        event_times = np.reshape(event_times, (-1, 1))
    # define event starts
    event_starts = event_times[:, 0]

    # check if number of events and values is sufficient
    if time.size < 3 or event_starts.size < 3:
        if time.size < 3:
            message1 = f"Number of entries in time-series ({time.size}) is too few; "
        else:
            message1 = ""
        if event_starts.size < 3:
            message2 = f"Number of events ({event_starts.size}) is too few; "
        else:
            message2 = ""
            logging.warning("plot_ts_zeta: " + message1 + message2 + "defaulting to p=1.0")

    # unpack zeta_data
    try:
        # ZETA significance
        zeta_p = zeta_data['zeta_p_value']
        zeta_score = zeta_data['zeta_score']
        mean_z_score = zeta_data['ttest_z_score']
        mean_p_value = zeta_data['ttest_p_value']
        zeta_deviation = zeta_data['zeta_deviation']
        latency_zeta = zeta_data['latency_zeta']
        deviation_inv_sign = zeta_data['deviation_inv_sign']
        latency_inv_zeta = zeta_data['latency_inv_zeta']
        real_time = zeta_data['real_time']
        real_deviation = zeta_data['real_deviation']
        real_fraction = zeta_data['real_fraction']
        real_fraction_linear = zeta_data['real_fraction_linear']
        random_times = zeta_data['random_times']
        random_deviations = zeta_data['random_deviations']
        max_duration = zeta_data['max_duration']

    except KeyError as e:
        raise Exception(
            f"plot_ts_zeta error: information is missing from zeta_data dictionary: {e}")

    # %% calculate heat map
    # sampling interval
    sampling_interval = np.median(np.diff(time))
    reference_time_vector = np.arange(sampling_interval / 2, max_duration, sampling_interval)
    reference_time_vector, activity_matrix = get_interpolated_time_series(time, data, event_starts, reference_time_vector)

    # %% plot
    # Plot maximally 50 traces (or however many are requested)
    plot_random_samples = np.min([len(random_times), plot_random_samples])

    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=300)

    # top left: heat map
    x_start = reference_time_vector[1]
    x_end = reference_time_vector[-1]
    x_width = x_end - x_start
    num_trials = activity_matrix.shape[0]
    y_height = num_trials - 1
    img_plot = ax1.imshow(activity_matrix, interpolation='none', extent=[x_start, x_end, 1, num_trials])
    ax1.set_aspect((x_width / y_height) / 2)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial number',
            title='Color indicates data value')
    fig.colorbar(img_plot, ax=ax1)

    # top right: mean +/- SEM
    mean_activity = np.mean(activity_matrix, axis=0)
    sem_activity = np.std(activity_matrix, axis=0) / np.sqrt(num_trials)
    ax2.errorbar(reference_time_vector, mean_activity, yerr=sem_activity)
    ax2.set(xlabel='Time after event (s)', ylabel='Data value',
            title='Mean +/- SEM over trials')

    # bottom left: cumulative plots
    ax3.plot(real_time, real_fraction)
    ax3.plot(real_time, real_fraction_linear, color=[0.7, 0.7, 0.7])
    ax3.set(xlabel='Time after event (s)', ylabel='Cumulative data', title='Time-series zeta-test')

    # bottom right: deviation with random jitters
    for i in range(plot_random_samples):
        ax4.plot(random_times[i], random_deviations[i], color=[0.7, 0.7, 0.7])
    ax4.plot(real_time, real_deviation)
    ax4.plot(latency_zeta, zeta_deviation, 'bx')
    ax4.plot(latency_inv_zeta, deviation_inv_sign, 'b*')
    ax4.set(xlabel='Time after event (s)', ylabel='Data amplitude anomaly')
    if mean_z_score is not None:
        ax4.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f}), z(Hz)={mean_z_score:.3f} (p={mean_p_value:.3f})')
    else:
        ax4.set(title=f'ZETA={zeta_score:.3f} (p={zeta_p:.3f})')

    fig.tight_layout()
    plt.show()

def calculate_peths(
        spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2,
        post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
    """
    Calcluate peri-event time histograms; return means and standard deviations
    for each time point across specified clusters
    
    Code modified from Brainbox library of the International Brain Laboratory
    https://github.com/int-brain-lab/ibllib/blob/master/brainbox/singlecell.py
    
    :param spike_times: spike times (in seconds)
    :type spike_times: array-like
    :param spike_clusters: cluster ids corresponding to each event in `spikes`
    :type spike_clusters: array-like
    :param cluster_ids: subset of cluster ids for calculating peths
    :type cluster_ids: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param bin_size: width of time windows (in seconds) to bin spikes
    :type bin_size: float
    :param smoothing: standard deviation (in seconds) of Gaussian kernel for
        smoothing peths; use `smoothing=0` to skip smoothing
    :type smoothing: float
    :param return_fr: `True` to return (estimated) firing rate, `False` to return spike counts
    :type return_fr: bool
    :return: peths, binned_spikes
    :rtype: peths: Bunch({'mean': peth_means, 'std': peth_stds, 'tscale': ts, 'cscale': ids})
    :rtype: binned_spikes: np.array (n_align_times, n_clusters, n_bins)
    """

    # initialize containers
    offset_bins = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    pre_bins = int(np.ceil(pre_time / bin_size)) + offset_bins
    post_bins = int(np.ceil(post_time / bin_size)) + offset_bins
    total_bins = pre_bins + post_bins
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), total_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = total_bins - 1 if total_bins % 2 == 0 else total_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    unique_cluster_ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    indices = np.bitwise_and(spike_times >= np.min(align_times) - (pre_bins + 1) * bin_size,
                          spike_times <= np.max(align_times) + (post_bins + 1) * bin_size)
    indices = np.bitwise_and(indices, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[indices]
    spike_clusters = spike_clusters[indices]

    # compute floating tscale
    tscale = np.arange(-pre_bins, post_bins + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        bin_edges = tscale + t_0
        # filter spikes
        indices = np.bitwise_and(spike_times >= bin_edges[0], spike_times <= bin_edges[-1])
        trial_spikes = spike_times[indices]
        trial_clusters = spike_clusters[indices]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        bin_indices = (np.floor((trial_spikes - np.min(bin_edges)) / bin_size)).astype(np.int64)
        unique_trial_clusters, cluster_indices = np.unique(trial_clusters, return_inverse=True)
        num_x_bins, num_y_clusters = [bin_edges.size, unique_trial_clusters.size]
        flat_indices = np.ravel_multi_index(np.c_[cluster_indices, bin_indices].transpose(), dims=(num_y_clusters, num_x_bins))
        binned_counts = np.bincount(flat_indices, minlength=num_x_bins * num_y_clusters, weights=None).reshape(num_y_clusters, num_x_bins)

        # store (tscale represent bin edges, so there are one fewer bins)
        binned_spike_indices = np.isin(unique_cluster_ids, unique_trial_clusters)
        binned_spikes[i, binned_spike_indices, :] = binned_counts[:, :-1]

        # smooth
        if smoothing > 0:
            indices = np.where(binned_spike_indices)[0]
            for j in range(binned_counts.shape[0]):
                binned_spikes_conv[i, indices[j], :] = convolve(
                    binned_counts[j, :], window, mode='same', method='auto')[:-1]

    # average
    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)
    if return_fr:
        binned_spikes_ /= bin_size

    peth_means = np.mean(binned_spikes_, axis=0)
    peth_stds = np.std(binned_spikes_, axis=0)
    peth_sems = np.std(binned_spikes_, axis=0) / np.sqrt(align_times.shape[0])

    if smoothing > 0:
        peth_means = peth_means[:, offset_bins:-offset_bins]
        peth_stds = peth_stds[:, offset_bins:-offset_bins]
        binned_spikes = binned_spikes[:, :, offset_bins:-offset_bins]
        tscale = tscale[offset_bins:-offset_bins]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = dict({'means': peth_means, 'stds': peth_stds, 'sems': peth_sems,
                  'tscale': tscale, 'cscale': unique_cluster_ids})
    return peths, binned_spikes
