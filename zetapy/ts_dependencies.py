# -*- coding: utf-8 -*-
import numpy as np
import logging
from zetapy.dependencies import get_gumbel_p_value, my_randint

# %%

def calc_ts_zeta_two(timestamps1, data1, event_times1, timestamps2, data2, event_times2, super_resolution_factor,
                     max_duration, resampling_number, direct_quantile):
    """
   Calculates neuronal responsiveness index zeta
    zeta_data = calc_ts_zeta_two(timestamps1, data1, event_times1, timestamps2, data2, event_times2, 
                          super_resolution_factor, max_duration, resampling_number, direct_quantile)
    zeta_data has entries:
        reference_time, real_difference, real_fraction1, real_fraction2, random_differences, zeta_p_value, zeta_score, zeta_index
    """

    # %% pre-allocate output
    reference_time = None
    real_difference = None
    real_fraction1 = None
    real_fraction2 = None
    random_differences = None
    zeta_p_value = 1.0
    zeta_score = 0.0
    zeta_index = None
    trace_per_trial1 = None
    trace_per_trial2 = None
   
    zeta_data = dict()
    zeta_data['reference_time'] = reference_time
    zeta_data['real_difference'] = real_difference
    zeta_data['real_fraction1'] = real_fraction1
    zeta_data['real_fraction2'] = real_fraction2
    zeta_data['random_differences'] = random_differences
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index
    zeta_data['trace_per_trial1'] = trace_per_trial1
    zeta_data['trace_per_trial2'] = trace_per_trial2

    # %% reduce data
    # assert that event_times is a 1D array of floats
    assert len(event_times1.shape) < 3 and len(event_times2.shape) < 3 \
        and issubclass(event_times1.dtype.type, np.floating) and issubclass(event_times2.dtype.type, np.floating), \
        "Input event_times1 or event_times2 is not a 1D or 2D float np.array"

    # ensure orientation event_times1
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

    # ensure orientation event_times2
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

    # reduce data 1
    event_starts1 = event_times1[:, 0]
    pre_use = -max_duration
    post_use = max_duration * 2
    start_time1 = np.min(event_starts1) + pre_use * 2
    stop_time1 = np.max(event_starts1) + post_use * 2

    keep_entries1 = np.logical_and(timestamps1 >= start_time1, timestamps1 <= stop_time1)
    timestamps1 = timestamps1[keep_entries1]
    data1 = data1[keep_entries1]

    if timestamps1.size < 3:
        logging.warning(
            "calc_ts_zeta_two:timestamps1: too few entries around events to calculate zeta")
        return zeta_data

    # reduce data 2
    event_starts2 = event_times2[:, 0]
    pre_use = -max_duration
    post_use = max_duration * 2
    start_time2 = np.min(event_starts2) + pre_use * 2
    stop_time2 = np.max(event_starts2) + post_use * 2

    keep_entries2 = np.logical_and(timestamps2 >= start_time2, timestamps2 <= stop_time2)
    timestamps2 = timestamps2[keep_entries2]
    data2 = data2[keep_entries2]

    if timestamps2.size < 3:
        logging.warning(
            "calc_ts_zeta_two:timestamps2: too few entries around events to calculate zeta")
        return zeta_data

    # %% rescale
    min_val = min(np.min(data1), np.min(data2))
    max_val = max(np.max(data1), np.max(data2))
    data_range = (max_val - min_val)
    if data_range == 0:
        data_range = 1
        logging.warning(
            "calc_ts_zeta_two:ZeroVar: Input data has zero variance")

    trace_activity1 = np.divide(data1 - min_val, data_range)
    trace_activity2 = np.divide(data2 - min_val, data_range)

    # %% build reference time and matrices
    # time
    reference_t1 = get_ts_ref_t(timestamps1, event_starts1, max_duration)
    reference_t2 = get_ts_ref_t(timestamps2, event_starts2, max_duration)
    # set tol
    sample_interval = (np.median(np.diff(reference_t1)) + np.median(np.diff(reference_t2))) / 2.0
    tolerance = sample_interval / super_resolution_factor
    reference_time = uniquetol(np.concatenate((reference_t1, reference_t2), axis=0), tolerance)
    num_time_points = len(reference_time)

    # matrices
    time1, trace_per_trial1 = get_interpolated_time_series(timestamps1, trace_activity1, event_starts1, reference_time)
    time2, trace_per_trial2 = get_interpolated_time_series(timestamps2, trace_activity2, event_starts2, reference_time)

    # %% get trial responses
    real_difference, real_fraction1, real_fraction2 = get_timeseries_offset_two(trace_per_trial1, trace_per_trial2)
    zeta_index = np.argmax(np.abs(real_difference))
    max_deviation = np.abs(real_difference[zeta_index])

    # repeat procedure, but swap trials randomly in each resampling
    random_differences = np.empty((resampling_number, num_time_points))
    random_differences.fill(np.nan)
    max_random_deviations = np.empty((resampling_number, 1))
    max_random_deviations.fill(np.nan)

    aggregate_trials = np.concatenate((trace_per_trial1, trace_per_trial2), axis=0)
    num_trials1 = trace_per_trial1.shape[0]
    num_trials2 = trace_per_trial2.shape[0]
    total_trials = num_trials1 + num_trials2

    # %% run resamplings (Optimized)
    for resampling_idx in range(resampling_number):
        # Randomly sample trial indices for both groups
        use_random1 = my_randint(total_trials, size=num_trials1)
        use_random2 = my_randint(total_trials, size=num_trials2)

        # Calculate the mean of each group and their difference
        # Using axis=0 directly avoids the .T transposition inside get_timeseries_offset_two
        mean_diff = np.nanmean(aggregate_trials[use_random1, :], axis=0) - \
                    np.nanmean(aggregate_trials[use_random2, :], axis=0)

        # Calculate the cumulative difference profile
        # cumsum(a) - cumsum(b) is mathematically equivalent to cumsum(a - b)
        random_diff = np.cumsum(mean_diff)
        
        # Mean-subtract the deviation profile
        random_diff -= np.mean(random_diff)

        # Store directly in pre-allocated arrays
        random_differences[resampling_idx, :] = random_diff
        add_value = np.max(np.abs(random_diff))
        
        # Fallback for null distributions
        max_random_deviations[resampling_idx] = add_value if (add_value and add_value != 0) else max_deviation

    # %% calculate significance
    zeta_p_value, zeta_score = get_gumbel_p_value(max_deviation, max_random_deviations, direct_quantile)
    
    # %% assign output
    zeta_data['reference_time'] = reference_time
    zeta_data['real_difference'] = real_difference
    zeta_data['real_fraction1'] = real_fraction1
    zeta_data['real_fraction2'] = real_fraction2
    zeta_data['random_differences'] = random_differences
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index
    zeta_data['trace_per_trial1'] = trace_per_trial1
    zeta_data['trace_per_trial2'] = trace_per_trial2
    
    return zeta_data

# %% get_timeseries_offset_two


def get_timeseries_offset_two(trace_per_trial1, trace_per_trial2):
    """
    Calculates the cumulative difference profile between two time-series conditions.

    Parameters
    ----------
    trace_per_trial1 : ndarray
        Trial-by-time matrix for condition 1.
    trace_per_trial2 : ndarray
        Trial-by-time matrix for condition 2.

    Returns
    -------
    this_difference : ndarray
        The mean-subtracted cumulative difference between the two conditions.
    this_fraction1, this_fraction2 : ndarray
        The cumulative sum of the mean traces for condition 1 and 2, respectively.
    """


    # cond1 goes to sum(v_mu1); cond2 goes to sum(v_mu2)
    mean_trace1 = np.nanmean(trace_per_trial1.T, axis=1)
    mean_trace2 = np.nanmean(trace_per_trial2.T, axis=1)

    # get real cumsums
    this_fraction1 = np.cumsum(mean_trace1)
    this_fraction2 = np.cumsum(mean_trace2)

    # take difference
    deviation = this_fraction1 - this_fraction2

    # mean-subtract
    this_difference = deviation - np.mean(deviation)

    # return
    return this_difference, this_fraction1, this_fraction2

# %%


def calc_ts_zeta_one(timestamps, data, event_times, max_duration, resampling_number, direct_quantile, jitter_size,
                     stitch_enabled):
    """
    Calculates the ZETA-test for a single time-series compared to a jittered null distribution.

    Parameters
    ----------
    timestamps : ndarray
        1D array of timestamps for the continuous data.
    data : ndarray
        1D array of values corresponding to the timestamps.
    event_times : ndarray
        1D or 2D array of event onset times.
    max_duration : float
        The duration of the window to analyze after each event.
    resampling_number : int
        Number of jittered resamplings to perform for the null distribution.
    direct_quantile : bool
        Whether to calculate the p-value directly from the distribution or via Gumbel fit.
    jitter_size : float
        The multiplier for max_duration to determine the jitter range.
    stitch_enabled : bool
        Whether to stitch stimulus periods together to create a pseudo-continuous trace.

    Returns
    -------
    zeta_data : dict
        Dictionary containing the ZETA score, p-value, and intermediate calculation results
        (real_deviation, real_time, random_deviations, etc.).
    """
    
    # pre-allocate output
    real_time = None
    real_deviation = None
    real_fraction = None
    real_fraction_linear = None
    random_times = None
    random_deviations = None
    zeta_p_value = 1.0
    zeta_score = 0.0
    zeta_index = None

    zeta_data = dict()
    zeta_data['real_time'] = real_time
    zeta_data['real_deviation'] = real_deviation
    zeta_data['real_fraction'] = real_fraction
    zeta_data['real_fraction_linear'] = real_fraction_linear
    zeta_data['random_times'] = random_times
    zeta_data['random_deviations'] = random_deviations
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index

    # %% reduce data
    # ensure orientation and assert that event_times is a 1D array of floats
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

    pre_use = -max_duration * jitter_size
    post_use = max_duration * (jitter_size + 1)
    start_time = np.min(event_starts) + pre_use * 2
    stop_time = np.max(event_starts) + post_use * 2

    keep_entries = np.logical_and(timestamps >= start_time, timestamps <= stop_time)
    timestamps = timestamps[keep_entries]
    data = data[keep_entries]

    min_val = np.min(data)
    max_val = np.max(data)
    data_range = (max_val - min_val)
    if data_range == 0:
        data_range = 1
        logging.warning(
            "calc_ts_zeta_one:ZeroVar: Input data has zero variance")
    data = (data - min_val) / data_range

    # %% build pseudo data, stitching stimulus periods
    if stitch_enabled:
        pseudo_timestamps, pseudo_data, pseudo_event_starts = get_pseudo_time_series(
            timestamps, data, event_starts, max_duration)
    else:
        pseudo_timestamps = timestamps
        pseudo_data = data
        pseudo_event_starts = event_starts

    pseudo_data = pseudo_data - np.min(pseudo_data)

    if timestamps.size < 3:
        logging.warning(
            "calc_ts_zeta_one:pseudo_timestamps: too few entries around events to calculate zeta")
        return zeta_data

    # %% run normal
    # get data
    super_resolution_factor_or_ref_t = 100
    real_deviation, real_fraction, real_fraction_linear, real_time = get_timeseries_offset_one(
        pseudo_timestamps, pseudo_data, pseudo_event_starts, max_duration, super_resolution_factor_or_ref_t)

    if real_deviation.size < 3:
        logging.warning(
            "calc_ts_zeta_one:real_deviation: too few spikes around events to calculate zeta")
        return zeta_data

    real_deviation = real_deviation - np.mean(real_deviation)
    zeta_index = np.argmax(np.abs(real_deviation))
    max_deviation = np.abs(real_deviation[zeta_index])

    # %% create random jitters
    # run pre-set number of iterations
    random_times = []
    random_deviations = []
    max_random_deviations = np.empty((resampling_number, 1))
    max_random_deviations.fill(np.nan)

    event_starts_only = np.reshape(pseudo_event_starts, (-1, 1))
    num_trials = event_starts_only.size
    jitter_per_trial = jitter_size * max_duration * ((np.random.rand(resampling_number, num_trials).T - 0.5) * 2)

    # %% this part is only to check if matlab and python give the same exact results
    test_mode = False
    if test_mode:
        from scipy.io import loadmat
        print('Loading deterministic jitter data for comparison with matlab')
        logging.warning(
            "calc_ts_zeta_one:debugMode: set test_mode to False to suppress this warning")
        data_load = loadmat('matJitterPerTrialTsZeta.mat')
        jitter_per_trial = data_load['matJitterPerTrial']

        # reset rng
        np.random.seed(1)

    # %% run resamplings
    for resampling_idx in range(resampling_number):
        # Calculate jittered stimulus onset times for this iteration
        stim_use_on_time = event_starts_only[:, 0] + jitter_per_trial[:, resampling_idx]

        # Directly use interpolation to avoid the overhead of get_timeseries_offset_one
        _, trace_per_trial = get_interpolated_time_series(
            pseudo_timestamps, pseudo_data, stim_use_on_time, real_time
        )

        # Calculate mean across trials and normalize to a cumulative fraction
        mean_trace = np.nanmean(trace_per_trial, axis=0)
        sum_mean = np.sum(mean_trace)
        if sum_mean == 0:
            sum_mean = 1.0

        this_fraction = np.cumsum(mean_trace) / sum_mean
        
        # Deviation from the linear ramp (real_fraction_linear is constant for all resamplings)
        deviation = this_fraction - real_fraction_linear
        deviation -= np.mean(deviation)

        random_times.append(real_time)
        random_deviations.append(deviation)
        max_random_deviations[resampling_idx] = np.max(np.abs(deviation))

    # %% calculate significance
    zeta_p_value, zeta_score = get_gumbel_p_value(max_deviation, max_random_deviations, direct_quantile)

    # %% assign output
    zeta_data['real_time'] = real_time
    zeta_data['real_deviation'] = real_deviation
    zeta_data['real_fraction'] = real_fraction
    zeta_data['real_fraction_linear'] = real_fraction_linear
    zeta_data['random_times'] = random_times
    zeta_data['random_deviations'] = random_deviations
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index

    return zeta_data

# %% get_pseudo_time_series


def get_pseudo_time_series(timestamps, data, event_times, window_duration):
    """
    Stitches together segments of a time-series around events to create a
    pseudo-continuous trace, removing inter-stimulus intervals.

    Parameters
    ----------
    timestamps : ndarray
        1D array of timestamps for the continuous data.
    data : ndarray
        1D array of values corresponding to the timestamps.
    event_times : ndarray
        1D array of event onset times.
    window_duration : float
        The duration of the window to extract after each event.

    Returns
    -------
    pseudo_timestamps : ndarray
        Stitched timestamps starting from 0.
    pseudo_data : ndarray
        Stitched data values corresponding to pseudo_timestamps.
    pseudo_event_times : list
        The new onset times of the events within the stitched trace.
    """

    # %% prep
    # ensure sorting and alignment
    timestamps = np.squeeze(np.vstack(timestamps))
    data = np.squeeze(np.vstack(data))
    reorder_indices = np.argsort(timestamps, axis=0)
    timestamps = timestamps[reorder_indices]
    data = data[reorder_indices]
    event_times = np.squeeze(np.sort(np.vstack(event_times), axis=0))
    num_samples = timestamps.size
    num_trials = event_times.size
    median_duration = np.median(np.diff(timestamps, axis=0))
    pseudo_time_list = []
    pseudo_data_list = []
    pseudo_event_times = []
    start_next_at_time = 0
    last_used_sample = -1

    # %% run
    for trial_idx, event_time in enumerate(event_times):
        start_sample = np.searchsorted(timestamps, event_time, side='right') - 1
        end_sample_idx = np.searchsorted(timestamps, event_time + window_duration, side='right')
        if end_sample_idx >= len(timestamps):
            end_sample = start_sample
        else:
            end_sample = end_sample_idx
        eligible_samples = np.arange(start_sample, end_sample + 1)
        use_samples_mask = np.logical_and(eligible_samples >= 0, eligible_samples < num_samples)
        use_samples = eligible_samples[use_samples_mask]

		# check if beginning or end
        if trial_idx == 0 and use_samples.size > 0:
            use_samples = np.arange(0, use_samples[-1] + 1)
        if trial_idx == (num_trials - 1) and use_samples.size > 0:
            use_samples = np.arange(use_samples[0], num_samples)

        use_timestamps = timestamps[use_samples]
        overlap_mask = (use_samples <= last_used_sample)
        if np.any(overlap_mask):
            use_samples = use_samples[np.logical_not(overlap_mask)]
            use_timestamps = timestamps[use_samples]
            
        # make local pseudo event time
        if use_samples.size == 0:
            local_pseudo_timestamps = None
            local_pseudo_data = None
            pseudo_event_time = event_time - timestamps[last_used_sample] + start_next_at_time
        else:
            last_used_sample = use_samples[-1]
            local_pseudo_data = data[use_samples]
            local_pseudo_timestamps = use_timestamps - use_timestamps[0] + start_next_at_time
            pseudo_event_time = event_time - use_timestamps[0] + start_next_at_time

            if len(timestamps) > (last_used_sample + 1):
                step_end = timestamps[last_used_sample + 1] - timestamps[last_used_sample]
            else:
                step_end = median_duration

            start_next_at_time = local_pseudo_timestamps[-1] + step_end

        pseudo_time_list.append(local_pseudo_timestamps)
        pseudo_data_list.append(local_pseudo_data)
        pseudo_event_times.append(pseudo_event_time)

    # %% filter out None values and recombine into 1D vector
    pseudo_timestamps = np.concatenate([t for t in pseudo_time_list if t is not None]).ravel()
    pseudo_data = np.concatenate([d for d in pseudo_data_list if d is not None]).ravel()
    return pseudo_timestamps, pseudo_data, pseudo_event_times


# %% get_timeseries_offset_one
def get_timeseries_offset_one(timestamps, data, event_start_times, max_duration, super_resolution_factor_or_ref_t=100):
    """
    Calculates the deviation of the mean time-series response from a linear accumulation.

    Parameters
    ----------
    timestamps : ndarray
        1D array of timestamps for the data.
    data : ndarray
        1D array of time-series data values.
    event_start_times : ndarray
        1D array of event onset times.
    max_duration : float
        The duration of the window to analyze after each event.
    super_resolution_factor_or_ref_t : int or ndarray, optional
        Factor for time-vector construction or a pre-defined reference time vector. Default is 100.

    Returns
    -------
    deviation, this_fraction, this_fraction_linear, time_vector
    """

    # %% prepare
    if isinstance(super_resolution_factor_or_ref_t, (np.ndarray, list)):
        time_vector = super_resolution_factor_or_ref_t
    else:
        time_vector = get_ts_ref_t(timestamps, event_start_times, max_duration, super_resolution_factor_or_ref_t)

    # build interpolated data
    time_vector, trace_per_trial = get_interpolated_time_series(timestamps, data, event_start_times, time_vector)
    keep_points = np.logical_and(time_vector >= 0, time_vector <= max_duration)
    time_vector = time_vector[keep_points]
    trace_per_trial = trace_per_trial[:, keep_points]
    mean_trace = np.nanmean(trace_per_trial, axis=0)
    this_fraction = np.cumsum(mean_trace) / np.sum(mean_trace)

    # get linear fractions
    this_fraction_linear = np.linspace(np.mean(mean_trace), np.sum(
        mean_trace), len(mean_trace)) / np.sum(mean_trace)

    # assign data
    deviation = this_fraction - this_fraction_linear
    deviation = deviation - np.mean(deviation)

    # %% return
    return deviation, this_fraction, this_fraction_linear, time_vector

# %% get_ts_ref_t


def get_ts_ref_t(timestamps, event_start_times, max_duration, super_resolution_factor=1):
    # pre-allocate
    timestamps = np.ravel(timestamps)
    event_start_times = np.sort(event_start_times)
    num_time_points = len(timestamps) - 1

    # build common timeframe
    reference_t_list = []
    for trial_idx, start_time in enumerate(event_start_times):
        begin_index = np.searchsorted(timestamps, start_time, side='right')
        if begin_index >= len(timestamps):
            start_index = 0
        else:
            start_index = np.max([0, begin_index - 1])

        stop_time = start_time + max_duration
        end_index = np.searchsorted(timestamps, stop_time, side='right')
        if end_index >= len(timestamps):
            stop_index = num_time_points
        else:
            stop_index = np.min([num_time_points, end_index])

        select_samples = np.arange(start_index, stop_index + 1)

        # save data
        reference_t_list.append(timestamps[select_samples] - start_time)

    # %% set tol
    if super_resolution_factor == 1:
        use_entry_index = np.argmax([len(item) for item in reference_t_list])
        reference_t = reference_t_list[use_entry_index].flatten()
        median_diff = np.median(np.diff(reference_t))
        time_vector = np.round(10 * (reference_t / median_diff)) / (10 / median_diff)
    else:
        sample_interval = np.median(np.diff(timestamps, axis=0))
        tolerance = sample_interval / 100
        values = np.sort(np.concatenate(reference_t_list))
        time_vector = uniquetol(values, tolerance)

    # return
    return time_vector

# %% get_interpolated_time_series


def get_interpolated_time_series(timestamps, data, event_start_times, reference_time):
    """
    Interpolates time-series data into a trial-by-time matrix based on event onsets.

    Parameters
    ----------
    timestamps : ndarray
        1D array of timestamps for the continuous data.
    data : ndarray
        1D array of values corresponding to the timestamps.
    event_start_times : ndarray
        1D array of event onset times (e.g., stimulus start).
    reference_time : ndarray
        1D array of relative time points (e.g., 0 to max_duration) to interpolate onto.

    Returns
    -------
    reference_time : ndarray
        The reference time vector used.
    trace_per_trial : ndarray
        A 2D array of shape (num_events, num_reference_time_points) containing interpolated data.
    """

    # ensure 1D arrays
    reference_time = np.ravel(reference_time)
    timestamps = np.ravel(timestamps)
    data = np.ravel(data)
    trace_per_trial = np.zeros((len(event_start_times), len(reference_time)))
    for trial_idx, start_time in enumerate(event_start_times):
        # original times
        begin_index = np.searchsorted(timestamps, start_time + reference_time[0], side='right')
        if begin_index >= len(timestamps):
            raise Exception(
                "get_interpolated_time_series error - no time stamps exist after trial start")

        start_index = np.max([0, begin_index - 1])
        end_index = np.searchsorted(timestamps, start_time + reference_time[-1], side='right')
        if end_index >= len(timestamps):
            stop_index = len(timestamps)
        else:
            stop_index = np.min([len(timestamps), end_index])

        select_samples = np.arange(start_index, stop_index)

        # get data
        use_times = timestamps[select_samples]
        use_data = data[select_samples]

        # interpolate to
        use_interp_times = reference_time + start_time

        # get interpolated data
        trace_per_trial[trial_idx, :] = np.interp(use_interp_times, use_times, use_data)

    # return
    return reference_time, trace_per_trial

# %% uniquetol
def uniquetol(input_array, tolerance):
    """
    Returns unique values from an array within a specified tolerance.

    Parameters
    ----------
    input_array : ndarray
        The array of values to process.
    tolerance : float
        The precision threshold for grouping values.

    Returns
    -------
    unique_values : ndarray
        An array of unique values, rounded to the nearest multiple of
        the tolerance.
    """
    return (np.unique(np.round(input_array / tolerance).astype(int))) * tolerance
