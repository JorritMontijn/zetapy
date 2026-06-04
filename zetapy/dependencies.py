# -*- coding: utf-8 -*-
import numpy as np
import logging
from scipy import stats
from math import pi, sqrt, exp
from collections.abc import Iterable

# %% calc_zeta_two

def calc_zeta_two(spike_times1, event_times1, spike_times2, event_times2, max_duration, resampling_number, direct_quantile):
    """
    Calculates the two-sample ZETA-test to compare neuronal responsiveness between two conditions.

    Parameters
    ----------
    spike_times1 : 1D float np.array
        Vector of spike times for condition 1 (seconds).
    event_times1 : 1D or 2D float np.array
        Vector of event start times for condition 1 (seconds).
    spike_times2 : 1D float np.array
        Vector of spike times for condition 2 (seconds).
    event_times2 : 1D or 2D float np.array
        Vector of event start times for condition 2 (seconds).
    max_duration : float
        Duration of the trial in seconds.
    resampling_number : int
        Number of bootstrap iterations for the null distribution.
    direct_quantile : bool
        Whether to use empirical quantiles (True) or Gumbel distribution (False) for p-value calculation.

    Returns
    -------
    dict
        Dictionary containing ZETA-test results, including 'zeta_p_value' and 'zeta_score'.
    """

    # %% pre-allocate output
    spike_time_vector = None
    real_difference = None
    real_fraction1 = None
    real_fraction2 = None
    random_times = None
    random_differences = None
    zeta_p_value = 1.0
    zeta_score = 0.0
    zeta_index = None

    fast_interp = False  # Not implemented yet in python

    zeta_data = dict()
    zeta_data['spike_time_vector'] = spike_time_vector
    zeta_data['real_difference'] = real_difference
    zeta_data['real_fraction1'] = real_fraction1
    zeta_data['real_fraction2'] = real_fraction2
    zeta_data['random_times'] = random_times
    zeta_data['random_differences'] = random_differences
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index

    # %% ensure input is correct
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

    # %% get spikes per trial
    event_starts1 = event_times1[:, 0]
    event_starts2 = event_times2[:, 0]
    trial_per_spike1, time_per_spike1 = get_spikes_in_trial(spike_times1, event_starts1, max_duration)
    trial_per_spike2, time_per_spike2 = get_spikes_in_trial(spike_times2, event_starts2, max_duration)

    # %% run normal
    # normalize to cumsum(v1)+cumsum(v2) = 1
    # take difference
    # mean-subtract

    # get difference
    spike_time_vector, real_difference, real_fraction1, this_spike_times1, real_fraction2, this_spike_times2 = \
        get_temporal_offset_two(time_per_spike1, time_per_spike2, max_duration)

    if len(real_difference) < 2:
        return zeta_data

    zeta_index = np.argmax(np.abs(real_difference))
    max_deviation = np.abs(real_difference[zeta_index])

    # repeat procedure, but swap trials randomly in each resampling
    random_times = [None] * resampling_number
    random_differences = [None] * resampling_number
    max_random_deviations = np.empty((resampling_number, 1))
    max_random_deviations.fill(np.nan)

    aggregate_trials = time_per_spike1 + time_per_spike2
    num_trials1 = len(time_per_spike1)
    num_trials2 = len(time_per_spike2)
    total_trials = num_trials1 + num_trials2

    # run bootstraps; try parallel, otherwise run normal loop
    # repeat procedure, but swap trials randomly in each resampling
    for resampling_idx in range(resampling_number):
        # get random subsample
        # if cond1 has 10 trials, and cond2 has 100, then:
        # for shuffle of cond1: take 10 trials from set of 110
        # for shuffle of cond2: take 100 trials from set of 110
        # use_random1 = np.random.randint(total_trials, size=num_trials1)
        # use_random2 = np.random.randint(total_trials, size=num_trials2)

        use_random1 = my_randint(total_trials, size=num_trials1)
        use_random2 = my_randint(total_trials, size=num_trials2)


        time_per_spike1_rand = [aggregate_trials[i] for i in use_random1]
        time_per_spike2_rand = [aggregate_trials[j] for j in use_random2]

        if np.sum([len(xi) for xi in time_per_spike1_rand]) == 0 and np.sum([len(yi) for yi in time_per_spike2_rand]) == 0:
            add_value = None
        else:
            # get difference
            random_t, random_diff, random_frac1, this_spike_times1, random_frac2, this_spike_times2 = \
                get_temporal_offset_two(time_per_spike1_rand, time_per_spike2_rand, max_duration, fast_interp, spike_time_vector)

            # assign data
            random_times[resampling_idx] = random_t
            random_differences[resampling_idx] = random_diff
            add_value = np.max(np.abs(random_diff))

        # assign read-out
        if add_value is None or add_value == 0:
            add_value = max_deviation
        max_random_deviations[resampling_idx] = add_value

    # calculate significance
    zeta_p_value, zeta_score = get_gumbel_p_value(max_deviation, max_random_deviations, direct_quantile)

    zeta_data = dict()
    zeta_data['spike_time_vector'] = spike_time_vector
    zeta_data['real_difference'] = real_difference
    zeta_data['real_fraction1'] = real_fraction1
    zeta_data['real_fraction2'] = real_fraction2
    zeta_data['random_times'] = random_times
    zeta_data['random_differences'] = random_differences
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index
    return zeta_data

# %%
def calc_zeta_one(spike_times, event_times, max_duration, resampling_number, direct_quantile, jitter_size, stitch_enabled):
    """
    Calculates neuronal responsiveness index zeta

    Parameters
    ----------
    spike_times : 1D float np.array
        Vector of spike times in seconds.
    event_times : 1D or 2D float np.array
        Vector of event start times in seconds.
    max_duration : float
        Duration of the trial in seconds.
    resampling_number : int
        Number of bootstrap/jitter iterations.
    direct_quantile : bool
        Whether to use empirical quantiles (True) or Gumbel distribution (False).
    jitter_size : float
        Multiplier for the jitter window.
    stitch_enabled : bool
        Whether to use the stitching method for pseudo-data.
    """

    # %% pre-allocate output
    spike_time_vector = None
    real_deviation = None
    real_fraction = None
    real_fraction_linear = None
    random_times = None
    random_deviations = None
    zeta_p_value = 1.0
    zeta_score = 0.0
    zeta_index = None

    zeta_data = dict()
    zeta_data['spike_time_vector'] = spike_time_vector
    zeta_data['real_deviation'] = real_deviation
    zeta_data['real_fraction'] = real_fraction
    zeta_data['real_fraction_linear'] = real_fraction_linear
    zeta_data['random_times'] = random_times
    zeta_data['random_deviations'] = random_deviations
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index

    # %% reduce spikes
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

    min_pre_event_time = np.min(event_starts) - max_duration * 5 * jitter_size
    start_time = max([spike_times[0], min_pre_event_time])
    stop_time = np.max(event_starts) + max_duration * 5 * jitter_size
    spike_times = spike_times[np.logical_and(spike_times >= start_time, spike_times <= stop_time)]

    if spike_times.size < 3:
        logging.warning(
            "calc_zeta_one:spike_times: too few spikes around events to calculate zeta")
        return zeta_data

    # %% build pseudo data, stitching stimulus periods
    if stitch_enabled:
        pseudo_spike_times, pseudo_event_times = get_pseudo_spike_vectors(spike_times, event_starts, max_duration)
    else:
        pseudo_spike_times = spike_times
        pseudo_event_times = event_starts

    # %% run normal
    # get data
    real_deviation, real_fraction, real_fraction_linear, spike_time_vector = get_temporal_offset_one(
        pseudo_spike_times, pseudo_event_times, max_duration)

    if real_deviation.size < 3:
        logging.warning(
            "calc_zeta_one:real_deviation: too few spikes around events to calculate zeta")
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

    event_starts_only = np.reshape(pseudo_event_times, (-1, 1))
    num_trials = event_starts_only.size
    jitter_per_trial = np.empty((num_trials, resampling_number))
    jitter_per_trial.fill(np.nan)

    # uniform jitters between jitter_size*[-tau, +tau]
    for resampling_idx in range(resampling_number):
        jitter_per_trial[:, resampling_idx] = jitter_size * max_duration * \
            ((np.random.rand(event_starts_only.shape[0]) - 0.5) * 2)

    # %% this part is only to check if matlab and python give the same exact results
    # unfortunately matlab's randperm() and numpy's np.random.permutation give different outputs even with
    # identical seeds and identical random number generators, so I've had to load in a table of random values here...
    test_mode = False
    if test_mode:
        from scipy.io import loadmat
        import tempfile
        import os
        jitter_filename = 'matJitterPerTrial.mat';
        print('Loading deterministic jitter data for comparison with matlab from ' + jitter_filename)
        logging.warning("calc_zeta_one:debugMode: set test_mode to False to suppress this warning")
        data_load = loadmat(jitter_filename)
        jitter_per_trial = data_load['matJitterPerTrial']

        # reset rng
        np.random.seed(1)

    # %% run resamplings (Optimized Sequential Loop)
    for resampling_idx in range(resampling_number):
        # Calculate jittered stimulus onset times for this iteration
        event_starts_use_time = event_starts_only[:, 0] + jitter_per_trial[:, resampling_idx]

        # Calculate temporal offset and deviation. 
        # Note: get_temporal_offset_one already performs mean-subtraction on the deviation.
        res_deviation, _, _, res_times = get_temporal_offset_one(
            pseudo_spike_times, event_starts_use_time, max_duration
        )

        # Direct assignment to pre-allocated objects
        random_times.append(res_times)
        random_deviations.append(res_deviation)
        max_random_deviations[resampling_idx] = np.max(np.abs(res_deviation))

    # %% calculate significance
    zeta_p_value, zeta_score = get_gumbel_p_value(max_deviation, max_random_deviations, direct_quantile)

    # %% assign output
    zeta_data = dict()
    zeta_data['spike_time_vector'] = spike_time_vector
    zeta_data['real_deviation'] = real_deviation
    zeta_data['real_fraction'] = real_fraction
    zeta_data['real_fraction_linear'] = real_fraction_linear
    zeta_data['random_times'] = random_times
    zeta_data['random_deviations'] = random_deviations
    zeta_data['zeta_p_value'] = zeta_p_value
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_index'] = zeta_index
    return zeta_data

# %%


def get_temporal_offset_two(time_per_spike1, time_per_spike2, max_duration, fast_interp=False, spike_time_vector=None):
    """
    Calculates the temporal offset and difference in spike fractions between two conditions.

    Parameters
    ----------
    time_per_spike1 : list of 1D float np.array
        List where each element contains spike times relative to event start for condition 1.
    time_per_spike2 : list of 1D float np.array
        List where each element contains spike times relative to event start for condition 2.
    max_duration : float
        Duration of the trial in seconds.
    fast_interp : bool, optional
        Whether to use a faster interpolation method (not yet implemented). Default is False.
    spike_time_vector : 1D float np.array, optional
        Pre-defined vector of spike times to interpolate onto. If None, it is generated.

    Returns
    -------
    tuple
        (spike_time_vector, this_difference, this_fraction1, this_spike_times1, this_fraction2, this_spike_times2)
        - spike_time_vector: The time points used for the calculation.
        - this_difference: The mean-subtracted difference between the two cumulative fractions.
        - this_fraction1, this_fraction2: The cumulative spike fractions for each condition.
        - this_spike_times1, this_spike_times2: The unique, jittered spike times for each condition.
    """

    # introduce minimum jitter to identical spikes
    spikes1 = np.concatenate(time_per_spike1)
    spikes2 = np.concatenate(time_per_spike2)
    
    this_spike_times1 = get_unique_spikes(np.sort(spikes1))
    this_spike_times2 = get_unique_spikes(np.sort(spikes2))

    # ref time
    if spike_time_vector is None:
        spike_time_vector = np.sort(np.concatenate((
            np.zeros(1), this_spike_times1, this_spike_times2, np.array([max_duration])), axis=0))

    # cond1 goes to S1_n/T1_n; cond2 goes to S2_n/T2_n
    num_spikes1 = len(this_spike_times1)
    num_spikes2 = len(this_spike_times2)
    num_trials1 = len(time_per_spike1)
    num_trials2 = len(time_per_spike2)

    # spike fraction #1
    unique_spike_fractions1 = np.linspace(1, this_spike_times1.size, this_spike_times1.size)/num_trials1
    spikes1 = np.concatenate((np.zeros(1), this_spike_times1, np.array([max_duration])), axis=0)
    fractions1 = np.concatenate((np.zeros(1), unique_spike_fractions1, np.array([num_spikes1/num_trials1])), axis=0)
    this_fraction1 = np.interp(spike_time_vector, spikes1, fractions1, 1/num_trials1, num_spikes1/num_trials1)

    # spike fraction #2
    unique_spike_fractions2 = np.linspace(1, this_spike_times2.size, this_spike_times2.size)/num_trials2
    spikes2 = np.concatenate((np.zeros(1), this_spike_times2, np.array([max_duration])), axis=0)
    fractions2 = np.concatenate((np.zeros(1), unique_spike_fractions2, np.array([num_spikes2/num_trials2])), axis=0)
    this_fraction2 = np.interp(spike_time_vector, spikes2, fractions2, 1/num_trials2, num_spikes2/num_trials2)

    # take difference
    deviation = this_fraction1 - this_fraction2

    # mean-subtract?
    this_difference = deviation - np.mean(deviation)

    return spike_time_vector, this_difference, this_fraction1, this_spike_times1, this_fraction2, this_spike_times2

# %%


def get_spikes_in_trial(spikes, trial_starts, max_duration):
    """
    get_spikes_in_trial Retrieves spiking times per trial
    syntax: trial_per_spike,time_per_spike = get_spikes_in_trial(spikes,trial_starts,max_duration)
    input:
        - spikes; spike times (s)
        - trial_starts: trial start times (s)
        - max_duration: trial duration (s)
    returns:
        - trial_per_spike
        - time_per_spike
    """

    # loop
    num_trials = len(trial_starts)
    trial_per_spike = []
    time_per_spike = []
    for trial_idx, start_time in enumerate(trial_starts):
        # get spikes
        these_spikes = spikes[np.logical_and(spikes >= start_time, spikes < (start_time + max_duration))] - start_time

        # assign
        trial_per_spike.append(trial_idx * np.ones(len(these_spikes)))
        time_per_spike.append(these_spikes)

    return trial_per_spike, time_per_spike

# %%


def get_gumbel_p_value(max_deviation, max_random_deviations, direct_quantile):
    # %% calculate significance
    # find highest peak and retrieve value
    max_random_deviations = np.sort(np.unique(max_random_deviations), axis=0)
    if not isinstance(max_deviation, Iterable):
        max_deviation = np.array([max_deviation])

    if direct_quantile:
        # calculate statistical significance using empirical quantiles
        # define p-value
        zeta_p_values = np.empty(max_deviation.size)
        zeta_p_values.fill(np.nan)
        for i, d in enumerate(max_deviation):
            if d < np.min(max_random_deviations) or np.isnan(d):
                value = 0
            elif d > np.max(max_random_deviations) or np.isinf(d):
                value = max_random_deviations.size
            else:
                value = np.interp(
                    d, max_random_deviations, np.arange(0, max_random_deviations.size)+1)

            zeta_p_values[i] = 1 - (value/(1+max_random_deviations.size))

        # transform to output z-score
        zeta_scores = -stats.norm.ppf(zeta_p_values/2)
    else:
        # calculate statistical significance using Gumbel distribution
        zeta_p_values, zeta_scores = get_gumbel(
            np.mean(max_random_deviations), np.var(max_random_deviations, ddof=1), max_deviation)  # default ddof for numpy var() is incorrect

    # return
    if zeta_p_values.size == 1:
        zeta_p_values = zeta_p_values[0]
    if zeta_scores.size == 1:
        zeta_scores = zeta_scores[0]
    return zeta_p_values, zeta_scores

# %%


def get_gumbel(mean_distribution, variance_distribution, values):
    """
    Calculates p-values and z-scores using the Gumbel distribution.

    Parameters
    ----------
    mean_distribution : float
        The mean of the distribution of maximum values.
    variance_distribution : float
        The variance of the distribution of maximum values.
    values : 1D float np.array
        The observed maximum values for which to calculate significance.

    Returns
    -------
    tuple
        (p_values, z_scores)

    Sources:
        Baglivo (2005)
        Elfving (1947), https://doi.org/10.1093/biomet/34.1-2.111
        Royston (1982), DOI: 10.2307/2347982
        https://stats.stackexchange.com/questions/394960/variance-of-normal-order-statistics
        https://stats.stackexchange.com/questions/9001/approximate-order-statistics-for-normal-random-variables
        https://en.wikipedia.org/wiki/Extreme_value_theory
        https://en.wikipedia.org/wiki/Gumbel_distribution
    """

    # %% define constants
    # define Euler-Mascheroni constant
    euler_mascheroni = 0.5772156649015328606065120900824  # vpa(eulergamma)

    # %% define Gumbel parameters from mean and variance
    # derive beta parameter from variance
    beta = (sqrt(6)*sqrt(variance_distribution))/(pi)

    # derive mode from mean, beta and E-M constant
    mode = mean_distribution - beta * euler_mascheroni

    # define Gumbel cdf
    def gumbel_cdf_func(x): return np.exp(-np.exp(-((x-mode) / beta)))

    # %% calculate output variables
    # calculate cum dens at X
    gumbel_cdf = gumbel_cdf_func(values)

    # define p-value
    p_values = 1 - gumbel_cdf

    # transform to output z-score
    z_scores = -stats.norm.ppf(np.divide(p_values, 2))

    # approximation for large X
    for i, z_score in enumerate(z_scores):
        if np.isinf(z_score):
            p_values[i] = exp(mode - values[i] / beta)
            z_scores[i] = -stats.norm.ppf(p_values[i]/2)

    # return
    return p_values, z_scores

# %%


def get_temporal_offset_one(spike_times, event_times, max_duration):
    """
    Calculates the temporal offset and deviation from a linear spike distribution for a single condition.

    Parameters
    ----------
    spike_times : 1D float np.array
        Vector of spike times (seconds).
    event_times : 1D float np.array
        Vector of event start times (seconds).
    max_duration : float
        Duration of the trial in seconds.

    Returns
    -------
    tuple
        (this_deviation, this_spike_fractions, this_fraction_linear, this_spike_times)
        - this_deviation: The mean-subtracted difference between empirical and linear cumulative fractions.
        - this_spike_fractions: The empirical cumulative spike fractions.
        - this_fraction_linear: The linear (null hypothesis) cumulative fractions.
        - this_spike_times: The unique, jittered spike times relative to event starts.
    """

    # %% get temp diff vector
    # pre-allocate
    spikes_in_trial = get_spike_t(spike_times, event_times, max_duration)

    # introduce minimum jitter to identical spikes
    this_spike_times = get_unique_spikes(spikes_in_trial)

    # turn into fractions
    this_spike_fractions = np.linspace(
        1/this_spike_times.size, 1, this_spike_times.size)

    # get linear fractions
    this_fraction_linear = this_spike_times/max_duration

    # calc difference
    this_deviation = this_spike_fractions - this_fraction_linear
    this_deviation = this_deviation - np.mean(this_deviation)

    return this_deviation, this_spike_fractions, this_fraction_linear, this_spike_times

# %%

def get_unique_spikes(spike_times):
    """
    Ensures all spike times are unique by adding a microscopic jitter to duplicates.

    Parameters
    ----------
    spike_times : 1D float np.array
        Vector of spike times.

    Returns
    -------
    1D float np.array
        Sorted vector of spike times where no two values are identical within
        machine epsilon.
    """

    # introduce minimum jitter to identical spikes
    spike_times = np.sort(spike_times)
    unique_offset = np.finfo(spike_times.dtype.type).eps
    shift = unique_offset
    duplicates_mask = np.append(False,np.diff(spike_times)<unique_offset)
    while np.any(duplicates_mask):
        not_unique = spike_times[duplicates_mask]
        jitter = np.concatenate( (1+9*np.random.rand(len(not_unique)),-1-9*np.random.rand(len(not_unique))),axis=0)
        jitter = shift * jitter[my_randperm(len(jitter),len(not_unique))]
        spike_times[duplicates_mask] = spike_times[duplicates_mask] + jitter
        spike_times = np.sort(spike_times)
        duplicates_mask = np.append(False,np.diff(spike_times)<unique_offset)
        shift = shift * 2; # to avoid endless loop if jitter is too small
    return spike_times

def my_randperm(n, k):
    # randperm introduced to make results reproducable between python and
    #  MATLAB implementation
    indices = np.argsort(np.random.rand(n))
    return indices[:k]
 

# %%


def get_spike_t(spike_times, event_times, max_duration):
    """
    Aggregates spike times relative to event starts across all trials.

    Parameters
    ----------
    spike_times : 1D float np.array
        Vector of absolute spike times (seconds).
    event_times : 1D float np.array
        Vector of event start times (seconds).
    max_duration : float
        Duration of the trial window (seconds).

    Returns
    -------
    1D float np.array
        Sorted vector of relative spike times, including 0 and max_duration.
    """


    # pre-allocate
    spikes_in_trial = np.empty((spike_times.size*2))
    spikes_in_trial.fill(np.nan)
    index = 0

    # go through trials to build spike time vector
    for start_time in event_times:
        # get times
        stop_time = start_time + max_duration

        # build trial assignment
        temp_spikes = spike_times[np.logical_and(spike_times < stop_time, spike_times > start_time)] - start_time
        temp_spike_number = temp_spikes.size
        assign_index = np.arange(index, index + temp_spike_number)
        if assign_index.shape[0] > 0 and assign_index[-1] >= spikes_in_trial.size:
            spikes_in_trial = np.resize(spikes_in_trial, spikes_in_trial.size*2)
        spikes_in_trial[assign_index] = temp_spikes
        index = index + temp_spike_number

    # remove trailing nan entries
    spikes_in_trial = spikes_in_trial[:index]

    # sort spikes in window and add start/end entries
    spikes_in_trial = np.concatenate((np.zeros(1), np.sort(spikes_in_trial, axis=0, kind='quicksort'),
                                       np.array([max_duration])))

    return spikes_in_trial

# %%


def get_pseudo_spike_vectors(spike_times, event_times, window_duration, discard_edges=False):
    """
    Stitches together spike times from stimulus periods to create a continuous pseudo-timeline.

    Parameters
    ----------
    spike_times : 1D float np.array
        Vector of absolute spike times (seconds).
    event_times : 1D float np.array
        Vector of event start times (seconds).
    window_duration : float
        Duration of the trial window (seconds).
    discard_edges : bool, optional
        Whether to discard spikes occurring before the first event and after the last event's
        window. Default is False.

    Returns
    -------
    tuple
        (pseudo_spike_times, pseudo_event_times)
        - pseudo_spike_times: 1D array of spike times in the stitched timeline.
        - pseudo_event_times: 1D array of event start times in the stitched timeline.
    """


    # ensure sorting and alignment
    #spike_times = np.sort(np.reshape(spike_times, (-1, 1)), axis=0)
    #event_times = np.sort(np.reshape(event_times, (-1, 1)), axis=0)

    # pre-allocate
    num_samples = spike_times.size
    num_trials = event_times.size
    median_duration = np.median(np.diff(spike_times, axis=0))
    pseudo_spike_times_list = []
    pseudo_event_times = np.empty((num_trials, 1))
    pseudo_event_times.fill(np.nan)
    pseudo_event_time = 0.0
    last_used_sample = 0
    first_sample = None

    # run
    for trial_idx, event_time in enumerate(event_times):
        # get eligible samples
        start_sample = np.searchsorted(spike_times, event_time, side='right')
        end_sample = np.searchsorted(spike_times, event_time + window_duration, side='right')

        if start_sample is not None and end_sample is not None and start_sample > end_sample:
            end_sample = None
            start_sample = None

        if end_sample is None:
            end_sample = len(spike_times)

        if start_sample is None or end_sample is None:
            use_samples = np.empty(0, dtype=int)
        else:
            end_sample = end_sample - 1
            eligible_samples = np.arange(start_sample, end_sample + 1)
            in_use_samples = np.logical_and(eligible_samples >= 0, eligible_samples < num_samples)
            use_samples = eligible_samples[in_use_samples]

        # check if beginning or end
        if use_samples.size > 0:
            if trial_idx == 0 and not discard_edges:
                use_samples = np.arange(0, use_samples[-1] + 1)
            elif trial_idx == (num_trials - 1) and not discard_edges:
                use_samples = np.arange(use_samples[0], num_samples)

        # add spikes
        if use_samples.size > 0:
            add_times = spike_times[use_samples]
            overlap_mask = use_samples <= last_used_sample

        # get event t
        if trial_idx == 0:
            pseudo_event_time = 0.0
        else:
            if trial_idx > 0 and window_duration > (event_time - event_times[trial_idx - 1]):
                # remove spikes from overlapping epochs
                if use_samples.size > 0:
                    use_samples = use_samples[~overlap_mask]
                    add_times = spike_times[use_samples]

                pseudo_event_time = pseudo_event_time + event_time - event_times[trial_idx - 1]
            else:
                pseudo_event_time = pseudo_event_time + window_duration

        # %% make local pseudo event time
        if use_samples.size == 0:
            local_pseudo_time = np.empty(0)
        else:
            last_used_sample = use_samples[-1]
            local_pseudo_time = add_times - event_time + pseudo_event_time

        if first_sample is None and use_samples.size > 0:
            first_sample = use_samples[0]
            pseudo_t0 = pseudo_event_time

        # assign data for this trial
        pseudo_spike_times_list.append(local_pseudo_time)
        pseudo_event_times[trial_idx] = pseudo_event_time

    # %% add beginning
    if not discard_edges and first_sample is not None and first_sample > 0:
        step_begin = spike_times[first_sample] - spike_times[first_sample - 1]
        samples_add_beginning = np.arange(0, first_sample)
        add_beginning_spikes = spike_times[samples_add_beginning] - spike_times[samples_add_beginning[0]] \
            + pseudo_t0 - step_begin - \
            np.ptp(spike_times[samples_add_beginning]
                   )  # make local to first spike in array, then preceding pseudo event t0
        pseudo_spike_times_list.append(add_beginning_spikes)

    # %% add end
    total_num_spikes = spike_times.size
    last_used_sample = np.searchsorted(spike_times, event_times[-1] + window_duration, side='right')
    if not discard_edges and last_used_sample is not None and (total_num_spikes - 1) > last_used_sample:
        samples_add_end = np.arange(last_used_sample, total_num_spikes)
        add_end_spikes = spike_times[samples_add_end] - event_time + pseudo_event_time + window_duration
        pseudo_spike_times_list.append(add_end_spikes)

    # %% recombine into vector
    pseudo_spike_times = np.concatenate(pseudo_spike_times_list)
    return pseudo_spike_times, pseudo_event_times

# %%

def gen_flatten(list_of_lists):
    for element in list_of_lists:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from gen_flatten(element)
        else:
            yield element

def my_randint(low, high=None, size=None):
    # random.randint(low, high=None, size=None, dtype=int)
    #
    # implementation of randint that returns same values as MATLAB's randi (tested for MATLAB R2023b)

    if high is None:
        high = low
        low = 0
    
    result = np.floor((np.random.random_sample(size)*(high-low)) + low).astype(np.int64)

    return result
