#%%
# -*- coding: utf-8 -*-
import numpy as np
import logging

# from zetapy import msd
from scipy import stats
from zetapy.dependencies import calc_zeta_one, calc_zeta_two, get_temporal_offset_one
from zetapy.ifr_dependencies import get_multi_scale_deriv
from zetapy.plot_dependencies import plotzeta, plotzeta2, plottszeta, plottszeta2
from zetapy.ts_dependencies import calc_ts_zeta_one, calc_ts_zeta_two

# %% two-sample time-series zeta test

def zetatstest2(time1, value1, event_times1, time2, value2, event_times2, max_duration=None, resampling_number=250,
                plot_enabled=False, direct_quantile=False, super_resolution_factor=100):
    """
    Calculates two-sample time-series zeta-test

    Parameters
    ----------
    time1 : 1D array (float)
        timestamps in seconds corresponding to entries in value1.
    value1 : 1D array (float)
        values for condition 1 (e.g., dF/F0 activity).
    event_times1 : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times for condition 1 to calculate mean-rate difference.
    time2 : 1D array (float)
        timestamps in seconds corresponding to entries in value2.
    value2 : 1D array (float)
        values for condition 2 (e.g., dF/F0 activity).
    event_times2 : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times for condition 2 to calculate mean-rate difference.

    Optional Parameters
    ----------
    max_duration : float
        window length for calculating ZETA: ignore all entries beyond this duration after event onset
        (default: minimum of event onset to event onset)
    resampling_number : integer
        number of resamplings (default: 250)
        [Note: if your p-value is close to significance, you should increase this number to enhance the precision]
    plot_enabled : boolean switch
        plotting switch (False: no plot, True: plot figure) (default: False)
    direct_quantile: boolean
        switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
        [Note: requires many resamplings!]
    super_resolution_factor : scalar
        upsampling of data when calculating zeta (default: 100)

    Returns
    -------
    zeta_p : float
        p-value based on Zenith of Event-based Time-locked Anomalies for two-sample comparison
    zeta_data : dict
        additional information of ZETA test
            zeta_p_value; p-value based on Zenith of Event-based Time-locked Anomalies (same as above)
            zeta_score; responsiveness z-score (i.e., >2 is significant)
            ttest_z_score; z-score for mean-rate stim/base difference (i.e., >2 is significant)
            ttest_p_value; p-value based on mean-rate stim/base difference
            zeta_deviation; temporal deviation value underlying ZETA
            zeta_time; time corresponding to ZETA
            zeta_index; entry corresponding to ZETA
            mu1; average spiking rate values per event underlying t-test for condition 1
            mu2; average spiking rate values per event underlying t-test for condition 2
            reference_time: timestamps of trace entries (corresponding to real_difference/random_difference_matrix)
            real_difference; difference between condition 1 and 2
            random_difference_matrix; random differences in cumulative density of spikes
            real_fraction1; cumulative spike vector of condition 1
            real_fraction2; cumulative spike vector of condition 2
            zeta_deviation_inv_sign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            zeta_time_inv_sign; time corresponding to -ZETA
            zeta_index_inv_sign; entry corresponding to -ZETA
            max_duration; window length used to calculate ZETA

    """

    # %% build placeholder outputs
    zeta_p = 1.0
    zeta_data = dict()
    zeta_data['zeta_p_value'] = zeta_p
    zeta_data['zeta_score'] = None
    zeta_data['ttest_z_score'] = None
    zeta_data['ttest_p_value'] = None
    zeta_data['zeta_deviation'] = None
    zeta_data['zeta_time'] = None
    zeta_data['zeta_index'] = None
    zeta_data['mu1'] = None
    zeta_data['mu2'] = None
    zeta_data['zeta_deviation_inv_sign'] = None
    zeta_data['zeta_time_inv_sign'] = None
    zeta_data['zeta_index_inv_sign'] = None
    zeta_data['reference_time'] = None
    zeta_data['real_difference'] = None
    zeta_data['random_differences'] = None
    zeta_data['real_fraction1'] = None
    zeta_data['real_fraction2'] = None
    zeta_data['trace_per_trial1'] = None
    zeta_data['trace_per_trial2'] = None
    zeta_data['max_duration'] = None

    # %% prep data and assert inputs are correct

    # time1 and value1 must be [N by 1] arrays
    assert len(time1.shape) == len(
        value1.shape) and time1.shape == value1.shape, "time1 and value1 have different shapes"
    assert (len(time1.shape) == 1 or time1.shape[1] == 1) and issubclass(
        time1.dtype.type, np.floating), "Input time1 is not a 1D float np.array with >2 spike times"
    time1 = time1.flatten()
    value1 = value1.flatten()
    reorder_indices = np.argsort(time1, axis=0)
    time1 = time1[reorder_indices]
    value1 = value1[reorder_indices]

    # time2 and value2 must be [N by 1] arrays
    assert len(time2.shape) == len(
        value2.shape) and time2.shape == value2.shape, "time2 and value2 have different shapes"
    assert (len(time2.shape) == 1 or time2.shape[1] == 1) and issubclass(
        time2.dtype.type, np.floating), "Input time2 is not a 1D float np.array with >2 spike times"
    time2 = time2.flatten()
    value2 = value2.flatten()
    reorder_indices = np.argsort(time2, axis=0)
    time2 = time2[reorder_indices]
    value2 = value2[reorder_indices]

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

    # check if number of events and values is sufficient
    if time1.size < 3 or event_starts1.size < 3:
        if time1.size < 3:
            message1 = f"Number of entries in time-series ({time1.size}) is too few to calculate zeta; "
        else:
            message1 = ""
        if event_starts1.size < 3:
            message2 = f"Number of events ({event_starts1.size}) is too few to calculate zeta; "
        else:
            message2 = ""

        logging.warning("zetatstest2: " + message1 + message2 + "defaulting to p=1.0")
        return zeta_p, zeta_data

    # check if number of events and values is sufficient
    if time2.size < 3 or event_starts2.size < 3:
        if time2.size < 3:
            message1 = f"Number of entries in time-series ({time2.size}) is too few to calculate zeta; "
        else:
            message1 = ""
        if event_starts2.size < 3:
            message2 = f"Number of events ({event_starts2.size}) is too few to calculate zeta; "
        else:
            message2 = ""

        logging.warning("zetatstest2: " + message1 + message2 + "defaulting to p=1.0")
        return zeta_p, zeta_data

    # is stop supplied?
    if len(event_times1.shape) > 1 and event_times1.shape[1] > 1 and len(event_times2.shape) > 1 and event_times2.shape[1] > 1:
        stop_time_supplied = True
        event_stops1 = event_times1[:, 1]
        event_on_duration1 = event_times1[:, 1] - event_times1[:, 0]
        event_stops2 = event_times2[:, 1]
        event_on_duration2 = event_times2[:, 1] - event_times2[:, 0]
        assert np.all(event_on_duration1 > 0) and np.all(event_on_duration2 >
                                                     0), "at least one event in event_times has a non-positive duration"
        mu1 = np.zeros(event_stops1.shape)
        mu2 = np.zeros(event_stops2.shape)

    else:
        stop_time_supplied = False
        mean_z_score = np.nan
        mean_p_value = np.nan
        mu1 = []
        mu2 = []

    # trial dur
    if max_duration is None:
        max_duration = min(np.min(np.diff(event_times1[:, 0])), np.min(np.diff(event_times2[:, 0])))
    else:
        max_duration = np.float64(max_duration)
        assert max_duration.size == 1 and max_duration > 0, "max_duration is not a positive scalar float"

    # get resampling num
    if resampling_number is None:
        resampling_number = np.int64(250)
    else:
        resampling_number = np.int64(resampling_number)
        assert resampling_number.size == 1 and resampling_number > 1, "resampling_number is not a positive integer"

    # plotting
    if plot_enabled is None:
        plot_enabled = False
    else:
        assert isinstance(plot_enabled, bool), "plot_enabled is not a boolean"

    # direct quantile computation
    if direct_quantile is None:
        direct_quantile = False
    else:
        assert isinstance(direct_quantile, bool), "direct_quantile is not a boolean"

    # %% check data length
    assert (len(time1) == len(value1) and len(time2) == len(value2)), 'Input lengths do not match'

    assert np.min(event_times1[:, 0]) > np.min(time1) and np.max(event_times1[:, 0]) < (np.max(time1)-max_duration) \
        and np.min(event_times2[:, 0]) > np.min(time2) and np.max(event_times2[:, 0]) < (np.max(time2)-max_duration),\
        'Events exist outside of data period'

    # %% calculate zeta
    zeta_data_two = calc_ts_zeta_two(
        time1, value1, event_times1, time2, value2, event_times2, super_resolution_factor, max_duration,
        resampling_number, direct_quantile)

    # update and unpack
    zeta_data.update(zeta_data_two)
    reference_time = zeta_data['reference_time']
    real_difference = zeta_data['real_difference']
    real_fraction1 = zeta_data['real_fraction1']
    real_fraction2 = zeta_data['real_fraction2']
    random_differences = zeta_data['random_differences']
    zeta_p = zeta_data['zeta_p_value']
    zeta_score = zeta_data['zeta_score']
    zeta_index = zeta_data['zeta_index']
    trace_per_trial1 = zeta_data['trace_per_trial1']
    trace_per_trial2 = zeta_data['trace_per_trial2']

    # check if calculation is valid, otherwise return empty values
    if zeta_index is None:
        logging.warning("zetatstest2: calculation failed, defaulting to p=1.0")
        return zeta_p, zeta_data

    # %% extract real outputs
    # get location
    zeta_time = reference_time[zeta_index]
    zeta_deviation = real_difference[zeta_index]

    # find peak of inverse sign
    zeta_index_inv_sign = np.argmax(-np.sign(zeta_deviation)*real_difference)
    zeta_time_inv_sign = reference_time[zeta_index_inv_sign]
    zeta_deviation_inv_sign = real_difference[zeta_index_inv_sign]

    # %% calculate mean-rate difference
    if stop_time_supplied:
        for trace_number in [1, 2]:
            if trace_number == 1:
                event_starts = event_times1[:, 0]
                event_stops = event_times1[:, 1]
                current_trace_time = time1
                current_trace_activity = value1
            else:
                event_starts = event_times2[:, 0]
                event_stops = event_times2[:, 1]
                current_trace_time = time2
                current_trace_activity = value2

            time_points_count = len(current_trace_time)
            max_repetitions = len(event_starts)

            # Vectorize index finding using np.searchsorted
            stim_start_indices = np.searchsorted(current_trace_time, event_starts, side='right') - 1
            stim_stop_indices = np.searchsorted(current_trace_time, event_stops, side='right') + 1
            base_stop_times = event_starts + max_duration
            base_stop_indices = np.searchsorted(current_trace_time, base_stop_times, side='right') + 1

            # Apply bounds to indices
            stim_start_indices = np.maximum(0, stim_start_indices)
            stim_stop_indices = np.minimum(time_points_count, stim_stop_indices)
            base_stop_indices = np.minimum(time_points_count, base_stop_indices)

            mu_base = np.empty(max_repetitions)
            mu_base.fill(np.nan)
            mu_duration = np.empty(max_repetitions)
            mu_duration.fill(np.nan)

            # Loop to calculate means using pre-calculated indices
            for event_index in range(max_repetitions):
                # Check for valid duration
                if (base_stop_times[event_index] - event_stops[event_index]) <= 0:
                    raise Exception(
                        "Input error: event stop times do not precede the next stimulus' start time")

                # Extract activity for base and stim periods
                base_trace_values = current_trace_activity[stim_stop_indices[event_index]:base_stop_indices[event_index]]
                stim_trace_values = current_trace_activity[stim_start_indices[event_index]:stim_stop_indices[event_index]]

                # Calculate means, handling empty slices
                if len(base_trace_values) > 0:
                    mu_base[event_index] = np.mean(base_trace_values)
                if len(stim_trace_values) > 0:
                    mu_duration[event_index] = np.mean(stim_trace_values)

            if trace_number == 1:
                mu_base1 = mu_base
                mu_duration1 = mu_duration
            else:
                mu_base2 = mu_base
                mu_duration2 = mu_duration

        # difference
        mu1 = mu_duration1 - mu_base1
        mu2 = mu_duration2 - mu_base2

        # get metrics
        mean_p_value = stats.ttest_ind(mu1, mu2)[1]
        mean_z_score = -stats.norm.ppf(mean_p_value/2)

    # %% build output structure
    # fill zeta_data
    zeta_data['zeta_p_value'] = zeta_p
    zeta_data['zeta_score'] = zeta_score
    zeta_data['zeta_deviation'] = zeta_deviation
    zeta_data['zeta_time'] = zeta_time
    zeta_data['zeta_index'] = zeta_index
    if stop_time_supplied:
        zeta_data['ttest_z_score'] = mean_z_score
        zeta_data['ttest_p_value'] = mean_p_value
        zeta_data['mu1'] = mu1
        zeta_data['mu2'] = mu2

    zeta_data['zeta_deviation_inv_sign'] = zeta_deviation_inv_sign
    zeta_data['zeta_time_inv_sign'] = zeta_time_inv_sign
    zeta_data['zeta_index_inv_sign'] = zeta_index_inv_sign
    zeta_data['reference_time'] = reference_time
    zeta_data['real_difference'] = real_difference
    zeta_data['random_differences'] = random_differences
    zeta_data['real_fraction1'] = real_fraction1
    zeta_data['real_fraction2'] = real_fraction2
    zeta_data['max_duration'] = max_duration
    zeta_data['trace_per_trial1'] = trace_per_trial1
    zeta_data['trace_per_trial2'] = trace_per_trial2

    # %% plot
    if plot_enabled:
        plottszeta2(zeta_data)

    # %% return
    return zeta_p, zeta_data

# %% two-sample zeta test


def zetatest2(spike_times1, event_times1, spike_times2, event_times2, max_duration=None, resampling_number=250,
              plot_enabled=False, direct_quantile=False):
    """
    Calculates two-sample zeta-test

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

    max_duration : float
        window length for calculating ZETA: ignore all spikes beyond this duration after event onset
        (default: minimum of event onset to event onset)
    resampling_number : integer
        number of resamplings (default: 250)
        [Note: if your p-value is close to significance, you should increase this number to enhance the precision]
    plot_enabled : boolean switch
        plotting switch (False: no plot, True: plot figure) (default: False)
    direct_quantile: boolean
         switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
         [Note: requires many resamplings!]

    Returns
    -------
    zeta_p : float
        p-value based on Zenith of Event-based Time-locked Anomalies for two-sample comparison
    zeta_data : dict
        additional information of ZETA test
            zeta_p; p-value based on Zenith of Event-based Time-locked Anomalies (same as above)
            zeta_score; responsiveness z-score (i.e., >2 is significant)
            mean_z_score; z-score for mean-rate stim/base difference (i.e., >2 is significant)
            mean_p_value; p-value based on mean-rate stim/base difference
            zeta_deviation; temporal deviation value underlying ZETA
            zeta_time; time corresponding to ZETA
            zeta_index; entry corresponding to ZETA
            mu1; average spiking rate values per event underlying t-test for condition 1
            mu2; average spiking rate values per event underlying t-test for condition 2
            spike_time_vector: timestamps of spike times (corresponding to real_difference)
            real_difference; difference between condition 1 and 2
            real_fraction1; cumulative spike vector of condition 1
            real_fraction2; cumulative spike vector of condition 2
            deviation_inv_sign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            zeta_time_inv_sign; time corresponding to -ZETA
            zeta_index_inv_sign; entry corresponding to -ZETA
            random_times; timestamps for null-hypothesis resampled data
            random_difference; null-hypothesis temporal deviation vectors of resampled data
            max_duration; window length used to calculate ZETA

    """

    # %% build placeholder outputs
    zeta_p = 1.0
    zeta_data = dict()

    # fill zeta_data
    # ZETA significance
    zeta_data['zeta_p_value'] = zeta_p
    zeta_data['zeta_score'] = None
    zeta_data['ttest_z_score'] = None
    zeta_data['ttest_p_value'] = None
    zeta_data['zeta_deviation'] = None
    zeta_data['zeta_time'] = None
    zeta_data['zeta_index'] = None
    zeta_data['mu1'] = None
    zeta_data['mu2'] = None
    zeta_data['deviation_inv_sign'] = None
    zeta_data['zeta_time_inv_sign'] = None
    zeta_data['zeta_index_inv_sign'] = None
    zeta_data['spike_time_vector'] = None
    zeta_data['real_difference'] = None
    zeta_data['real_fraction1'] = None
    zeta_data['real_fraction2'] = None
    zeta_data['random_times'] = None
    zeta_data['random_differences'] = None
    zeta_data['max_duration'] = None

    # %% prep data and assert inputs are correct

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

    # is stop supplied?
    if len(event_times1.shape) > 1 and event_times1.shape[1] > 1 and len(event_times2.shape) > 1 and event_times2.shape[1] > 1:
        stop_time_supplied = True
        event_on_duration1 = event_times1[:, 1] - event_times1[:, 0]
        assert np.all(event_on_duration1 > 0), "at least one event in event_times1 has a negative duration"

        event_on_duration2 = event_times2[:, 1] - event_times2[:, 0]
        assert np.all(event_on_duration2 > 0), "at least one event in event_times2 has a negative duration"

        # trial dur
        if max_duration is None:
            max_duration = min(np.min(event_on_duration1), np.min(event_on_duration2))
    else:
        stop_time_supplied = False
        mean_z_score = np.nan
        mean_p_value = np.nan

    # trial dur
    if max_duration is None:
        max_duration = min(np.min(np.diff(event_times1[:, 0])), np.min(np.diff(event_times2[:, 0])))
    else:
        max_duration = np.float64(max_duration)
        assert max_duration.size == 1 and max_duration > 0, "max_duration is not a positive scalar float"

    # get resampling num
    if resampling_number is None:
        resampling_number = np.int64(250)
    else:
        resampling_number = np.int64(resampling_number)
        assert resampling_number.size == 1 and resampling_number > 1, "resampling_number is not a positive integer"

    # plotting
    if plot_enabled is None:
        plot_enabled = False
    else:
        assert isinstance(plot_enabled, bool), "plot_enabled is not a boolean"

    # direct quantile comnputation
    if direct_quantile is None:
        direct_quantile = False
    else:
        assert isinstance(direct_quantile, bool), "direct_quantile is not a boolean"

    # %% calculate zeta
    event_starts1 = event_times1[:, 0]
    event_starts2 = event_times2[:, 0]
    # if len(event_starts1) > 1 and (len(spike_times1)+len(spike_times2)) > 0 and max_duration is not None and max_duration > 0:
    zeta_data_two = calc_zeta_two(spike_times1, event_starts1, spike_times2, event_starts2, max_duration,
                                  resampling_number, direct_quantile)

    # %% calculate zeta
    # update and unpack
    zeta_data.update(zeta_data_two)
    spike_time_vector = zeta_data['spike_time_vector']
    real_difference = zeta_data['real_difference']
    zeta_p = zeta_data['zeta_p_value']
    zeta_index = zeta_data['zeta_index']

    # check if calculation is valid, otherwise return empty values
    if zeta_index is None:
        logging.warning("zetatest2: calculation failed, defaulting to p=1.0")
        return zeta_p, zeta_data

    # %% extract real outputs
    # get location
    zeta_time = spike_time_vector[zeta_index]
    zeta_deviation = real_difference[zeta_index]

    # find peak of inverse sign
    zeta_index_inv_sign = np.argmax(-np.sign(zeta_deviation)*real_difference)
    zeta_time_inv_sign = spike_time_vector[zeta_index_inv_sign]
    deviation_inv_sign = real_difference[zeta_index_inv_sign]

    # %% calculate mean-rate difference with t-test
    if stop_time_supplied:
        # calculate spike counts and durations during baseline and stimulus times
        response_bins_duration = np.sort(np.reshape(event_times1, -1))
        counts, bins = np.histogram(spike_times1, bins=response_bins_duration)
        durations = np.diff(response_bins_duration)
        mu1 = np.divide(np.float64(counts[0:len(counts):2]), durations[0:len(durations):2])

        # calculate mean rates during off-times
        response_bins_duration = np.sort(np.reshape(event_times2, -1))
        counts, bins = np.histogram(spike_times2, bins=response_bins_duration)
        durations = np.diff(response_bins_duration)
        mu2 = np.divide(np.float64(counts[0:len(counts):2]), durations[0:len(durations):2])

        # get metrics
        mean_p_value = stats.ttest_ind(mu1, mu2)[1]
        mean_z_score = -stats.norm.ppf(mean_p_value/2)

    # %% build output dictionary
    # fill zeta_data
    zeta_data['zeta_deviation'] = zeta_deviation
    zeta_data['zeta_time'] = zeta_time
    if stop_time_supplied:
        zeta_data['ttest_z_score'] = mean_z_score
        zeta_data['ttest_p_value'] = mean_p_value
        zeta_data['mu1'] = mu1
        zeta_data['mu2'] = mu2

    # inverse-sign ZETA
    zeta_data['deviation_inv_sign'] = deviation_inv_sign
    zeta_data['zeta_time_inv_sign'] = zeta_time_inv_sign
    zeta_data['zeta_index_inv_sign'] = zeta_index_inv_sign
    # window used for analysis
    zeta_data['max_duration'] = max_duration

    # %% plot
    if plot_enabled:
        plotzeta2(spike_times1, event_starts1, spike_times2, event_starts2, zeta_data)

    # %% return outputs
    return zeta_p, zeta_data

# %% time-series zeta


def zetatstest(time, value, event_times, max_duration=None, resampling_number=100, plot_enabled=False, jitter_size=2.0,
               direct_quantile=False, stitch_enabled=True):
    """
    Calculates responsiveness index zeta for timeseries data

    Parameters
    ----------
    time : 1D array (float)
        timestamps in seconds corresponding to entries in value.
    value : 1D array (float)
        values (e.g., dF/F0 activity).
    event_times : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate mean-rate difference.
    max_duration : float
        window length for calculating ZETA: ignore all entries beyond this duration after event onset
        (default: minimum of event onset to event onset)
    resampling_number : integer
        number of resamplings (default: 100)
        [Note: if your p-value is close to significance, you should increase this number to enhance the precision]
    plot_enabled : boolean switch
        plotting switch (False: no plot, True: plot figure) (default: False)
    jitter_size : float
        sets the temporal jitter window relative to max_duration (default: 2.0)
    direct_quantile: boolean
         switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
         [Note: requires many resamplings!]
    stitch_enabled: boolean
        switch to perform data stitching (default: True)


    Returns
    -------
    zeta_p : float
        p-value based on Zenith of Event-based Time-locked Anomalies
    zeta_data : dict
        additional information of ZETA test
            zeta_p; p-value based on Zenith of Event-based Time-locked Anomalies (same as above)
            zeta_score; responsiveness z-score (i.e., >2 is significant)
            mean_z_score; z-score for mean-rate stim/base difference (i.e., >2 is significant)
            mean_p_value; p-value based on mean-rate stim/base difference
            zeta_deviation; temporal deviation value underlying ZETA
            latency_zeta; time corresponding to ZETA
            zeta_index; entry corresponding to ZETA
            mu_duration; mean activity per trial during stim (used for mean-rate test)
            mu_base; mean activity per trial during baseline (used for mean-rate test)
            deviation_inv_sign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            latency_inv_zeta; time corresponding to -ZETA
            index_inv_sign; entry corresponding to -ZETA
            real_time: timestamps of event-centered time-series values (corresponding to real_deviation)
            real_deviation; temporal deviation vector of data
            real_fraction; cumulative distribution of spike times
            real_fraction_linear; linear baseline of cumulative distribution
            random_deviation_matrix; baseline temporal deviation matrix of jittered data
            max_duration; window length used to calculate ZETA

    """
    # %% build placeholder outputs
    zeta_p = 1.0
    zeta_data = dict()

    # fill zeta_data
    # ZETA significance
    zeta_data['zeta_p_value'] = zeta_p
    zeta_data['zeta_score'] = None
    zeta_data['ttest_z_score'] = None
    zeta_data['ttest_p_value'] = None
    zeta_data['zeta_deviation'] = None
    zeta_data['latency_zeta'] = None
    zeta_data['zeta_index'] = None
    zeta_data['mu_duration'] = None
    zeta_data['mu_base'] = None
    zeta_data['deviation_inv_sign'] = None
    zeta_data['latency_inv_zeta'] = None
    zeta_data['index_inv_sign'] = None
    zeta_data['real_time'] = None
    zeta_data['real_deviation'] = None
    zeta_data['real_fraction'] = None
    zeta_data['real_fraction_linear'] = None
    zeta_data['random_times'] = None
    zeta_data['random_deviations'] = None
    zeta_data['max_duration'] = None

    # %% prep data and assert inputs are correct

    # time and value must be [N by 1] arrays
    assert len(time.shape) == len(
        value.shape) and time.shape == value.shape, "time and value have different shapes"
    assert (len(time.shape) == 1 or time.shape[1] == 1) and issubclass(
        time.dtype.type, np.floating), "Input time is not a 1D float np.array with >2 spike times"
    time = time.flatten()
    value = value.flatten()
    reorder_indices = np.argsort(time, axis=0)
    time = time[reorder_indices]
    value = value[reorder_indices]

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
            message1 = f"Number of entries in time-series ({time.size}) is too few to calculate zeta; "
        else:
            message1 = ""
        if event_starts.size < 3:
            message2 = f"Number of events ({event_starts.size}) is too few to calculate zeta; "
        else:
            message2 = ""
        logging.warning("zetatstest: " + message1 + message2 + "defaulting to p=1.0")

        return zeta_p, zeta_data

    # is stop supplied?
    if len(event_times.shape) > 1 and event_times.shape[1] > 1:
        stop_time_supplied = True
        event_stops = event_times[:, 1]
        event_on_duration = event_times[:, 1] - event_times[:, 0]
        assert np.all(event_on_duration > 0), "at least one event in event_times has a non-positive duration"
        mu_duration = np.zeros(event_stops.shape)
        mu_base = np.zeros(event_stops.shape)

    else:
        stop_time_supplied = False
        mean_z_score = np.nan
        mean_p_value = np.nan
        mu_duration = []
        mu_base = []

    # trial dur
    if max_duration is None:
        max_duration = np.min(np.diff(event_times[:, 0]))
    else:
        max_duration = np.float64(max_duration)
        assert max_duration.size == 1 and max_duration > 0, "max_duration is not a positive scalar float"

    # get resampling num
    if resampling_number is None:
        resampling_number = np.int64(100)
    else:
        resampling_number = np.int64(resampling_number)
        assert resampling_number.size == 1 and resampling_number > 1, "resampling_number is not a positive integer"

    # plotting
    if plot_enabled is None:
        plot_enabled = False
    else:
        assert isinstance(plot_enabled, bool), "plot_enabled is not a boolean"

    # jitter
    if jitter_size is None:
        jitter_size = np.float64(2.0)
    else:
        jitter_size = np.float64(jitter_size)
        assert jitter_size.size == 1 and jitter_size > 0, "jitter_size is not a postive scalar float"

    # direct quantile computation
    if direct_quantile is None:
        direct_quantile = False
    else:
        assert isinstance(direct_quantile, bool), "direct_quantile is not a boolean"

    # stitch?
    if stitch_enabled is None:
        stitch_enabled = True
    else:
        assert isinstance(stitch_enabled, bool), "stitch_enabled is not a boolean"
    if stitch_enabled & (np.min(np.diff(event_times[:,0])) < max_duration):
        logging.warning('zetatstest: some events are too close together and will be excluded from the stitching procedure')


    # %% check data length
    data_t0 = np.min(time)
    required_t0 = np.min(event_starts) - jitter_size*max_duration
    if data_t0 > required_t0:
        logging.warning("zetatstest: leading data preceding first event is insufficient for maximal jittering")

    data_t_end = np.max(time)
    required_t_end = np.max(event_starts) + jitter_size*max_duration + max_duration
    if data_t_end < required_t_end:
        logging.warning("zetatstest: lagging data after last event is insufficient for maximal jittering")

    # %% calculate zeta
    zeta_data_one = calc_ts_zeta_one(time, value, event_starts, max_duration, resampling_number,
                              direct_quantile, jitter_size, stitch_enabled)

    # update and unpack
    zeta_data.update(zeta_data_one)
    real_time = zeta_data['real_time']
    real_deviation = zeta_data['real_deviation']
    zeta_p = zeta_data['zeta_p_value']
    zeta_index = zeta_data['zeta_index']

    # check if calculation is valid, otherwise return empty values
    if zeta_index is None:
        logging.warning("zetatstest: calculation failed, defaulting to p=1.0")
        return zeta_p, zeta_data

    # %% extract real outputs
    # get location
    latency_zeta = real_time[zeta_index]
    zeta_deviation = real_deviation[zeta_index]

    # find peak of inverse sign
    index_inv_sign = np.argmax(-np.sign(zeta_deviation)*real_deviation)
    latency_inv_zeta = real_time[index_inv_sign]
    deviation_inv_sign = real_deviation[index_inv_sign]

    # %% calculate mean-rate difference
    if stop_time_supplied:
        # pre-allocate
        time_points_count = len(time)-1

        # Vectorize index finding using np.searchsorted
        stim_start_indices = np.searchsorted(time, event_starts, side='right') - 1
        stim_stop_indices = np.searchsorted(time, event_stops, side='right') + 1
        base_stop_times = event_starts + max_duration
        base_stop_indices = np.searchsorted(time, base_stop_times, side='right') + 1

        # Apply bounds to indices
        stim_start_indices = np.maximum(0, stim_start_indices)
        stim_stop_indices = np.minimum(time_points_count, stim_stop_indices)
        base_stop_indices = np.minimum(time_points_count, base_stop_indices)

        # Loop to calculate means using pre-calculated indices
        for event_index in range(len(event_starts)):
            # Check for valid duration
            if (base_stop_times[event_index] - event_stops[event_index]) <= 0:
                raise Exception(
                    "Input error: event stop times do not precede the next stimulus' start time")

            # Extract activity for base and stim periods
            base_trace_values = value[stim_stop_indices[event_index]:base_stop_indices[event_index]]
            stim_trace_values = value[stim_start_indices[event_index]:stim_stop_indices[event_index]]

            # Calculate means, handling empty slices
            if len(base_trace_values) > 0:
                mu_base[event_index] = np.mean(base_trace_values)
            if len(stim_trace_values) > 0:
                mu_duration[event_index] = np.mean(stim_trace_values)

        # get metrics
        use_trials = np.logical_and(~np.isnan(mu_duration), ~np.isnan(mu_base))
        mu_duration = mu_duration[use_trials]
        mu_base = mu_base[use_trials]
        mean_p_value = stats.ttest_rel(mu_duration, mu_base)[1]
        mean_z_score = -stats.norm.ppf(mean_p_value/2)

    # %% build output structure
    # fill zeta_data
    zeta_data['zeta_deviation'] = zeta_deviation
    zeta_data['latency_zeta'] = latency_zeta
    if stop_time_supplied:
        zeta_data['ttest_z_score'] = mean_z_score
        zeta_data['ttest_p_value'] = mean_p_value
        zeta_data['mu_duration'] = mu_duration
        zeta_data['mu_base'] = mu_base

    # inverse-sign ZETA
    zeta_data['deviation_inv_sign'] = deviation_inv_sign
    zeta_data['latency_inv_zeta'] = latency_inv_zeta
    zeta_data['index_inv_sign'] = index_inv_sign
    # window used for analysis
    zeta_data['max_duration'] = max_duration

    # %% plot
    if plot_enabled:
        plottszeta(time, value, event_starts, zeta_data)

    # %% return
    return zeta_p, zeta_data

# %% zetatest


def zetatest(spike_times, event_times, max_duration=None, resampling_number=100, plot_enabled=False, jitter_size=2.0,
             restrict_range=(-np.inf, np.inf), stitch_enabled=True, direct_quantile=False, return_rate=False):
    """
    Calculates neuronal responsiveness index ZETA.

    Syntax:
    zeta_p, zeta_data, rate_data = zetatest(spike_times, event_times,
                                                   max_duration=None, resampling_number=100, plot_enabled=False, jitter_size=2.0,
                                                   restrict_range=(-np.inf, np.inf), stitch_enabled=True,
                                                   direct_quantile=False, return_rate=False):

    Parameters
    ----------
    spike_times : 1D array (float)
        spike times (in seconds).
    event_times : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate mean-rate difference.

    max_duration : float
        window length for calculating ZETA: ignore all spikes beyond this duration after event onset
        (default: minimum of event onset to event onset)
    resampling_number : integer
        number of resamplings (default: 100)
        [Note: if your p-value is close to significance, you should increase this number to enhance the precision]
    plot_enabled : boolean switch
        plotting switch (False: no plot, True: plot figure) (default: False)
    jitter_size : float
        sets the temporal jitter window relative to max_duration (default: 2.0)
    restrict_range : 2-element tuple
        temporal range within which to restrict onset/peak latencies (default: [-inf inf])
    stitch_enabled : boolean
        switch to use data-stitching to ensure continuous time (default: True)
    direct_quantile: boolean
         switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
         [Note: requires many resamplings!]
    return_rate : boolean
        switch to return dictionary with spiking rate features [note: return-time is much faster if this is False]

    Returns
    -------
    zeta_p : float
        p-value based on Zenith of Event-based Time-locked Anomalies
    zeta_data : dict
        additional information of ZETA test
            zeta_p; p-value based on Zenith of Event-based Time-locked Anomalies (same as above)
            zeta_score; responsiveness z-score (i.e., >2 is significant)
            mean_z_score; z-score for mean-rate stim/base difference (i.e., >2 is significant)
            mean_p_value; p-value based on mean-rate stim/base difference
            zeta_deviation; temporal deviation value underlying ZETA
            latency_zeta; time corresponding to ZETA
            zeta_index; entry corresponding to ZETA
            mu_duration; spiking rate per trial during stim (used for mean-rate test)
            mu_pre; spiking rate per trial during baseline (used for mean-rate test)
            deviation_inv_sign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            latency_inv_zeta; time corresponding to -ZETA
            index_inv_sign; entry corresponding to -ZETA
            spike_time_vector: timestamps of spike times (corresponding to real_deviation)
            real_deviation; temporal deviation vector of data
            real_fraction; cumulative distribution of spike times
            real_fraction_linear; linear baseline of cumulative distribution
            random_times; jittered spike times corresponding to random_deviation_matrix
            random_deviation_matrix; baseline temporal deviation matrix of jittered data
            max_duration; window length used to calculate ZETA
            latencies; 4-element array with latency times for different events:
                1) Latency of ZETA [same as zeta_deviation]
                2) Latency of largest z-score with inverse sign to ZETA (same as latency_inv_zeta])
                3) Peak time of instantaneous firing rate (same as rate_data['peak_latency'])
                4) Onset time, defined as the first crossing of peak half-height (same as rate_data['peak_onset_latency'])
               For true onset latencies, we recommend using LatenZy
               https://github.com/Herseninstituut/latenZy, based on the zeta-test.
            latency_values; values corresponding to above latencies (ZETA, -ZETA, rate at peak, rate at onset)

    rate_data : dict (empty if return_rate was not set to True)
        additional parameters of the firing rate, return with return_rate
            rate_vector; instantaneous spiking rates (like a PSTH)
            time_vector; time-points corresponding to rate_vector (same as zeta_data.spike_time_vector)
            mean_multi_scale_derivative; Mean of multi-scale derivatives
            scale_vector; timescales used to calculate derivatives
            multi_scale_derivative_matrix; multi-scale derivatives matrix
            values_vector; values on which rate_vector is calculated (same as zeta_data.zeta_score)
        Data on the peak and onset:
            peak_latency; time of peak (in seconds) [latencies entry #3]
            peak_width; duration of peak (in seconds) [latencies entry #3]
            peak_start_stop_times; start and stop time of peak (in seconds) [latencies entry #3]
            peak_location_index; spike index of peak (corresponding to zeta_data.spike_time_vector) [latencies entry #3]
            peak_start_stop_indices; spike indices of peak start/stop (corresponding to zeta_data.spike_time_vector) [latencies entry #3]
            peak_onset_latency: latency for peak onset [latencies entry #4]

            For true onset latencies, we recommend using LatenZy
            https://github.com/Herseninstituut/latenZy, based on the zeta-test.

    """

    # %% build placeholder outputs
    zeta_p = 1.0
    zeta_data = dict()
    rate_data = dict()
    latencies = np.empty((4, 1))
    latencies.fill(np.nan)
    latency_values = latencies

    # fill zeta_data
    # ZETA significance
    zeta_data['zeta_p_value'] = zeta_p
    zeta_data['zeta_score'] = None
    zeta_data['ttest_z_score'] = None
    zeta_data['ttest_p_value'] = None
    zeta_data['zeta_deviation'] = None
    zeta_data['latency_zeta'] = None
    zeta_data['zeta_index'] = None
    zeta_data['mu_duration'] = None
    zeta_data['mu_pre'] = None
    zeta_data['deviation_inv_sign'] = None
    zeta_data['latency_inv_zeta'] = None
    zeta_data['index_inv_sign'] = None
    zeta_data['spike_time_vector'] = None
    zeta_data['real_deviation'] = None
    zeta_data['real_fraction'] = None
    zeta_data['real_fraction_linear'] = None
    zeta_data['random_times'] = None
    zeta_data['random_deviations'] = None
    zeta_data['max_duration'] = None
    rate_data['rate_vector'] = None
    rate_data['timestamps'] = None
    rate_data['mean_derivative'] = None
    rate_data['scales'] = None
    rate_data['msd_matrix'] = None
    rate_data['values'] = None

    # %% prep data and assert inputs are correct

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

    # check if number of events and spikes is sufficient
    if spike_times.size < 3 or event_starts.size < 3:
        if spike_times.size < 3:
            message1 = f"Number of spikes ({spike_times.size}) is too few to calculate zeta; "
        else:
            message1 = ""
        if event_starts.size < 3:
            message2 = f"Number of events ({event_starts.size}) is too few to calculate zeta; "
        else:
            message2 = ""
        logging.warning("zetatest: " + message1 + message2 + "defaulting to p=1.0")

        return zeta_p, zeta_data, rate_data, latencies

    # is stop supplied?
    if len(event_times.shape) > 1 and event_times.shape[1] > 1:
        stop_time_supplied = True
        event_on_duration = event_times[:, 1] - event_times[:, 0]
        assert np.all(event_on_duration > 0), "at least one event in event_times has a negative duration"

    else:
        stop_time_supplied = False
        mean_z_score = np.nan
        mean_p_value = np.nan

    # trial dur
    if max_duration is None:
        max_duration = np.min(np.diff(event_times[:, 0]))
    else:
        max_duration = np.float64(max_duration)
        assert max_duration.size == 1 and max_duration > 0, "max_duration is not a positive scalar float"

    # get resampling num
    if resampling_number is None:
        resampling_number = np.int64(100)
    else:
        resampling_number = np.int64(resampling_number)
        assert resampling_number.size == 1 and resampling_number > 1, "resampling_number is not a positive integer"

    # plotting
    if plot_enabled is None:
        plot_enabled = False
    else:
        assert isinstance(plot_enabled, bool), "plot_enabled is not a boolean"

    # jitter
    if jitter_size is None:
        jitter_size = np.float64(2.0)
    else:
        jitter_size = np.float64(jitter_size)
        assert jitter_size.size == 1 and jitter_size > 0, "jitter_size is not a postive scalar float"

    # restrict range
    if restrict_range is None:
        restrict_range = np.float64((-np.inf, np.inf))
    else:
        restrict_range = np.float64(restrict_range)
        assert restrict_range.size == 2, "restrict_range does not have two values"

    # stitching
    if stitch_enabled is None:
        stitch_enabled = True
    else:
        assert isinstance(stitch_enabled, bool), "stitch_enabled is not a boolean"

    # direct quantile comnputation
    if direct_quantile is None:
        direct_quantile = False
    else:
        assert isinstance(direct_quantile, bool), "direct_quantile is not a boolean"

    # return rate_data
    if return_rate is None:
        return_rate = False
    else:
        assert isinstance(return_rate, bool), "return_rate is not a boolean"
    if plot_enabled is True and return_rate is False:
        return_rate = True
        logging.warning(
            "zetatest: plot_enabled was True, but you requested plotting, so return_rate is now set to True")

    # %% calculate zeta
    zeta_data_one = calc_zeta_one(spike_times, event_starts, max_duration, resampling_number,
                                  direct_quantile, jitter_size, stitch_enabled)

    # update and unpack
    zeta_data.update(zeta_data_one)
    spike_time_vector = zeta_data['spike_time_vector']
    real_deviation = zeta_data['real_deviation']
    zeta_p = zeta_data['zeta_p_value']
    zeta_index = zeta_data['zeta_index']

    # check if calculation is valid, otherwise return empty values
    if zeta_index is None:
        logging.warning("zetatest: calculation failed, defaulting to p=1.0")
        return zeta_p, zeta_data, rate_data

    # %% extract real outputs
    # get location
    latency_zeta = spike_time_vector[zeta_index]
    zeta_deviation = real_deviation[zeta_index]

    # find peak of inverse sign
    index_inv_sign = np.argmax(-np.sign(zeta_deviation)*real_deviation)
    latency_inv_zeta = spike_time_vector[index_inv_sign]
    deviation_inv_sign = real_deviation[index_inv_sign]

    # %% calculate mean-rate difference with t-test
    if stop_time_supplied:
        # calculate spike counts and durations during baseline and stimulus times
        response_bins_duration = np.sort(np.reshape(event_times, -1))
        counts, bins = np.histogram(spike_times, bins=response_bins_duration)
        durations = np.diff(response_bins_duration)

        # mean rate during on-time
        mu_duration = np.divide(np.float64(counts[0:len(counts):2]), durations[0:len(durations):2])

        # calculate mean rates during off-times
        start1 = np.min(response_bins_duration)
        first_pre_duration = start1 - np.max(start1 - np.median(durations[1:len(durations):2]), initial=0) + np.finfo(float).eps
        r1 = np.sum(np.logical_and(spike_times > (start1 - first_pre_duration), spike_times < start1))
        counts = np.concatenate([[r1], counts[1:len(counts):2]])
        durations = np.concatenate([[first_pre_duration], durations[1:len(durations):2]])
        mu_pre = np.divide(counts, durations)

        # get metrics
        zeta_data['ttest_p_value'] = stats.ttest_rel(mu_duration, mu_pre)[1]
        zeta_data['ttest_z_score'] = -stats.norm.ppf(zeta_data['ttest_p_value'] / 2)
        zeta_data['mu_duration'] = mu_duration
        zeta_data['mu_pre'] = mu_pre


    # %% calculate instantaneous firing rates
    if return_rate:
        # get average of multi-scale derivatives, and rescaled to instantaneous spiking rate
        mean_rate = spike_time_vector.size / (max_duration * event_starts.size)
        rate_vector, rate_data = get_multi_scale_deriv(
            spike_time_vector, real_deviation, mean_rate=mean_rate, max_duration=max_duration)

    # %% build output dictionary

    zeta_data['zeta_deviation'] = zeta_deviation
    zeta_data['latency_zeta'] = latency_zeta
    zeta_data['deviation_inv_sign'] = deviation_inv_sign
    zeta_data['latency_inv_zeta'] = latency_inv_zeta
    zeta_data['index_inv_sign'] = index_inv_sign
    zeta_data['max_duration'] = max_duration

    # Plot
    if plot_enabled:
        plotzeta(spike_times, event_starts, zeta_data, rate_data)

    return zeta_p, zeta_data, rate_data

# %% IFR

def ifr(spike_times, event_times, max_duration=None, smooth_sd=2.0, min_scale=None, base_value=1.5):
    """
    Calculates the Instantaneous Firing Rate (IFR) using multi-scale derivatives.

    Parameters
    ----------
    spike_times : 1D array (float)
        Spike times (in seconds).
    event_times : 1D or 2D array (float)
        Event onset times (s), or [T x 2] including event off times.
    max_duration : float, optional
        Window length for calculating IFR (default: minimum inter-event interval).
    smooth_sd : float, optional
        Standard deviation for Gaussian smoothing (default: 2.0).
    min_scale : float, optional
        Minimum timescale for derivative calculation.
    base_value : float, optional
        Base value for scale calculation (default: 1.5).

    Returns
    -------
    time_vector : 1D array
        Timestamps corresponding to the rate_vector.
    rate_vector : 1D array
        Calculated instantaneous firing rates.
    ifr_data : dict
        Additional information including timestamps, rate_vector,
        deviation_vector, and scales used.
    """

    # %% prep data and assert inputs are correct
    # pre-allocate outputs
    time_vector, rate_vector = np.empty(0), np.empty(0)
    ifr_data = dict()
    ifr_data['timestamps'] = time_vector
    ifr_data['rate_vector'] = rate_vector
    ifr_data['deviation_vector'] = np.empty(0)
    ifr_data['scales'] = np.empty(0)

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

    # check if number of events and spikes is sufficient
    if spike_times.size < 3 or event_starts.size < 3:
        if spike_times.size < 3:
            message1 = f"Number of spikes ({spike_times.size}) is too few to calculate zeta; "
        else:
            message1 = ""
        if event_starts.size < 3:
            message2 = f"Number of events ({event_starts.size}) is too few to calculate zeta; "
        else:
            message2 = ""
        logging.warning("zetatest: " + message1 + message2 + "defaulting to p=1.0")

        return time_vector, rate_vector, ifr_data

    # trial dur
    if max_duration is None:
        max_duration = np.min(np.diff(event_starts))
    else:
        max_duration = np.float64(max_duration)
        assert max_duration.size == 1 and max_duration > 0, "max_duration is not a positive scalar float"

    # %% get difference from uniform
    this_deviation, this_spike_fractions, this_fraction_linear, this_spike_times = get_temporal_offset_one(
        spike_times, event_starts, max_duration)
    spike_number = this_spike_times.size

    # check if sufficient spikes are present
    if this_deviation.size < 3:
        logging.warning("ifr: too few spikes, returning empty variables")

        return time_vector, rate_vector, ifr_data

    # %% get multi-scale derivative
    max_repetitions = event_starts.size
    mean_rate = (spike_number/(max_duration*max_repetitions))
    rate_vector, msd_data = get_multi_scale_deriv(this_spike_times, this_deviation,
                                         smooth_sd=smooth_sd, min_scale=min_scale,
                                         base_value=base_value, mean_rate=mean_rate,
                                         max_duration=max_duration)

    # %% build output
    time_vector = msd_data['timestamps']
    rate_vector = rate_vector  # unnecessary, but for clarity of output building
    ifr_data = dict()
    ifr_data['timestamps'] = time_vector
    ifr_data['rate_vector'] = rate_vector
    ifr_data['deviation_vector'] = this_deviation
    ifr_data['scales'] = msd_data['scales']
    return time_vector, rate_vector, ifr_data
