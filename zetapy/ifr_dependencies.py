# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:21:03 2023

@author: Jorrit
"""

import logging
from scipy.stats import norm
from scipy.signal import convolve2d
import numpy as np
from scipy import stats, signal

def get_multi_scale_deriv(timestamps, values, smooth_sd=2.0, min_scale=None, base_value=1.5, mean_rate=1.0,
                          max_duration=None):
    """
    Calculates the multi-scale derivative of a signal, typically used for instantaneous firing rates.

    Parameters
    ----------
    timestamps : ndarray
        1D array of time points.
    values : ndarray
        1D array of values (e.g., cumulative spike counts).
    smooth_sd : float, optional
        Standard deviation for Gaussian smoothing across time points. Default is 2.0.
    min_scale : float, optional
        The minimum temporal scale for the derivative. If None, it is calculated based on base_value.
    base_value : float, optional
        The base for the exponential scaling of windows. Default is 1.5.
    mean_rate : float, optional
        The target mean rate for rescaling the output vector. Default is 1.0.
    max_duration : float, optional
        The total duration of the trial. If None, it is inferred from timestamps.

    Returns
    -------
    rate_vector : ndarray
        The rescaled instantaneous firing rate vector.
    rate_data : dict
        A dictionary containing intermediate steps: msd_matrix, scales, and mean_derivative.
    """

    # check inputs
    # trial dur
    time_range = np.ptp(timestamps)
    if max_duration is None:
        max_duration = time_range

    # min scale
    if min_scale is None:
        min_scale = round(np.log(1/1000) / np.log(base_value))

    # flatten and reorder timestamps
    timestamps = timestamps.flatten()
    values = values.flatten()
    reorder_indices = np.argsort(timestamps, axis=0)
    timestamps = timestamps[reorder_indices]
    values = values[reorder_indices]
    keep_mask = ~np.logical_or(timestamps == 0, timestamps == max_duration)
    timestamps = timestamps[keep_mask]
    values = values[keep_mask]

    # %% get multi-scale derivative
    max_scale = np.log(time_range/10) / np.log(base_value)
    exponents = np.arange(min_scale, max_scale)
    scales = base_value**exponents
    num_scales = len(scales)
    num_points = len(timestamps)
    msd_matrix = np.zeros((num_points, num_scales))

    for scale_idx, scale in enumerate(scales):
        msd_matrix[:, scale_idx] = calc_single_msd(scale, timestamps, values)

    # %% smoothing
    if smooth_sd > 0:
        smooth_range = 2 * np.ceil(smooth_sd).astype(int)
        filter_kernel = norm.pdf(range(-smooth_range, smooth_range + 1), 0, smooth_sd)
        filter_kernel = filter_kernel / sum(filter_kernel)

        # pad array
        pad_size = np.floor(len(filter_kernel) / 2).astype(int)
        msd_matrix = np.pad(msd_matrix, ((pad_size, pad_size), (0, 0)), 'edge')

        # filter
        msd_matrix = convolve2d(msd_matrix, np.reshape(filter_kernel, (-1, 1)), 'valid')

    # mean
    mean_derivative = np.mean(msd_matrix, axis=1)

    # weighted average of mean_derivative by inter-spike intervals
    mean_of_mean_derivative = (1.0 / max_duration) * sum(((mean_derivative[:-1] + mean_derivative[1:]) / 2.0) * np.diff(timestamps))

    # rescale to real firing rates
    rate_vector = mean_rate * ((mean_derivative + 1.0 / max_duration) / (mean_of_mean_derivative + 1.0 / max_duration))

    # output
    rate_data = dict()
    rate_data['rate_vector'] = rate_vector
    rate_data['timestamps'] = timestamps
    rate_data['mean_derivative'] = mean_derivative
    rate_data['scales'] = scales
    rate_data['msd_matrix'] = msd_matrix
    rate_data['values'] = values
    rate_data['smooth_sd'] = smooth_sd
    rate_data['mean_rate'] = mean_rate

    return rate_vector, rate_data

# %%

def calc_single_msd(scale, timestamps, values):
    num_points = timestamps.size
    msd_vector = np.zeros((num_points,))

    # run through all points
    for point_idx, current_timestamp in enumerate(timestamps):
        # select points within window
        min_edge = current_timestamp - scale / 2
        max_edge = current_timestamp + scale / 2
        min_t_idx = np.searchsorted(timestamps, min_edge, side='right')
        if min_t_idx is None:
            min_t_idx = 0
        max_t_idx = np.searchsorted(timestamps, max_edge, side='right')
        if max_t_idx is None:
            max_t_idx = num_points - 1
        else:
            max_t_idx = max_t_idx - 1

        if (min_t_idx > max_t_idx):
            derivative = 0
        else:
            if (min_t_idx == max_t_idx) and (min_t_idx > 0) and (min_t_idx < (num_points - 1)):
                max_t_idx = min_t_idx + 1
                min_t_idx = min_t_idx - 1

            delta_t = np.max([scale, (timestamps[max_t_idx] - timestamps[min_t_idx])])
            derivative = (values[max_t_idx] - values[min_t_idx]) / delta_t

        # select points within window
        msd_vector[point_idx] = derivative

    # return single msd vector
    return msd_vector

