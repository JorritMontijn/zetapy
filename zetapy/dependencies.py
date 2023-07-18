import logging
import scipy
import collections
import numpy as np
from math import pi, sqrt, exp
from scipy import stats, interpolate, signal
from scipy.signal import convolve, gaussian


def getGumbel(dblE,dblV,dblX):
    """"Calculate p-value and z-score for maximum value of N samples drawn from Gaussian
       [dblP,dblZ] = getGumbel(dblE,dblV,dblX)

        input:
        - dblE: mean of distribution of maximum values
        - dblV: variance of distribution of maximum values
        - dblX: maximum value to express in quantiles of Gumbel

        output:
        - dblP; p-value for dblX (chance that sample originates from distribution given by dblE/dblV)
        - dblZ; z-score corresponding to P

    Version history:
    1.0 - June 17, 2020, Created by Jorrit Montijn translated by Alexander Heimel

    Sources:
    Baglivo (2005), ISBN: 9780898715668
    Elfving (1947), https://doi.org/10.1093/biomet/34.1-2.111
    Royston (1982), DOI: 10.2307/2347982
    https://stats.stackexchange.com/questions/394960/variance-of-normal-order-statistics
    https://stats.stackexchange.com/questions/9001/approximate-order-statistics-for-normal-random-variables
    https://en.wikipedia.org/wiki/Extreme_value_theory
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """

    ## define Gumbel parameters from mean and variance
    #derive beta parameter from variance
    dblBeta = sqrt(6) * sqrt(dblV) / pi

    # define Euler-Mascheroni constant
    dblEulerMascheroni = 0.5772156649015328606065120900824 #vpa(eulergamma)

    # derive mode from mean, beta and E-M constant
    dblMode = dblE - dblBeta * dblEulerMascheroni

    # define Gumbel cdf
    ###    fGumbelCDF = @(x) exp(-exp(-((x(:)-dblMode)./dblBeta)));
    fGumbelCDF = lambda x : exp(-exp(-((x-dblMode) /dblBeta)))

    ## calculate output variables
    # calculate cum dens at X
    dblGumbelCDF = fGumbelCDF(dblX)
    # define p-value
    dblP = 1-dblGumbelCDF
    # transform to output z-score
    ### dblZ = -norminv(dblP/2);
    dblZ = -scipy.stats.norm.ppf(dblP/2)

    # approximation for large X
    ### dblP[isinf(dblZ)] = exp( (dblMode-dblX(isinf(dblZ)))./dblBeta ) ;
    if dblZ>1E30:
        dblP = exp( (dblMode-dblX) / dblBeta )
    # transform to output z-score
    ### dblZ = -norminv(dblP/2);
    dblZ = -scipy.stats.norm.ppf(dblP/2)

    return dblP,dblZ

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def getTempOffset(vecSpikeT, vecSpikeTimes, vecStimUseOnTime, dblUseMaxDur):
    """Calculate temporal offset vectors across folds and offsets.

    Syntax:
    [vecThisDiff,vecThisFrac,vecThisFracLinear] =
        getTempOffset(vecSpikeT,vecSpikeTimes,vecStimUseOnTime,dblUseMaxDur)
    """

    # go through trials to build spike time vector
    cellSpikeTimesPerTrial = []
    for intEvent, dblStartT in enumerate(vecStimUseOnTime):
        # get times
        dblStopT = dblStartT + dblUseMaxDur

        # build trial assignment
        cellSpikeTimesPerTrial.append(vecSpikeTimes[(vecSpikeTimes < dblStopT)
                                                    & (vecSpikeTimes > dblStartT)] - dblStartT)

    # get spikes in fold
    vecThisSpikeT = np.concatenate(cellSpikeTimesPerTrial)

    # get real fractions for training set
    vecThisSpikeTimes = np.sort(np.concatenate(([0], vecThisSpikeT, [dblUseMaxDur])))
    vecThisSpikeFracs = np.linspace(0, 1, vecThisSpikeTimes.shape[0])
    vecThisFrac = interpolate.interp1d(vecThisSpikeTimes, vecThisSpikeFracs)(vecSpikeT)

    # get linear fractions
    vecThisFracLinear = vecSpikeT / dblUseMaxDur

    # calc difference
    vecThisDiff = vecThisFrac - vecThisFracLinear

    return vecThisDiff, vecThisFrac, vecThisFracLinear


def getPeak(vecData, vecT, vecRestrictRange=(-np.inf,np.inf), intSwitchZ=1):
    """Returns highest peak time, width, and location. Syntax:
        [dblPeakValue,dblPeakTime,dblPeakWidth,vecPeakStartStop,intPeakLoc,vecPeakStartStopIdx] = getPeak(vecData,vecT,vecRestrictRange)

    Required input:
        - vecData [N x 1]: values

    Optional inputs:
        - vecT [N x 1]: timestamps corresponding to vecData (default: [1:N])
        - vecRestrictRange: restrict peak to lie within vecRestrictRange(1) and vecRestrictRange(end)

    Outputs:
        - dblPeakTime: time of peak
        - dblPeakWidth: width of peak
        - vecPeakStartStop: start/stop times of peak
        - intPeakLoc: index of peak
        - vecPeakStartStopIdx: start/stop indices of peak

    Version history:
    1.0 - June 19, 2020, Created by Jorrit Montijn, Translated to python by Alexander Heimel
    """

    # check inputs
    if len(vecT) == 0:
        vecT = np.arange(len(vecData))

    # z-score
    if intSwitchZ == 1:
        vecDataZ = stats.zscore(vecData)
    elif intSwitchZ == 2:
        dblMu = np.mean(vecData[vecT<0.02])
        vecDataZ = (vecData - dblMu) / np.std(vecData)
    else:
        vecDataZ = vecData

    # get most prominent POSITIVE peak times
    vecLocsPos, peakProps = signal.find_peaks(vecDataZ, threshold=0, prominence=-np.inf)
    vecValsPos = vecDataZ[vecLocsPos]
    vecPromsPos = peakProps['prominences']

    # remove peaks outside window
    indKeepPeaks = (vecT[vecLocsPos] >= vecRestrictRange[0]) & (vecT[vecLocsPos] <= vecRestrictRange[1])

    if np.sum(indKeepPeaks) == 0:
        dblMaxPosVal = None
    else:
        # select peak
        vecValsPos = vecValsPos[indKeepPeaks]
        vecLocsPos = vecLocsPos[indKeepPeaks]
        vecPromsPos = vecPromsPos[indKeepPeaks]
        intPosIdx = np.argmax(vecValsPos)
        dblMaxPosVal = vecValsPos[intPosIdx]

    # get most prominent NEGATIVE peak times
    vecLocsNeg, peakProps = signal.find_peaks(-vecDataZ, threshold=0, prominence=-np.inf)
    vecValsNeg = -vecDataZ[vecLocsNeg]
    vecPromsNeg = peakProps['prominences']

    # remove peaks outside window
    indKeepPeaks = (vecT[vecLocsNeg] >= vecRestrictRange[0]) & (vecT[vecLocsNeg] <= vecRestrictRange[1])

    if np.sum(indKeepPeaks) == 0:
        dblMaxNegVal = None
    else:
        # select peak
        vecValsNeg = vecValsNeg[indKeepPeaks]
        vecLocsNeg = vecLocsNeg[indKeepPeaks]
        vecPromsNeg = vecPromsNeg[indKeepPeaks]
        intNegIdx = np.argmax(vecValsNeg)
        dblMaxNegVal = vecValsNeg[intNegIdx]

    if dblMaxPosVal is None and dblMaxNegVal is None :
        indPeakMembers = None
    elif ((dblMaxPosVal is not None and dblMaxNegVal is None)
          or (dblMaxPosVal is not None and (np.abs(dblMaxPosVal) >= np.abs(dblMaxNegVal)))):
        intIdx = intPosIdx
        intPeakLoc = vecLocsPos[intIdx]
        dblPeakProm = vecPromsPos[intIdx]
        dblCutOff = vecDataZ[intPeakLoc] - dblPeakProm / 2
        indPeakMembers = (vecDataZ > dblCutOff)
    elif ((dblMaxPosVal is None and dblMaxNegVal is not None)
          or (dblMaxNegVal is not None and (np.abs(dblMaxPosVal) < np.abs(dblMaxNegVal)))):
        intIdx = intNegIdx
        intPeakLoc = vecLocsNeg[intIdx]
        dblPeakProm = vecPromsNeg[intIdx]
        dblCutOff = vecDataZ[intPeakLoc] + dblPeakProm / 2
        indPeakMembers = (vecDataZ < dblCutOff)

    if indPeakMembers is not None:
        # get potential starts/stops
        ### vecPeakStarts = find(diff(indPeakMembers)==1);
        vecPeakStarts = np.where(np.diff([float(f) for f in indPeakMembers])==1)[0]
        ### vecPeakStops = find(diff(indPeakMembers)==-1);
        vecPeakStops = np.where(np.diff([float(f) for f in indPeakMembers])==-1)[0]
        if indPeakMembers[0] == True:
            ### vecPeakStarts = [1 vecPeakStarts(:)'];
            vecPeakStarts = [0] + vecPeakStarts
        ### if indPeakMembers(end) == 1
        if indPeakMembers[-1] == True:
            ### vecPeakStops = [vecPeakStops(:)' numel(indPeakMembers)];
            vecPeakStops = vecPeakStops + [len(indPeakMembers)-1]

        # find closest points
        ###    intPeakStart = intPeakLoc-min(intPeakLoc - vecPeakStarts(vecPeakStarts<intPeakLoc));

        intPeakStart = intPeakLoc - np.min(intPeakLoc - vecPeakStarts[vecPeakStarts<intPeakLoc])
        intPeakStop = intPeakLoc + np.min(vecPeakStops[vecPeakStops>=intPeakLoc] - intPeakLoc)
        dblPeakStartT = vecT[intPeakStart]
        if intPeakStop >= vecT.shape[0]:
            intPeakStop = vecT.shape[0] - 1
        dblPeakStopT = vecT[intPeakStop]
        # assign peak data
        dblPeakValue = vecData[intPeakLoc]
        dblPeakTime = vecT[intPeakLoc]
        dblPeakWidth = dblPeakStopT - dblPeakStartT
        vecPeakStartStop = [dblPeakStartT, dblPeakStopT]
        vecPeakStartStopIdx = [intPeakStart, intPeakStop]
    else:
        # assign placeholder peak data
        dblPeakValue = np.nan
        dblPeakTime = np.nan
        dblPeakWidth = np.nan
        vecPeakStartStop = [np.nan, np.nan]
        intPeakLoc = None
        vecPeakStartStopIdx = [None, None]

    return dblPeakValue, dblPeakTime, dblPeakWidth, vecPeakStartStop, intPeakLoc, vecPeakStartStopIdx


def getOnset(vecData,vecT,dblPeakT,vecRestrictRange,intSwitchZ=1):
    """Returns peak onset. Syntax:
        [dblOnset,dblValue] = getOnset(vecData,vecT,dblPeakT,vecRestrictRange)

    Required input:
        - vecData [N x 1]: values

    Optional inputs:
        - vecT [N x 1]: timestamps corresponding to vecData (default: [1:N])
        - dblPeakT (float): timestamp corresponding to peak
        - vecRestrictRange [2 x 1]: restrict peak to lie within vecRestrictRange(1) and vecRestrictRange(end)

    Outputs:
        - dblOnset: time of peak onset (first crossing half-height of peak)
        - dblValue: value at peak onset

    Version history:
    1.0 - June 19, 2020 Created by Jorrit Montijn, Translated to Python by Alexander Heimel
    """

    ##
    # check inputs
    if vecT == []:
        vecT = np.arange(len(vecData))

    if vecRestrictRange == None:
        ### vecRestrictRange = [min(vecT) min(vecT)+range(vecT)/2];
        vecRestrictRange = (np.min(vecT), np.min(vecT) + (np.max(vecT)-np.min(vecT))/2)

    # z-score
    if intSwitchZ == 1:
        vecDataZ = stats.zscore(vecData)
    elif intSwitchZ == 2:
        dblMu = np.mean(vecData[vecT<0.02])
        vecDataZ = (vecData - dblMu) / np.std(vecData)
    else:
        vecDataZ = vecData

    if dblPeakT==None:
        dblPeakT = getPeak(vecDataZ, vecT, vecRestrictRange, 0)[1]

    # remove time points outside restricted range
    indRemove = (vecT < vecRestrictRange[0]) | (vecT > vecRestrictRange[1])
    vecCropT = vecT[np.invert(indRemove)]
    vecDataZ = vecDataZ[np.invert(indRemove)]

    # calculate first timepoint crossing half-height of peak
    intPeakIdx = np.argmin(abs(vecCropT-dblPeakT))

    dblPeakVal = vecDataZ[intPeakIdx]
    dblBaseVal = vecDataZ[0]
    dblThresh = (dblPeakVal - dblBaseVal)/2 + dblBaseVal
    ### if dblThresh > 0
    ###     intOnsetIdx = find(vecDataZ >= dblThresh,1,'first');
    ### else
    ###     intOnsetIdx = find(vecDataZ <= dblThresh,1,'first');
    ### end
    if dblThresh > 0:
        intOnsetIdx = np.where(vecDataZ >= dblThresh)[0]
    else:
        intOnsetIdx = np.where(vecDataZ <= dblThresh)[0]

    if len(intOnsetIdx)>0:
        intOnsetIdx = intOnsetIdx[0]
        dblOnset = vecCropT[intOnsetIdx]
        dblValue = vecData[vecT > dblOnset]
    else:
        dblOnset = None
        dblValue = None

    return dblOnset, dblValue, dblBaseVal, dblPeakT


def calculatePeths(
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
    n_offset = 5 * int(np.ceil(smoothing / bin_size))  # get rid of boundary effects for smoothing
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    # build gaussian kernel if requested
    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        # half (causal) gaussian filter
        # window[int(np.ceil(w/2)):] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    # filter spikes outside of the loop
    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    # compute floating tscale
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    # bin spikes
    for i, t_0 in enumerate(align_times):
        # define bin edges
        ts = tscale + t_0
        # filter spikes
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        # store (ts represent bin edges, so there are one fewer bins)
        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        # smooth
        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

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
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    # package output
    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = dict({'means': peth_means, 'stds': peth_stds, 'sems': peth_sems,
                  'tscale': tscale, 'cscale': ids})
    return peths, binned_spikes