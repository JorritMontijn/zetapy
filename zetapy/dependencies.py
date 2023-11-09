# -*- coding: utf-8 -*-
import numpy as np
import logging
from scipy import stats
from math import pi, sqrt, exp, factorial
from collections.abc import Iterable
from numba import njit

# %% calcZetaTwo


def calcZetaTwo(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2, dblUseMaxDur, intResampNum, boolDirectQuantile):
    """
    Calculates two-sample zeta
    dZETA = calcZetaTwo(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2,
                        vecEventStarts2, dblUseMaxDur, intResampNum, boolDirectQuantile)
    dZETA has entries:
        vecSpikeT, vecRealDeviation, vecRealFrac, vecRealFracLinear, cellRandTime, cellRandDeviation, dblZetaP, dblZETA, intZETAIdx
    """

    # %% pre-allocate output
    vecSpikeT = None
    vecRealDiff = None
    vecRealFrac1 = None
    vecRealFrac2 = None
    cellRandTime = None
    cellRandDiff = None
    dblZetaP = 1.0
    dblZETA = 0.0
    intZETAIdx = None

    dZETA = dict()
    dZETA['vecSpikeT'] = vecSpikeT
    dZETA['vecRealDiff'] = vecRealDiff
    dZETA['vecRealFrac1'] = vecRealFrac1
    dZETA['vecRealFrac2'] = vecRealFrac2
    dZETA['cellRandTime'] = cellRandTime
    dZETA['cellRandDiff'] = cellRandDiff
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = dblZETA
    dZETA['intZETAIdx'] = intZETAIdx

    # %% ensure input is correct
    # assert that arrEventTimes is a 1D array of floats
    assert len(arrEventTimes1.shape) < 3 and len(arrEventTimes2.shape) < 3 \
        and issubclass(arrEventTimes1.dtype.type, np.floating) and issubclass(arrEventTimes2.dtype.type, np.floating), \
        "Input arrEventTimes1 or arrEventTimes2 is not a 1D or 2D float np.array"

    # ensure orientation arrEventTimes1
    if len(arrEventTimes1.shape) > 1:
        if arrEventTimes1.shape[1] < 3:
            pass
        elif arrEventTimes1.shape[0] < 3:
            arrEventTimes1 = arrEventTimes1.T
        else:
            raise Exception(
                "Input error: arrEventTimes1 must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        arrEventTimes1 = np.reshape(arrEventTimes1, (-1, 1))

    # ensure orientation arrEventTimes2
    if len(arrEventTimes2.shape) > 1:
        if arrEventTimes2.shape[1] < 3:
            pass
        elif arrEventTimes2.shape[0] < 3:
            arrEventTimes2 = arrEventTimes2.T
        else:
            raise Exception(
                "Input error: arrEventTimes2 must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        arrEventTimes2 = np.reshape(arrEventTimes2, (-1, 1))

    # %% get spikes per trial
    vecEventT1 = arrEventTimes1[:, 0]
    vecEventT2 = arrEventTimes2[:, 0]
    cellTrialPerSpike1, cellTimePerSpike1 = getSpikesInTrial(vecSpikeTimes1, vecEventT1, dblUseMaxDur)
    cellTrialPerSpike2, cellTimePerSpike2 = getSpikesInTrial(vecSpikeTimes2, vecEventT2, dblUseMaxDur)

    # %% run normal
    # normalize to cumsum(v1)+cumsum(v2) = 1
    # take difference
    # mean-subtract

    # get difference
    vecSpikeT, vecRealDiff, vecRealFrac1, vecThisSpikeTimes1, vecRealFrac2, vecThisSpikeTimes2 = \
        getTempOffsetTwo(cellTimePerSpike1, cellTimePerSpike2, dblUseMaxDur)

    if len(vecRealDiff) < 2:
        return dZETA

    intZETAIdx = np.argmax(np.abs(vecRealDiff))
    dblMaxD = np.abs(vecRealDiff[intZETAIdx])

    # repeat procedure, but swap trials randomly in each resampling
    cellRandTime = [None] * intResampNum
    cellRandDiff = [None] * intResampNum
    vecMaxRandD = np.empty((intResampNum, 1))
    vecMaxRandD.fill(np.nan)

    cellAggregateTrials = cellTimePerSpike1 + cellTimePerSpike2
    intTrials1 = len(cellTimePerSpike1)
    intTrials2 = len(cellTimePerSpike2)
    intTotTrials = intTrials1+intTrials2

    # %% run bootstraps; try parallel, otherwise run normal loop
    # repeat procedure, but swap trials randomly in each resampling
    for intResampling in range(intResampNum):
        # %% get random subsample
        # if cond1 has 10 trials, and cond2 has 100, then:
        # for shuffle of cond1: take 10 trials from set of 110
        # for shuffle of cond2: take 100 trials from set of 110
        vecUseRand1 = np.random.randint(intTotTrials, size=intTrials1)
        vecUseRand2 = np.random.randint(intTotTrials, size=intTrials2)

        cellTimePerSpike1_Rand = [cellAggregateTrials[i] for i in vecUseRand1]
        cellTimePerSpike2_Rand = [cellAggregateTrials[j] for j in vecUseRand2]

        if np.sum([len(xi) for xi in cellTimePerSpike1_Rand]) == 0 and np.sum([len(yi) for yi in cellTimePerSpike2_Rand]) == 0:
            dblAddVal = None
        else:
            # get difference
            vecRandT, vecRandDiff, vecRandFrac1, vecThisSpikeTimes1, vecRandFrac2, vecThisSpikeTimes2 = \
                getTempOffsetTwo(cellTimePerSpike1_Rand, cellTimePerSpike2_Rand, dblUseMaxDur)

            # assign data
            cellRandTime[intResampling] = vecRandT
            cellRandDiff[intResampling] = vecRandDiff
            dblAddVal = np.max(np.abs(vecRandDiff))

        # assign read-out
        if dblAddVal is None or dblAddVal == 0:
            dblAddVal = dblMaxD
        vecMaxRandD[intResampling] = dblAddVal

    # %% calculate significance
    dblZetaP, dblZETA = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)

    # %% return
    dZETA = dict()
    dZETA['vecSpikeT'] = vecSpikeT
    dZETA['vecRealDiff'] = vecRealDiff
    dZETA['vecRealFrac1'] = vecRealFrac1
    dZETA['vecRealFrac2'] = vecRealFrac2
    dZETA['cellRandTime'] = cellRandTime
    dZETA['cellRandDiff'] = cellRandDiff
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = dblZETA
    dZETA['intZETAIdx'] = intZETAIdx
    return dZETA

# %%


def calcZetaOne(vecSpikeTimes, arrEventTimes, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch, boolParallel):
    """
    Calculates neuronal responsiveness index zeta
    dZETA = calcZetaOne(vecSpikeTimes, vecEventStarts, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch,boolParallel, intUseJitterDistro)
    dZETA has entries:
        vecSpikeT, vecRealDeviation, vecRealFrac, vecRealFracLinear, cellRandTime, cellRandDeviation, dblZetaP, dblZETA, intZETAIdx
    """

    # %% pre-allocate output
    vecSpikeT = None
    vecRealDeviation = None
    vecRealFrac = None
    vecRealFracLinear = None
    cellRandTime = None
    cellRandDeviation = None
    dblZetaP = 1.0
    dblZETA = 0.0
    intZETAIdx = None

    dZETA = dict()
    dZETA['vecSpikeT'] = vecSpikeT
    dZETA['vecRealDeviation'] = vecRealDeviation
    dZETA['vecRealFrac'] = vecRealFrac
    dZETA['vecRealFracLinear'] = vecRealFracLinear
    dZETA['cellRandTime'] = cellRandTime
    dZETA['cellRandDeviation'] = cellRandDeviation
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = dblZETA
    dZETA['intZETAIdx'] = intZETAIdx

    # %% prep parallel processing
    # to do

    # %% reduce spikes
    # ensure orientation and assert that arrEventTimes is a 1D array of floats
    assert len(arrEventTimes.shape) < 3 and issubclass(
        arrEventTimes.dtype.type, np.floating), "Input arrEventTimes is not a 1D or 2D float np.array"
    if len(arrEventTimes.shape) > 1:
        if arrEventTimes.shape[1] < 3:
            pass
        elif arrEventTimes.shape[0] < 3:
            arrEventTimes = arrEventTimes.T
        else:
            raise Exception(
                "Input error: arrEventTimes must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        arrEventTimes = np.reshape(arrEventTimes, (-1, 1))
    # define event starts
    vecEventT = arrEventTimes[:, 0]

    dblMinPreEventT = np.min(vecEventT)-dblUseMaxDur*5*dblJitterSize
    dblStartT = max([vecSpikeTimes[0], dblMinPreEventT])
    dblStopT = max(vecEventT)+dblUseMaxDur*5*dblJitterSize
    vecSpikeTimes = vecSpikeTimes[np.logical_and(vecSpikeTimes >= dblStartT, vecSpikeTimes <= dblStopT)]

    if vecSpikeTimes.size < 3:
        logging.warning(
            "calcZetaOne:vecSpikeTimes: too few spikes around events to calculate zeta")
        return dZETA

    # %% build pseudo data, stitching stimulus periods
    if boolStitch:
        vecPseudoSpikeTimes, vecPseudoEventT = getPseudoSpikeVectors(vecSpikeTimes, vecEventT, dblUseMaxDur)
    else:
        vecPseudoSpikeTimes = vecSpikeTimes
        vecPseudoEventT = vecEventT

    # %% run normal
    # get data
    vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT = getTempOffsetOne(
        vecPseudoSpikeTimes, vecPseudoEventT, dblUseMaxDur)

    if vecRealDeviation.size < 3:
        logging.warning(
            "calcZetaOne:vecRealDeviation: too few spikes around events to calculate zeta")
        return dZETA

    vecRealDeviation = vecRealDeviation - np.mean(vecRealDeviation)
    intZETAIdx = np.argmax(np.abs(vecRealDeviation))
    dblMaxD = np.abs(vecRealDeviation[intZETAIdx])

    # %% create random jitters
    # run pre-set number of iterations
    cellRandTime = []
    cellRandDeviation = []
    vecMaxRandD = np.empty((intResampNum, 1))
    vecMaxRandD.fill(np.nan)

    vecStartOnly = np.reshape(vecPseudoEventT, (-1, 1))
    intTrials = vecStartOnly.size
    matJitterPerTrial = np.empty((intTrials, intResampNum))
    matJitterPerTrial.fill(np.nan)

    # uniform jitters between dblJitterSize*[-tau, +tau]
    for intResampling in range(intResampNum):
        matJitterPerTrial[:, intResampling] = dblJitterSize*dblUseMaxDur * \
            ((np.random.rand(vecStartOnly.shape[0]) - 0.5) * 2)

    # %% this part is only to check if matlab and python give the same exact results
    # unfortunately matlab's randperm() and numpy's np.random.permutation give different outputs even with
    # identical seeds and identical random number generators, so I've had to load in a table of random values here...
    boolTest = False
    if boolTest:
        from scipy.io import loadmat
        print('Loading deterministic jitter data for comparison with matlab')
        logging.warning(
            "calcZetaOne:debugMode: set boolTest to False to suppress this warning")
        dLoad = loadmat('matJitterPerTrial.mat')
        matJitterPerTrial = dLoad['matJitterPerTrial']

        # reset rng
        np.random.seed(1)

    # %% run resamplings
    for intResampling in range(intResampNum):
        # get random subsample
        vecStimUseOnTime = vecStartOnly[:, 0] + matJitterPerTrial[:, intResampling].T

        # get temp offset
        vecRandDiff, vecThisSpikeFracs, vecThisFracLinear, vecThisSpikeTimes = getTempOffsetOne(
            vecPseudoSpikeTimes, vecStimUseOnTime, dblUseMaxDur)

        # assign data
        cellRandTime.append(vecThisSpikeTimes)
        cellRandDeviation.append(vecRandDiff - np.mean(vecRandDiff))
        vecMaxRandD[intResampling] = np.max(np.abs(cellRandDeviation[intResampling]))

    # %% calculate significance
    dblZetaP, dblZETA = getZetaP(dblMaxD, vecMaxRandD, boolDirectQuantile)

    # %% assign output
    dZETA = dict()
    dZETA['vecSpikeT'] = vecSpikeT
    dZETA['vecRealDeviation'] = vecRealDeviation
    dZETA['vecRealFrac'] = vecRealFrac
    dZETA['vecRealFracLinear'] = vecRealFracLinear
    dZETA['cellRandTime'] = cellRandTime
    dZETA['cellRandDeviation'] = cellRandDeviation
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = dblZETA
    dZETA['intZETAIdx'] = intZETAIdx
    return dZETA

# %%


def getTempOffsetTwo(cellTimePerSpike1, cellTimePerSpike2, dblUseMaxDur):
    '''
    vecSpikeT, vecThisDiff, vecThisFrac1, vecThisSpikeTimes1, vecThisFrac2, vecThisSpikeTimes2 = 
        getTempOffsetTwo(cellTimePerSpike1,cellTimePerSpike2,dblUseMaxDur)
    '''

    # introduce minimum jitter to identical spikes
    vecSpikes1 = flatten(cellTimePerSpike1)
    vecSpikes2 = flatten(cellTimePerSpike2)
    
    vecThisSpikeTimes1 = getUniqueSpikes(np.sort(vecSpikes1))
    vecThisSpikeTimes2 = getUniqueSpikes(np.sort(vecSpikes2))

    # ref time
    vecSpikeT = np.sort(np.concatenate((np.zeros(1), vecThisSpikeTimes1,
                        vecThisSpikeTimes2, np.array([dblUseMaxDur])), axis=0))

    # cond1 goes to S1_n/T1_n; cond2 goes to S2_n/T2_n
    intSp1 = len(vecThisSpikeTimes1)
    intSp2 = len(vecThisSpikeTimes2)
    intT1 = len(cellTimePerSpike1)
    intT2 = len(cellTimePerSpike2)

    # spike fraction #1
    vecUniqueSpikeFracs1 = np.linspace(1, vecThisSpikeTimes1.size, vecThisSpikeTimes1.size)/intT1
    vecSpikes1 = np.concatenate((np.zeros(1), vecThisSpikeTimes1, np.array([dblUseMaxDur])), axis=0)
    vecFracs1 = np.concatenate((np.zeros(1), vecUniqueSpikeFracs1, np.array([intSp1/intT1])), axis=0)
    vecThisFrac1 = np.interp(vecSpikeT,
                             vecSpikes1,
                             vecFracs1,
                             1/intT1, intSp1/intT1)

    # spike fraction #2
    vecUniqueSpikeFracs2 = np.linspace(1, vecThisSpikeTimes2.size, vecThisSpikeTimes2.size)/intT2
    vecSpikes2 = np.concatenate((np.zeros(1), vecThisSpikeTimes2, np.array([dblUseMaxDur])), axis=0)
    vecFracs2 = np.concatenate((np.zeros(1), vecUniqueSpikeFracs2, np.array([intSp2/intT2])), axis=0)
    vecThisFrac2 = np.interp(vecSpikeT,
                             vecSpikes2,
                             vecFracs2,
                             1/intT2, intSp2/intT2)

    # take difference
    vecDeviation = vecThisFrac1 - vecThisFrac2

    # mean-subtract?
    vecThisDiff = vecDeviation - np.mean(vecDeviation)

    return vecSpikeT, vecThisDiff, vecThisFrac1, vecThisSpikeTimes1, vecThisFrac2, vecThisSpikeTimes2

# %%


def getSpikesInTrial(vecSpikes, vecTrialStarts, dblMaxDur):
    '''
    getSpikesInTrial Retrieves spiking times per trial
    syntax: cellTrialPerSpike,cellTimePerSpike = getSpikesInTrial(vecSpikes,vecTrialStarts,dblMaxDur)
    input:
        - vecSpikes; spike times (s)
        - vecTrialStarts: trial start times (s)
        - dblTrialDur: trial duration (s)
    returns:
        - cellTrialPerSpike
        - cellTimePerSpike
    '''

    # loop
    intTrials = len(vecTrialStarts)
    cellTrialPerSpike = []
    cellTimePerSpike = []
    for intTrial, dblStartT in enumerate(vecTrialStarts):
        # get spikes
        vecTheseSpikes = vecSpikes[np.logical_and(vecSpikes >= dblStartT, vecSpikes < (dblStartT + dblMaxDur))] - dblStartT

        # assign
        cellTrialPerSpike.append(intTrial*np.ones(len(vecTheseSpikes)))
        cellTimePerSpike.append(vecTheseSpikes)

    return cellTrialPerSpike, cellTimePerSpike

# %%


def getZetaP(arrMaxD, vecMaxRandD, boolDirectQuantile):
    # %% calculate significance
    # find highest peak and retrieve value
    vecMaxRandD = np.sort(np.unique(vecMaxRandD), axis=0)
    if not isinstance(arrMaxD, Iterable):
        arrMaxD = np.array([arrMaxD])

    if boolDirectQuantile:
        # calculate statistical significance using empirical quantiles
        # define p-value
        arrZetaP = np.empty(arrMaxD.size)
        arrZetaP.fill(np.nan)
        for i, d in enumerate(arrMaxD):
            if d < np.min(vecMaxRandD) or np.isnan(d):
                dblValue = 0
            elif d > np.max(vecMaxRandD) or np.isinf(d):
                dblValue = vecMaxRandD.size
            else:
                dblValue = np.interp(
                    d, vecMaxRandD, np.arange(0, vecMaxRandD.size)+1)

            arrZetaP[i] = 1 - (dblValue/(1+vecMaxRandD.size))

        # transform to output z-score
        arrZETA = -stats.norm.ppf(arrZetaP/2)
    else:
        # calculate statistical significance using Gumbel distribution
        arrZetaP, arrZETA = getGumbel(
            np.mean(vecMaxRandD), np.var(vecMaxRandD, ddof=1), arrMaxD)  # default ddof for numpy var() is incorrect

    # return
    if arrZetaP.size == 1:
        arrZetaP = arrZetaP[0]
    if arrZETA.size == 1:
        arrZETA = arrZETA[0]
    return arrZetaP, arrZETA

# %%


def getGumbel(dblE, dblV, arrX):
    """"Calculate p-value and z-score for maximum value of N samples drawn from Gaussian
           dblP,dblZ = getGumbel(dblE,dblV,arrX)

                input:
                - dblE: mean of distribution of maximum values
                - dblV: variance of distribution of maximum values
                - arrX: maximum value to express in quantiles of Gumbel

                output:
                - arrP; p-value for dblX (chance that sample originates from distribution given by dblE/dblV)
                - arrZ; z-score corresponding to P

        Version history:
        1.0 - June 17, 2020
        Created by Jorrit Montijn, translated by Alexander Heimel
        3.0 - August 17 2023
        New translation to Python by Jorrit Montijn: Now supports array input of arrX

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
    dblEulerMascheroni = 0.5772156649015328606065120900824  # vpa(eulergamma)

    # %% define Gumbel parameters from mean and variance
    # derive beta parameter from variance
    dblBeta = (sqrt(6)*sqrt(dblV))/(pi)

    # derive mode from mean, beta and E-M constant
    dblMode = dblE - dblBeta*dblEulerMascheroni

    # define Gumbel cdf
    def fGumbelCDF(x): return np.exp(-np.exp(-((x-dblMode) / dblBeta)))

    # %% calculate output variables
    # calculate cum dens at X
    arrGumbelCDF = fGumbelCDF(arrX)

    # define p-value
    arrP = 1-arrGumbelCDF

    # transform to output z-score
    arrZ = -stats.norm.ppf(np.divide(arrP, 2))

    # approximation for large X
    for i, dblZ in enumerate(arrZ):
        if np.isinf(dblZ):
            arrP[i] = exp(dblMode-arrX[i] / dblBeta)
            arrZ[i] = -stats.norm.ppf(arrP[i]/2)

    # return
    return arrP, arrZ

# %%


def getTempOffsetOne(vecSpikeTimes, vecEventTimes, dblUseMaxDur):
    # %% get temp diff vector
    # pre-allocate
    vecSpikesInTrial = getSpikeT(vecSpikeTimes, vecEventTimes, dblUseMaxDur)

    # introduce minimum jitter to identical spikes
    vecThisSpikeTimes = getUniqueSpikes(vecSpikesInTrial)

    # turn into fractions
    vecThisSpikeFracs = np.linspace(
        1/vecThisSpikeTimes.size, 1, vecThisSpikeTimes.size)

    # get linear fractions
    vecThisFracLinear = vecThisSpikeTimes/dblUseMaxDur

    # calc difference
    vecThisDeviation = vecThisSpikeFracs - vecThisFracLinear
    vecThisDeviation = vecThisDeviation - np.mean(vecThisDeviation)

    return vecThisDeviation, vecThisSpikeFracs, vecThisFracLinear, vecThisSpikeTimes

# %%


def getUniqueSpikes(vecSpikesInTrial):
    # introduce random minimum jitter to identical spikes
    dblMinOffset = np.finfo(vecSpikesInTrial.dtype.type).eps
    vecOffsets = np.arange(-dblMinOffset*10, dblMinOffset*10, dblMinOffset)
    vecUniqueSpikes, vecIdx = np.unique(vecSpikesInTrial, return_index=True)
    while vecUniqueSpikes.shape[0] != vecSpikesInTrial.shape[0]:
        indDuplicates = ~np.isin(np.arange(vecSpikesInTrial.shape[0]), vecIdx)
        vecRandomOffsets = np.random.choice(vecOffsets, np.sum(indDuplicates))
        vecSpikesInTrial[indDuplicates] += vecRandomOffsets
        vecUniqueSpikes, vecIdx = np.unique(vecSpikesInTrial, return_index=True)

    return vecSpikesInTrial

# %%
@njit

def getSpikeT(vecSpikeTimes, vecEventTimes, dblUseMaxDur):
    # sorted vec of spike times relative to recording start
    # are expressed as times relative to the preceeding event time
    # vecSpikeTimes should be sorted.
    
    # use np.searchsorted here for performance.
    # find the indexes into vecSpikeTimes that vecEventTimes occur at
    Sidx = np.searchsorted(vecSpikeTimes, vecEventTimes)
    # do same for event 'ends'
    Eidx = np.searchsorted(vecSpikeTimes, vecEventTimes+dblUseMaxDur)
    # use the start and stop indexs to pre-index right size array.
    # add two extra slots for the beginning [0] and end [dblUseMaxDur]
    vecSpikesInTrial = np.empty(np.sum(Eidx-Sidx)+2)
    vecSpikesInTrial[0] = 0
    vecSpikesInTrial[-1] = dblUseMaxDur
    # loop over trials to fill spike time vector
    cnt = 1
    for i,dblStartT in enumerate(vecEventTimes):
        vecSpikesInTrial[cnt:cnt+Eidx[i]-Sidx[i]]=\
            vecSpikeTimes[Sidx[i]:Eidx[i]]-dblStartT
        cnt+=Eidx[i]-Sidx[i]
    return np.sort(vecSpikesInTrial)
# %%


def getSpikeT_old(vecSpikeTimes, vecEventTimes, dblUseMaxDur):
    # %% turn spike times relative to recording start into times relative to trial start

    # pre-allocate
    vecSpikesInTrial = np.empty((vecSpikeTimes.size*2))
    vecSpikesInTrial.fill(np.nan)
    intIdx = 0

    # go through trials to build spike time vector
    for dblStartT in vecEventTimes:
        # get times
        dblStopT = dblStartT + dblUseMaxDur

        # build trial assignment
        vecTempSpikes = vecSpikeTimes[np.logical_and(vecSpikeTimes < dblStopT, vecSpikeTimes > dblStartT)] - dblStartT
        intTempSpikeNr = vecTempSpikes.size
        vecAssignIdx = [i for i in range(intIdx, intIdx+intTempSpikeNr)]
        if len(vecAssignIdx) > 0 and vecAssignIdx[-1] >= vecSpikesInTrial.size:
            vecSpikesInTrial = np.resize(vecSpikesInTrial, vecSpikesInTrial.size*2)
        vecSpikesInTrial[vecAssignIdx] = vecTempSpikes
        intIdx = intIdx + intTempSpikeNr

    # remove trailing nan entries
    vecSpikesInTrial = vecSpikesInTrial[:intIdx]

    # sort spikes in window and add start/end entries
    vecSpikesInTrial = np.concatenate((np.zeros(1), np.sort(vecSpikesInTrial, axis=0), np.array([dblUseMaxDur])))

    return vecSpikesInTrial

# %%


def getPseudoSpikeVectors(vecSpikeTimes, vecEventTimes, dblWindowDur, boolDiscardEdges=False):
    # %% prep
    # ensure sorting and alignment
    vecSpikeTimes = np.sort(np.reshape(vecSpikeTimes, (-1, 1)), axis=0)
    vecEventTimes = np.sort(np.reshape(vecEventTimes, (-1, 1)), axis=0)

    # %% pre-allocate
    intSamples = vecSpikeTimes.size
    intTrials = vecEventTimes.size
    dblMedianDur = np.median(np.diff(vecSpikeTimes, axis=0))
    cellPseudoSpikeT = []
    vecPseudoEventT = np.empty((intTrials, 1))
    vecPseudoEventT.fill(np.nan)
    dblPseudoEventT = 0.0
    intLastUsedSample = 0
    intFirstSample = None

    # run
    for intTrial, dblEventT in enumerate(vecEventTimes):
        # get eligible samples
        intStartSample = findfirst(vecSpikeTimes >= dblEventT)
        intEndSample = findfirst(vecSpikeTimes > (dblEventT+dblWindowDur))

        if intStartSample is not None and intEndSample is not None and intStartSample > intEndSample:
            intEndSample = None
            intStartSample = None

        if intEndSample is None:
            intEndSample = len(vecSpikeTimes)

        if intStartSample is None or intEndSample is None:
            vecUseSamples = np.empty(0, dtype=int)
        else:
            intEndSample = intEndSample - 1
            vecEligibleSamples = np.arange(intStartSample, intEndSample+1)
            indUseSamples = np.logical_and(vecEligibleSamples >= 0, vecEligibleSamples < intSamples)
            vecUseSamples = vecEligibleSamples[indUseSamples]

        # check if beginning or end
        if vecUseSamples.size > 0:
            if intTrial == 0 and not boolDiscardEdges:
                vecUseSamples = np.arange(0, vecUseSamples[-1]+1)
            elif intTrial == (intTrials-1) and not boolDiscardEdges:
                vecUseSamples = np.arange(vecUseSamples[0], intSamples)

        # add spikes
        if vecUseSamples.size > 0:
            vecAddT = vecSpikeTimes[vecUseSamples]
            indOverlap = vecUseSamples <= intLastUsedSample

        # get event t
        if intTrial == 0:
            dblPseudoEventT = 0.0
        else:
            if intTrial > 0 and dblWindowDur > (dblEventT - vecEventTimes[intTrial-1]):
                # remove spikes from overlapping epochs
                if vecUseSamples.size > 0:
                    vecUseSamples = vecUseSamples[~indOverlap]
                    vecAddT = vecSpikeTimes[vecUseSamples]

                dblPseudoEventT = dblPseudoEventT + dblEventT - vecEventTimes[intTrial-1]
            else:
                dblPseudoEventT = dblPseudoEventT + dblWindowDur

        # %% make local pseudo event time
        if vecUseSamples.size == 0:
            vecLocalPseudoT = np.empty(0)
        else:
            intLastUsedSample = vecUseSamples[-1]
            vecLocalPseudoT = vecAddT - dblEventT + dblPseudoEventT

        if intFirstSample is None and vecUseSamples.size > 0:
            intFirstSample = vecUseSamples[0]
            dblPseudoT0 = dblPseudoEventT

        # assign data for this trial
        cellPseudoSpikeT.append(vecLocalPseudoT)
        vecPseudoEventT[intTrial] = dblPseudoEventT

    # %% add beginning
    if not boolDiscardEdges and intFirstSample is not None and intFirstSample > 0:
        dblStepBegin = vecSpikeTimes[intFirstSample] - vecSpikeTimes[intFirstSample-1]
        vecSampAddBeginning = np.arange(0, intFirstSample)
        vecAddBeginningSpikes = vecSpikeTimes[vecSampAddBeginning] - vecSpikeTimes[vecSampAddBeginning[0]] \
            + dblPseudoT0 - dblStepBegin - \
            np.ptp(vecSpikeTimes[vecSampAddBeginning]
                   )  # make local to first spike in array, then preceding pseudo event t0
        cellPseudoSpikeT.append(vecAddBeginningSpikes)

    # %% add end
    intTn = vecSpikeTimes.size
    intLastUsedSample = findfirst(vecSpikeTimes > (vecEventTimes[-1]+dblWindowDur))
    if not boolDiscardEdges and intLastUsedSample is not None and (intTn-1) > intLastUsedSample:
        vecSampAddEnd = np.arange(intLastUsedSample, intTn)
        vecAddEndSpikes = vecSpikeTimes[vecSampAddEnd] - dblEventT + dblPseudoEventT + dblWindowDur
        cellPseudoSpikeT.append(vecAddEndSpikes)

    # %% recombine into vector
    vecPseudoSpikeTimes = np.array(sorted(flatten(cellPseudoSpikeT)))
    return vecPseudoSpikeTimes, vecPseudoEventT

# %%


def findfirst(indArray):
    vecStartSamples = np.where(indArray)[0]
    if vecStartSamples.size == 0:
        intStartSample = None
    else:
        intStartSample = vecStartSamples[0]
    return intStartSample

# %%


def flatten(l):
    g = genFlatten(l)
    x = []
    for i,v in enumerate(g):
        x.append(v)
    return x

def genFlatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el