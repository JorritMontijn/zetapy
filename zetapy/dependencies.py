# -*- coding: utf-8 -*-
import numpy as np
import logging
from scipy import stats
from math import pi, sqrt, exp, factorial
from collections.abc import Iterable

# %% calcZetaTwo
def calcZetaTwo(vecSpikeTimes1, vecEventStarts1, vecSpikeTimes2, vecEventStarts2, dblUseMaxDur, intResampNum, boolDirectQuantile):
    """
    Calculates two-sample zeta
    dZETA = calcZetaTwo(vecSpikeTimes1, vecEventStarts1, vecSpikeTimes2, vecEventStarts2, dblUseMaxDur, intResampNum, boolDirectQuantile)
    dZETA has entries:
        vecSpikeT, vecRealDeviation, vecRealFrac, vecRealFracLinear, cellRandTime, cellRandDeviation, dblZetaP, dblZETA, intZETAIdx
    """
    
    
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
    vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT = getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblUseMaxDur)

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
        vecStimUseOnTime = vecStartOnly[:,0] + matJitterPerTrial[:, intResampling].T

        # get temp offset
        vecRandDiff, vecThisSpikeFracs, vecThisFracLinear, vecThisSpikeTimes = getTempOffsetOne(vecPseudoSpikeTimes, vecStimUseOnTime, dblUseMaxDur)

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
            np.mean(vecMaxRandD), np.var(vecMaxRandD,ddof=1), arrMaxD) #default ddof for numpy var() is incorrect
    
    # return
    if arrZetaP.size == 1:arrZetaP = arrZetaP[0]
    if arrZETA.size == 1:arrZETA = arrZETA[0]
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
    vecThisSpikeTimes, vecIdx = np.unique(vecSpikesInTrial, return_index=True)

    # introduce minimum jitter to identical spikes
    indDuplicates = np.concatenate((np.diff(vecIdx) > 1, [False]))
    vecNotUnique = vecSpikesInTrial[vecIdx[indDuplicates]]

    if vecNotUnique.size > 0:
        dblUniqueOffset = np.array(np.finfo(vecSpikesInTrial.dtype.type).eps)
        for dblNotUnique in vecNotUnique:
            indIdx = dblNotUnique == vecSpikesInTrial
            vecSpikesInTrial[indIdx] = vecSpikesInTrial[indIdx] + \
                dblUniqueOffset*(np.arange(sum(indIdx))-sum(indIdx)/2-1)

        vecThisSpikeTimes, vecIdx = np.unique(
            vecSpikesInTrial, return_index=True)

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
def getSpikeT(vecSpikeTimes, vecEventTimes, dblUseMaxDur):
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
        if len(vecAssignIdx) > 0 and vecAssignIdx[-1] >= vecSpikesInTrial.size :
            vecSpikesInTrial = np.resize(vecSpikesInTrial,vecSpikesInTrial.size*2)
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
            + dblPseudoT0 - dblStepBegin - np.ptp(vecSpikeTimes[vecSampAddBeginning])  # make local to first spike in array, then preceding pseudo event t0
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
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

