# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:21:03 2023

@author: Jorrit
"""

import logging
from scipy.stats import norm
from scipy.signal import convolve2d
import numpy as np
from zetapy.dependencies import (flatten, findfirst)
from scipy import stats, interpolate, signal


def getMultiScaleDeriv(vecT, vecV,
                       dblSmoothSd=0.0, dblMinScale=None, dblBase=1.5, dblMeanRate=1.0, dblUseMaxDur=None, boolParallel=False):
    """"Returns multi-scale derivative of the deviation vector; i.e., the ZETA-derived instantaneous firing rate
       [vecRate,dMSD] = getMultiScaleDeriv(vecT, vecV,
                              dblSmoothSd=0.0, dblMinScale=None, dblBase=1.5, dblMeanRate=1.0, dblUseMaxDur=None, boolParallel=False)

        Required input:
            - vecT [N x 1]: timestamps (e.g., spike times)
            - vecV [N x 1]: values (e.g., deviation vector)

        Optional inputs:
            - dblSmoothSd: Gaussian SD of smoothing kernel (in # of samples) [default: 0.0]
            - dblMinScale: minimum derivative scale [default: round(log(1/1000) / log(dblBase))]
            - dblBase: base for exponential scale step size [default: 1.5]
            - dblMeanRate: mean spiking rate to normalize vecRate [default: None; returns unscaled derivatives]
            - dblUseMaxDur: trial duration to normalize vecRate [default: np.ptp(vecT)]

        Outputs:
            - vecRate; Instantaneous spiking rate
            - dMSD; dictionary with entries:
                - vecRate; instantaneous spiking rates (same as first output)
                - vecT; time-points corresponding to vecRate (same as input vecT)
                - vecM; Mean of multi-scale derivatives (same as vecRate, but before rescaling)
                - vecScale; timescales used to calculate derivatives
                - matMSD; multi-scale derivatives matrix
                - vecV; values on which vecRate is calculated (same as input vecV)
                - intSmoothSd; smoothing strength (same as input intSmoothSd)
                - dblMeanRate; mean rate used for rescaling (same as input dblMeanRate)

        Version history:
        2.0 - 2023-08-22
            Created by Jorrit Montijn - updated translation to Python.
    """

    # %% check inputs
    # trial dur
    dblRange = np.ptp(vecT)
    if dblUseMaxDur is None:
        dblUseMaxDur = dblRange

    # min scale
    if dblMinScale is None:
        dblMinScale = round(np.log(1/1000) / np.log(dblBase))

    # parallel processing: to do
    boolUseParallel = False

    # flatten and reorder vecT
    vecT = vecT.flatten()
    vecV = vecV.flatten()
    vecReorder = np.argsort(vecT, axis=0)
    vecT = vecT[vecReorder]
    vecV = vecV[vecReorder]
    indKeep = ~np.logical_or(vecT == 0, vecT == dblUseMaxDur)
    vecT = vecT[indKeep]
    vecV = vecV[indKeep]

    # %% get multi-scale derivative
    dblMaxScale = np.log(dblRange/10) / np.log(dblBase)
    vecExp = np.arange(dblMinScale, dblMaxScale)
    vecScale = dblBase**vecExp
    intScaleNum = len(vecScale)
    intN = len(vecT)
    matMSD = np.zeros((intN, intScaleNum))

    if boolUseParallel and False:
        # not implemented yet
        pass
    else:
        for intScaleIdx, dblScale in enumerate(vecScale):
            # run through all points
            matMSD[:, intScaleIdx] = calcSingleMSD(dblScale, vecT, vecV)

    # %% smoothing
    if dblSmoothSd > 0:
        intSmoothRange = 2*np.ceil(dblSmoothSd).astype(int)
        vecFilt = norm.pdf(range(-intSmoothRange, intSmoothRange+1), 0, dblSmoothSd)
        vecFilt = vecFilt / sum(vecFilt)

        # pad array
        intPadSize = np.floor(len(vecFilt)/2).astype(int)
        matMSD = np.pad(matMSD, ((intPadSize, intPadSize), (0, 0)), 'edge')

        # filter
        matMSD = convolve2d(matMSD, np.reshape(vecFilt, (-1, 1)), 'valid')

    # mean
    vecM = np.mean(matMSD, axis=1)

    # weighted average of vecM by inter-spike intervals
    dblMeanM = (1.0/dblUseMaxDur) * sum(((vecM[:-1] + vecM[1:])/2.0) * np.diff(vecT))

    # rescale to real firing rates
    vecRate = dblMeanRate * ((vecM + 1.0/dblUseMaxDur)/(dblMeanM + 1.0/dblUseMaxDur))

    # %% build output
    dMSD = dict()
    dMSD['vecRate'] = vecRate
    dMSD['vecT'] = vecT
    dMSD['vecM'] = vecM
    dMSD['vecScale'] = vecScale
    dMSD['matMSD'] = matMSD
    dMSD['vecV'] = vecV
    dMSD['dblSmoothSd'] = dblSmoothSd
    dMSD['dblMeanRate'] = dblMeanRate

    return vecRate, dMSD

# %%


def calcSingleMSD(dblScale, vecT, vecV):
    intN = vecT.size
    vecMSD = np.zeros((intN,))

    # run through all points
    for intS, dblT in enumerate(vecT):
        # select points within window

        dblMinEdge = dblT - dblScale/2
        dblMaxEdge = dblT + dblScale/2
        intIdxMinT = findfirst(vecT > dblMinEdge)
        if intIdxMinT is None:
            intIdxMinT = 0

        intIdxMaxT = findfirst(vecT > dblMaxEdge)
        if intIdxMaxT is None:
            intIdxMaxT = intN - 1
        else:
            intIdxMaxT = intIdxMaxT - 1

        if (intIdxMinT > intIdxMaxT):
            dblD = 0
        else:
            if (intIdxMinT == intIdxMaxT) and (intIdxMinT > 0) and (intIdxMinT < (intN-1)):
                intIdxMaxT = intIdxMinT + 1
                intIdxMinT = intIdxMinT - 1

            dbl_dT = np.max([dblScale, (vecT[intIdxMaxT] - vecT[intIdxMinT])])
            dblD = (vecV[intIdxMaxT] - vecV[intIdxMinT]) / dbl_dT

        # select points within window
        vecMSD[intS] = dblD

    # return single msd vector
    return vecMSD

# %%


def getPeak(vecData, vecT, vecRestrictRange=(-np.inf, np.inf), intSwitchZ=1):
    """Returns highest peak time, width, and location. Syntax:
        [dblPeakValue,dblPeakTime,dblPeakWidth,vecPeakStartStop,intPeakLoc,vecPeakStartStopIdx] = getPeak(vecData,vecT,vecRestrictRange)

    Required input:
        - vecData [N x 1]: values

    Optional inputs:
        - vecT [N x 1]: timestamps corresponding to vecData (default: [1:N])
        - vecRestrictRange: restrict peak to lie within vecRestrictRange(1) and vecRestrictRange(end)

    Output:
        dPeak, dict with entries:
        - dblPeakTime: time of peak
        - dblPeakValue: value of peak (rate)
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
        dblMu = np.mean(vecData[vecT < 0.02])
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

    if dblMaxPosVal is None and dblMaxNegVal is None:
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
        # vecPeakStarts = find(diff(indPeakMembers)==1);
        vecPeakStarts = np.where(np.diff([float(f) for f in indPeakMembers]) == 1)[0]
        # vecPeakStops = find(diff(indPeakMembers)==-1);
        vecPeakStops = np.where(np.diff([float(f) for f in indPeakMembers]) == -1)[0]
        if indPeakMembers[0]:
            # vecPeakStarts = [1 vecPeakStarts(:)'];
            vecPeakStarts = [0] + vecPeakStarts
        # if indPeakMembers(end) == 1
        if indPeakMembers[-1]:
            # vecPeakStops = [vecPeakStops(:)' numel(indPeakMembers)];
            vecPeakStops = vecPeakStops + [len(indPeakMembers)-1]

        # find closest points
        # intPeakStart = intPeakLoc-min(intPeakLoc - vecPeakStarts(vecPeakStarts<intPeakLoc));

        intPeakStart = intPeakLoc - np.min(intPeakLoc - vecPeakStarts[vecPeakStarts < intPeakLoc])
        intPeakStop = intPeakLoc + np.min(vecPeakStops[vecPeakStops >= intPeakLoc] - intPeakLoc)
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

    dPeak = dict()
    dPeak['dblPeakTime'] = dblPeakTime
    dPeak['dblPeakValue'] = dblPeakValue
    dPeak['dblPeakWidth'] = dblPeakWidth
    dPeak['vecPeakStartStop'] = vecPeakStartStop
    dPeak['intPeakLoc'] = intPeakLoc
    dPeak['vecPeakStartStopIdx'] = vecPeakStartStopIdx
    return dPeak


# %%
def getOnset(vecData, vecT, dblPeakTime=None, vecRestrictRange=None):
    """Returns peak onset. Syntax:
        dOnset = getOnset(vecData,vecT,dblPeakTime=None,vecRestrictRange=None)

    Required input:
        - vecData [N x 1]: values
        - vecT [N x 1]: timestamps corresponding to vecData

    Optional inputs:
        - dblPeakTime (float): timestamp corresponding to peak
        - vecRestrictRange [2 x 1]: restrict peak to lie within vecRestrictRange(1) and vecRestrictRange(end)

    Output:
        dOnset: dict with entries:
        - dblOnset: time of peak onset (first crossing half-height of peak)
        - dblValue: value at peak onset
        - dblBaseValue: baseline value (first value of vecRestrictRange, or first value in vecData if None)
        - dblPeakTime: time of peak
        - dblPeakValue: value of peak

    Version history:
    1.0 - June 19, 2020 Created by Jorrit Montijn, Translated to Python by Alexander Heimel
    2.0 - August 22, 2023 Updated translation by JM
    """

    # check input
    if vecRestrictRange is None:
        vecRestrictRange = [np.min(vecT), np.min(vecT)+np.ptp(vecT)]

    # remove time points outside restricted range
    indKeep = np.logical_and(vecT >= vecRestrictRange[0], vecT <= vecRestrictRange[1])
    vecCropT = vecT[indKeep]
    vecDataCropped = vecData[indKeep]

    # find peak if none supplied
    if dblPeakTime is None:
        intPeakIdx = np.argmax(vecDataCropped)
        dblPeakTime = vecT[intPeakIdx]
    else:
        intPeakIdx = findfirst(vecCropT > dblPeakTime)

    dblPeakValue = vecDataCropped[intPeakIdx]

    # calculate first timepoint crossing half-height of peak
    dblBaseValue = vecDataCropped[0]
    dblThresh = (dblPeakValue - dblBaseValue)/2 + dblBaseValue
    if dblThresh > 0:
        intOnsetIdx = findfirst(vecDataCropped >= dblThresh)
    else:
        intOnsetIdx = findfirst(vecDataCropped <= dblThresh)

    if intOnsetIdx is None:
        dblOnset = np.nan
        dblValue = np.nan
    else:
        dblOnset = vecCropT[intOnsetIdx]
        dblValue = vecDataCropped[intOnsetIdx]

    # return
    dPeak = dict()
    dPeak['dblOnset'] = dblOnset
    dPeak['dblValue'] = dblValue
    dPeak['dblBaseValue'] = dblBaseValue
    dPeak['dblPeakTime'] = dblPeakTime
    dPeak['dblPeakValue'] = dblPeakValue

    return dPeak
