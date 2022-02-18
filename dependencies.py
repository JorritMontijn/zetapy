import logging
import scipy
import collections
import numpy as np
from math import pi, sqrt, exp
from scipy import stats, interpolate, signal


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
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def getTempOffset(vecSpikeT,vecSpikeTimes,vecStimUseOnTime,dblUseMaxDur):
    """Calculate temporal offset vectors across folds and offsets.

    Syntax:
    [vecThisDiff,vecThisFrac,vecThisFracLinear] =
        getTempOffset(vecSpikeT,vecSpikeTimes,vecStimUseOnTime,dblUseMaxDur)
    """

    ## get inputs
    ### intMaxRep = numel(vecStimUseOnTime);
    intMaxRep = len(vecStimUseOnTime)

    ## get temp diff vector
    #pre-allocate
    ### cellSpikeTimesPerTrial = cell(intMaxRep,1);
    cellSpikeTimesPerTrial = [None] * intMaxRep

    # go through trials to build spike time vector
    ### for intEvent=1:intMaxRep
    for intEvent in range(intMaxRep):
        # get times
        dblStartT = vecStimUseOnTime[intEvent]
        dblStopT = dblStartT + dblUseMaxDur

        # build trial assignment
        ### cellSpikeTimesPerTrial{intEvent} = vecSpikeTimes(vecSpikeTimes < dblStopT & vecSpikeTimes > dblStartT) - dblStartT;
        cellSpikeTimesPerTrial[intEvent] = vecSpikeTimes[(vecSpikeTimes < dblStopT) & (vecSpikeTimes > dblStartT)] - dblStartT

    # get spikes in fold
    ### vecThisSpikeT = unique(cell2vec(cellSpikeTimesPerTrial));
    vecThisSpikeT = list(set(flatten(cellSpikeTimesPerTrial)))

    # get real fractions for training set
    ### vecThisSpikeTimes = sort([0;vecThisSpikeT(:);dblUseMaxDur],'ascend');
    vecThisSpikeTimes = sorted([0] + vecThisSpikeT + [dblUseMaxDur])
    ### vecThisSpikeFracs = linspace(0,1,numel(vecThisSpikeTimes))';
    vecThisSpikeFracs = np.linspace(0, 1, len(vecThisSpikeTimes))
    ### vecThisFrac = interp1(vecThisSpikeTimes,vecThisSpikeFracs,vecSpikeT);
    vecThisFrac = interpolate.interp1d(vecThisSpikeTimes, vecThisSpikeFracs)(vecSpikeT)

    # get linear fractions
    vecThisFracLinear = vecSpikeT / dblUseMaxDur

    # calc difference
    vecThisDiff = vecThisFrac - vecThisFracLinear
    vecThisDiff = vecThisDiff - np.mean(vecThisDiff)

    return vecThisDiff, vecThisFrac, vecThisFracLinear


def getPeak(vecData,vecT,vecRestrictRange=(-np.inf,np.inf),intSwitchZ=1):
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
        vecDataZ = (vecData - dblMu)/np.std(vecData)
    else:
        vecDataZ = vecData

    # get most prominent positive peak times
    ### (vecValsPos, vecLocsPos, vecWidthPos, vecPromsPos) = findpeaks(vecDataZ)
    #scipy.signal.find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    (vecLocsPos, peakProps) = signal.find_peaks(vecDataZ, threshold=0, prominence=-np.inf)
    vecValsPos = vecDataZ[vecLocsPos]
    vecPromsPos = peakProps['prominences']

    # remove peaks outside window
    ### indRemPeaks = vecT(vecLocsPos) < vecRestrictRange(1) | vecT(vecLocsPos) > vecRestrictRange(end);
    ### vecValsPos(indRemPeaks) = [];
    ### vecLocsPos(indRemPeaks) = [];
    ### vecPromsPos(indRemPeaks) = [];
    indKeepPeaks = (vecT[vecLocsPos] >= vecRestrictRange[0]) & (vecT[vecLocsPos] <= vecRestrictRange[1])
    vecValsPos = vecValsPos[indKeepPeaks]
    vecLocsPos = vecLocsPos[indKeepPeaks]
    vecPromsPos = vecPromsPos[indKeepPeaks]

    # select peak
    intPosIdx = np.argmax(vecValsPos)
    dblMaxPosVal = vecValsPos[intPosIdx]

    # get most prominent negative peak times
    (vecLocsNeg, peakProps) = signal.find_peaks(-vecDataZ, threshold=0, prominence=-np.inf)
    vecValsNeg = -vecDataZ[vecLocsNeg]
    vecPromsNeg = peakProps['prominences']

    # remove peaks outside window
    ### indRemPeaks = vecT(vecLocsNeg) < vecRestrictRange(1) | vecT(vecLocsNeg) > vecRestrictRange(end);
    ### vecValsNeg(indRemPeaks) = [];
    ### vecLocsNeg(indRemPeaks) = [];
    ### vecPromsNeg(indRemPeaks) = [];
    indKeepPeaks = (vecT[vecLocsNeg] >= vecRestrictRange[0]) & (vecT[vecLocsNeg] <= vecRestrictRange[1])
    vecValsNeg = vecValsNeg[indKeepPeaks]
    vecLocsNeg = vecLocsNeg[indKeepPeaks]
    vecPromsNeg = vecPromsNeg[indKeepPeaks]

    # select peak
    ### [dblMaxNegVal,intNegIdx] = max(vecValsNeg);
    intNegIdx = np.argmax(vecValsNeg)
    dblMaxNegVal = vecValsNeg[intNegIdx]

    if dblMaxPosVal == None and dblMaxNegVal == None :
        indPeakMembers = None
    elif (dblMaxPosVal!=None and dblMaxNegVal==None) or (dblMaxPosVal!=None and (abs(dblMaxPosVal) >= abs(dblMaxNegVal))):
        intIdx = intPosIdx
        intPeakLoc = vecLocsPos[intIdx]
        dblPeakProm = vecPromsPos[intIdx]
        dblCutOff = vecDataZ[intPeakLoc] - dblPeakProm/2
        indPeakMembers = (vecDataZ > dblCutOff)
    elif (dblMaxPosVal == None and dblMaxNegVal!=None) or (dblMaxNegVal!=None and (abs(dblMaxPosVal) < abs(dblMaxNegVal))):
        intIdx = intNegIdx
        intPeakLoc = vecLocsNeg[intIdx]
        dblPeakProm = vecPromsNeg[intIdx]
        dblCutOff = vecDataZ[intPeakLoc] + dblPeakProm/2
        indPeakMembers = (vecDataZ < dblCutOff)

    if len(indPeakMembers)>0:
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
        if intPeakStop > vecT.shape[0]:
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
