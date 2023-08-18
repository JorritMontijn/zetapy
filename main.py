# -*- coding: utf-8 -*-
import numpy as np
import time
import logging
import math
import matplotlib.pyplot as plt
import tkinter as tk
# from zetapy import msd
from scipy import stats
from zetapy.dependencies import (calcZetaOne, plotzeta)
from zetapy.ifr_dependencies import (getMultiScaleDeriv, getPeak, getOnset)
# from zetapy.dependencies import (flatten, getTempOffset, getGumbel, getPeak, getOnset,
#                                 calculatePeths)


def zetatest(vecSpikeTimes, arrEventTimes,
             dblUseMaxDur=None, intResampNum=100, intPlot=0, dblJitterSize=2.0,
             intLatencyPeaks=2, tplRestrictRange=(-np.inf, np.inf),
             boolStitch=True, boolDirectQuantile=False,
             boolReturnRate=False, boolVerbose=False):
    """
    Calculates neuronal responsiveness index ZETA.

    Montijn, J.S., Seignette, K., Howlett, M.H., Cazemier, J.L., Kamermans, M., Levelt, C.N.,
    and Heimel, J.A. (2021). A parameter-free statistical test for neuronal responsiveness.
    eLife 10, e71969.

    Syntax:
    dblZetaP,dZETA,dRate,vecLatencies = zetatest(vecSpikeTimes,arrEventTimes,
                                                   dblUseMaxDur=None, intResampNum=100, intPlot=0, dblJitterSize=2.0,
                                                   intLatencyPeaks=2, tplRestrictRange=(-np.inf, np.inf),
                                                   boolStitch=True, boolDirectQuantile=False,
                                                   boolReturnRate=False, boolVerbose=False)

    Parameters
    ----------
    vecSpikeTimes : 1D array (float)
        spike times (in seconds).
    arrEventTimes : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate mean-rate difference.

    dblUseMaxDur : float
        window length for calculating ZETA: ignore all spikes beyond this duration after event onset
        (default: minimum of event onset to event onset)
    intResampNum : integer
        number of resamplings (default: 100)
        [Note: if your p-value is close to significance, you should increase this number to enhance the precision]
    intPlot : int
        plotting switch (0: no plot, 1: plot figure) (default: 0)
    dblJitterSize; float
        sets the temporal jitter window relative to dblUseMaxDur (default: 2.0)
    intLatencyPeaks : integer
        maximum number of latency peaks to return (1-4) (default: 2) [see below]
    tplRestrictRange : 2-element tuple
        temporal range within which to restrict onset/peak latencies (default: [-inf inf])
    boolStitch; boolean
        switch to use data-stitching to ensure continuous time (default: True)
    boolDirectQuantile: boolean
         switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
         [Note: requires many resamplings!]
    boolReturnRate : boolean
        switch to return dictionary with spiking rate features [note: return-time is much faster if this is False]
    boolVerbose : boolean
        switch to print progress messages (default: False)

    Returns
    -------
    dblZetaP : float
        p-value based on Zenith of Event-based Time-locked Anomalies
    dZETA : dict
        additional information of ZETA test
            dblZetaP; p-value based on Zenith of Event-based Time-locked Anomalies (same as above)
            dblZETA; responsiveness z-score (i.e., >2 is significant)
            dblMeanZ; z-score for mean-rate stim/base difference (i.e., >2 is significant)
            dblMeanP; p-value based on mean-rate stim/base difference
            dblZETADeviation; temporal deviation value underlying ZETA
            dblZETATime; time corresponding to ZETA
            intZETAIdx; entry corresponding to ZETA
            vecMu_Dur; spiking rate per trial during stim (used for mean-rate test)
            vecMu_Pre; spiking rate per trial during baseline (used for mean-rate test)
            dblD_InvSign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            dblT_InvSign; time corresponding to -ZETA
            intIdx_InvSign; entry corresponding to -ZETA
            vecSpikeT: timestamps of spike times (corresponding to vecRealDeviation)
            vecRealDeviation; temporal deviation vector of data
            vecRealFrac; cumulative distribution of spike times
            vecRealFracLinear; linear baseline of cumulative distribution
            cellRandTime; jittered spike times corresponding to cellRandDeviation
            cellRandDeviation; baseline temporal deviation matrix of jittered data
            dblUseMaxDur; window length used to calculate ZETA
            vecLatencies; latency times also provided as separate return variable (see below)
            vecLatencyVals; values corresponding to above latencies (ZETA, -ZETA, rate at peak, rate at onset)
            
    dRate : dict (empty if boolReturnRate was not set to True)
        additional parameters of the firing rate, return with boolReturnRate
            vecRate; instantaneous spiking rates (like a PSTH)
            vecT; time-points corresponding to vecRate (same as dZETA.vecSpikeT)
            vecM; Mean of multi-scale derivatives
            vecScale; timescales used to calculate derivatives
            matMSD; multi-scale derivatives matrix
            vecV; values on which vecRate is calculated (same as dZETA.vecZ)
        Data on the peak:
            dblPeakTime; time of peak (in seconds)
            dblPeakWidth; duration of peak (in seconds)
            vecPeakStartStop; start and stop time of peak (in seconds)
            intPeakLoc; spike index of peak (corresponding to dZETA.vecSpikeT)
            vecPeakStartStopIdx; spike indices of peak start/stop (corresponding to dZETA.vecSpikeT)
            dblOnset: latency for peak onset
    vecLatencies : 1D array
       Different latency estimates, number determined by intLatencyPeaks.
       If no peaks are detected, it returns NaNs
            1) Latency of ZETA
            2) Latency of largest z-score with inverse sign to ZETA
            3) Peak time of instantaneous firing rate
            4) Onset time of response peak, defined as the first crossing of peak half-height

    Code by Jorrit Montijn, Guido Meijer & Alexander Heimel

    Version history:
    2.5 - 17 June 2020 Jorrit Montijn, translated to python by Alexander Heimel
    2.5.1 - 18 February 2022 Bugfix by Guido Meijer of 1D arrEventTimes
    2.6 - 20 February 2022 Refactoring of python code by Guido Meijer
    3.0 - 16 Aug 2023 New port of zetatest to Python [by Jorrit Montijn]
    """

    # %% build placeholder outputs
    dblZetaP = 1.0
    dZETA = dict()
    dRate = dict()
    vecLatencies = np.empty((4, 1))
    vecLatencies.fill(np.nan)
    vecLatencyVals = vecLatencies
   
    ## fill dZETA
    #ZETA significance
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = None
    #mean-rate significance
    dZETA['dblMeanZ'] = None
    dZETA['dblMeanP'] = None
    #data on ZETA peak
    dZETA['dblZETADeviation'] = None
    dZETA['dblZETATime'] = None
    dZETA['intZETAIdx'] = None
    #data underlying mean-rate test
    dZETA['vecMu_Dur'] = None
    dZETA['vecMu_Pre'] = None
    #inverse-sign ZETA
    dZETA['dblD_InvSign'] = None
    dZETA['dblT_InvSign'] = None
    dZETA['intIdx_InvSign'] = None
    
    #derived from calcZetaOne
    dZETA['vecSpikeT'] = None
    dZETA['vecRealDeviation'] = None
    dZETA['vecRealFrac'] = None
    dZETA['vecRealFracLinear'] = None
    dZETA['cellRandTime'] = None
    dZETA['cellRandDeviation'] = None
    #dZETA['dblZetaP'] = None #<-updates automatically
    #dZETA['dblZETA'] = None #<-updates automatically
    #dZETA['intZETAIdx'] = None #<-updates automatically
    
    #window used for analysis
    dZETA['dblUseMaxDur'] = None
    #copy of latency vectors
    dZETA['vecLatencies'] = vecLatencies
    dZETA['vecLatencyVals'] = vecLatencyVals
    
    # fill dRate
    dRate['vecRate'] = None
    dRate['vecT'] = None
    dRate['vecM'] = None
    dRate['vecScale'] = None
    dRate['matMSD'] = None
    dRate['vecV'] = None
    dRate['dblPeakTime'] = None
    dRate['dblPeakWidth'] = None
    dRate['vecPeakStartStop'] = None
    dRate['intPeakLoc'] = None
    dRate['vecPeakStartStopIdx'] = None
    dRate['dblOnset'] = None

    # %% prep data and assert inputs are correct

    # vecSpikeTimes must be [S by 1] array
    assert (len(vecSpikeTimes.shape) == 1 or vecSpikeTimes.shape[1] == 1) and issubclass(
        vecSpikeTimes.dtype.type, np.floating), "Input vecSpikeTimes is not a 1D float np.array with >2 spike times"
    vecSpikeTimes = np.sort(np.reshape(vecSpikeTimes, (-1, 1)), axis=0)
    
    # ensure orientation and assert that arrEventTimes is a 1D or N-by-2 array of floats
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
    vecEventStarts = arrEventTimes[:, 0]

    # check if number of events and spikes is sufficient
    if vecSpikeTimes.size < 3 or arrEventTimes.size < 3:
        if vecSpikeTimes.size < 3:
            strMsg1 = f"Number of spikes ({vecSpikeTimes.size}) is too few to calculate zeta; "
        else:
            strMsg1 = ""
        if vecEventStarts.size < 3:
            strMsg2 = f"Number of events ({vecEventStarts.size}) is too few to calculate zeta; "
        else:
            strMsg2 = ""
        logging.warning("zetatest: " + strMsg1 + strMsg2 + "defaulting to p=1.0")

        return dblZetaP, dZETA, dRate, vecLatencies

    # is stop supplied?
    if len(arrEventTimes.shape) > 1 and arrEventTimes.shape[1] > 1:
        boolStopSupplied = True
        arrEventOnDur = arrEventTimes[:, 1] - arrEventTimes[:, 0]
        assert np.all(arrEventOnDur > 0), "at least one event in arrEventTimes has a negative duration"

    else:
        boolStopSupplied = False
        dblMeanZ = np.nan
        dblMeanP = np.nan

    # trial dur
    if dblUseMaxDur is None:
        dblUseMaxDur = np.min(np.diff(arrEventTimes[:, 0]))
    else:
        dblUseMaxDur = np.float64(dblUseMaxDur)
        assert dblUseMaxDur.size == 1 and dblUseMaxDur > 0, "dblUseMaxDur is not a positive scalar float"

    # get resampling num
    if intResampNum is None:
        intResampNum = np.int64(100)
    else:
        intResampNum = np.int64(intResampNum)
        assert intResampNum.size == 1 and intResampNum > 1, "intResampNum is not a positive integer"

    # plotting
    if intPlot is None:
        intPlot = np.int64(0)
    else:
        intPlot = np.int64(intPlot)
        assert intPlot.size == 1 and intPlot > -1 and intPlot < 5, "intPlot has an invalid value"

    # jitter
    if dblJitterSize is None:
        dblJitterSize = np.float64(2.0)
    else:
        dblJitterSize = np.float64(dblJitterSize)
        assert dblJitterSize.size == 1 and dblJitterSize > 0, "dblJitterSize is not a postive scalar float"

    # latency peaks
    if intLatencyPeaks is None:
        intLatencyPeaks = np.int64(0)
    else:
        intLatencyPeaks = np.int64(intLatencyPeaks)
        assert intLatencyPeaks.size == 1 and intLatencyPeaks > -1 and intLatencyPeaks < 5, "intLatencyPeaks has an invalid value"

    # latency peaks
    if tplRestrictRange is None:
        tplRestrictRange = np.float64((-np.inf, np.inf))
    else:
        tplRestrictRange = np.float64(tplRestrictRange)
        assert tplRestrictRange.size == 2, "tplRestrictRange does not have two values"

    # stitching
    if boolStitch is None:
        boolStitch = True
    else:
        assert isinstance(boolStitch, bool), "boolStitch is not a boolean"

    # direct quantile comnputation
    if boolDirectQuantile is None:
        boolDirectQuantile = False
    else:
        assert isinstance(boolDirectQuantile, bool), "boolDirectQuantile is not a boolean"

    # return dRate
    if boolReturnRate is None:
        boolReturnRate = False
    else:
        assert isinstance(boolReturnRate, bool), "boolReturnRate is not a boolean"
    if (intLatencyPeaks > 2 or intPlot > 0) and boolReturnRate is False:
        boolReturnRate = True
        logging.warning(
            "zetatest: boolReturnRate was False, but you requested plotting or latencies, so boolReturnRate is now set to True")

    # verbosity
    if boolVerbose is None:
        boolVerbose = False
    else:
        assert isinstance(boolVerbose, bool), "boolVerbose is not a boolean"

    # to do: parallel computing
    boolParallel = False

    # %% calculate zeta
    dZETA_One = calcZetaOne(vecSpikeTimes, vecEventStarts, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch, boolParallel)
    
    #update and unpack
    dZETA.update(dZETA_One)
    vecSpikeT = dZETA['vecSpikeT']
    vecRealDeviation = dZETA['vecRealDeviation']
    vecRealFrac = dZETA['vecRealFrac']
    vecRealFracLinear = dZETA['vecRealFracLinear']
    cellRandTime = dZETA['cellRandTime']
    cellRandDeviation = dZETA['cellRandDeviation']
    dblZetaP = dZETA['dblZetaP']
    dblZETA = dZETA['dblZETA']
    intZETAIdx = dZETA['intZETAIdx']
    
    # check if calculation is valid, otherwise return empty values
    if intZETAIdx is None:
        logging.warning("zetatest: calculation failed, defaulting to p=1.0")
        return dblZetaP, dZETA, dRate, vecLatencies

    # %% extract real outputs
    # get location
    dblZETATime = vecSpikeT[intZETAIdx]
    dblZETADeviation = vecRealDeviation[intZETAIdx]

    # find peak of inverse sign
    intIdx_InvSign = np.argmax(-np.sign(dblZETADeviation)*vecRealDeviation)
    dblT_InvSign = vecSpikeT[intIdx_InvSign]
    dblD_InvSign = vecRealDeviation[intIdx_InvSign]

    # %% calculate mean-rate difference with t-test
    if boolStopSupplied:
        # calculate spike counts and durations during baseline and stimulus times
        vecRespBinsDur = np.sort(np.reshape(arrEventTimes, -1))
        vecR, arrBins = np.histogram(vecSpikeTimes, bins=vecRespBinsDur)
        vecD = np.diff(vecRespBinsDur)

        # mean rate during on-time
        vecMu_Dur = np.divide(np.float64(vecR[0:len(vecR):2]), vecD[0:len(vecD):2])

        # calculate mean rates during off-times
        dblStart1 = np.min(vecRespBinsDur)
        dblFirstPreDur = dblStart1 - np.max(dblStart1 - np.median(vecD[1:len(vecD):2]), initial=0)
        dblR1 = np.sum(np.logical_and(vecSpikeTimes > (dblStart1 - dblFirstPreDur), vecSpikeTimes < dblStart1))
        vecMu_Pre = np.divide(np.concatenate([[dblR1], vecR[1:len(vecR):2]]),
                              np.concatenate([[dblFirstPreDur], vecD[1:len(vecD):2]]))

        # get metrics
        dblMeanP = stats.ttest_rel(vecMu_Dur, vecMu_Pre)[1]
        dblMeanZ = -stats.norm.ppf(dblMeanP/2)

    # %% calculate instantaneous firing rates
    if boolReturnRate:
        # get average of multi-scale derivatives, and rescaled to instantaneous spiking rate
        dblMeanRate = vecSpikeT.size/(dblUseMaxDur*vecEventStarts.size)
        vecRate, dRate = getMultiScaleDeriv(vecSpikeT, vecRealDeviation,
                                            dblMeanRate=dblMeanRate, dblUseMaxDur=dblUseMaxDur, boolParallel=boolParallel)

        # %% calculate IFR statistics
        if vecRate is not None and intLatencyPeaks > 0:
            # get IFR peak
            dPeak = getPeak(vecRate, vecSpikeT, tplRestrictRange=tplRestrictRange)
            dRate.update(dPeak)
            if dRate['dblPeakTime'] is not None and ~np.isnan(dRate['dblPeakTime']):
                # assign array data
                if intLatencyPeaks > 3:
                    # get onset
                    dblOnset, dblOnsetVal = getOnset(vecRate, vecSpikeT, dRate['dblPeakTime'], tplRestrictRange)[0:1]
                    dRate['dblOnset'] = dblOnset
                    vecLatencies = [dblZETATime, dblT_InvSign, dRate['dblPeakTime'], dblOnset]
                    vecLatencyVals = [vecRate[intZETAIdx], vecRate[intIdx_InvSign], vecRate[dPeak['intPeakLoc']], dblOnsetVal]
                else:
                    dRate['dblOnset'] = None
                    vecLatencies = [dblZETATime, dblT_InvSign, dRate['dblPeakTime'], None]
                    vecLatencyVals = [vecRate[intZETAIdx], vecRate[intIdx_InvSign], vecRate[dPeak['intPeakLoc']], None]

    # %% build output dictionary
    ## fill dZETA
    dZETA['dblZETADeviation'] = dblZETADeviation
    dZETA['dblZETATime'] = dblZETATime
    if boolStopSupplied:
        dZETA['dblMeanZ'] = dblMeanZ
        dZETA['dblMeanP'] = dblMeanP
        dZETA['vecMu_Dur'] = vecMu_Dur
        dZETA['vecMu_Pre'] = vecMu_Pre
    
    #inverse-sign ZETA
    dZETA['dblD_InvSign'] = dblD_InvSign
    dZETA['dblT_InvSign'] = dblT_InvSign
    dZETA['intIdx_InvSign'] = intIdx_InvSign
    #window used for analysis
    dZETA['dblUseMaxDur'] = dblUseMaxDur
    #copy of latency vectors
    dZETA['vecLatencies'] = vecLatencies
    dZETA['vecLatencyVals'] = vecLatencyVals
    
    # %% plot
    if intPlot > 0:
        plotzeta(dZETA, dRate, intPlot)

    # %% return outputs
    return dblZetaP, dZETA, dRate, vecLatencies
