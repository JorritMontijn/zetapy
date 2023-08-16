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
# from zetapy.dependencies import (flatten, getTempOffset, getGumbel, getPeak, getOnset,
#                                 calculatePeths)


def zetatest(arrSpikeTimes, arrEventTimes,
             dblUseMaxDur=None, intResampNum=100, intPlot=0, dblJitterSize=2.0,
             intLatencyPeaks=2, tplRestrictRange=(-np.inf, np.inf),
             boolStitch=True, boolDirectQuantile=False,
             boolReturnRate=False, boolReturnZETA=False, boolVerbose=False):
    """
    Calculates neuronal responsiveness index ZETA.

    Montijn, J.S., Seignette, K., Howlett, M.H., Cazemier, J.L., Kamermans, M., Levelt, C.N.,
    and Heimel, J.A. (2021). A parameter-free statistical test for neuronal responsiveness.
    eLife 10, e71969.

    Syntax:
    [dblZetaP,dZETA,dRate,arrLatencies] = zetatest(arrSpikeTimes,arrEventTimes,
                                                   dblUseMaxDur=None, intResampNum=100, intPlot=0, dblJitterSize=2.0,
                                                   intLatencyPeaks=2, tplRestrictRange=(-np.inf, np.inf),
                                                   boolStitch=True, boolDirectQuantile=False,
                                                   boolReturnRate=False, boolReturnZETA=False, boolVerbose=False)

    Parameters
    ----------
    arrSpikeTimes : 1D array (float)
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
    boolReturnZETA : boolean
        switch to return dictionary with additional ZETA parameters
    boolVerbose : boolean
        switch to print progress messages (default: False)

    Returns
    -------
    dblZetaP : float
        p-value based on Zenith of Event-based Time-locked Anomalies
    dZETA : dict (optional)
        additional parameters of ZETA test, return when using boolReturnZETA
            dblZETA; FDR-corrected responsiveness z-score (i.e., >2 is significant)
            dblD; temporal deviation value underlying ZETA
            dblP; p-value corresponding to ZETA
            dblPeakT; time corresponding to ZETA
            intPeakIdx; entry corresponding to ZETA
            dblMeanD; Cohen's D based on mean-rate stim/base difference
            dblMeanP; p-value based on mean-rate stim/base difference
            vecSpikeT: timestamps of spike times (corresponding to vecD)
            vecRealFrac; cumulative distribution of spike times
            vecRealFracLinear; linear baseline of cumulative distribution
            vecD; temporal deviation vector of data
            vecNoNormD; temporal deviation which is not mean subtracted
            matRandD; baseline temporal deviation matrix of jittered data
            dblD_InvSign; largest peak of inverse sign to ZETA (i.e., -ZETA)
            dblPeakT_InvSign; time corresponding to -ZETA
            intPeakIdx_InvSign; entry corresponding to -ZETA
            dblUseMaxDur; window length used to calculate ZETA
    dRate : dict (optional)
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
    arrLatencies : 1D array
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
    arrLatencies = np.empty((1, 4))
    arrLatencies.fill(np.nan)

    # fill dZETA
    dZETA['dblPeakRate'] = None
    dZETA['dblZETA'] = None
    dZETA['dblD'] = None
    dZETA['dblP'] = dblZetaP
    dZETA['dblPeakT'] = None
    dZETA['intPeakIdx'] = None
    dZETA['dblMeanD'] = None
    dZETA['dblMeanP'] = None
    dZETA['vecSpikeT'] = None
    dZETA['vecRealFrac'] = None
    dZETA['vecRealFracLinear'] = None
    dZETA['vecD'] = None
    dZETA['vecNoNormD'] = None
    dZETA['matRandD'] = None
    dZETA['dblD_InvSign'] = None
    dZETA['dblPeakT_InvSign'] = None
    dZETA['intPeakIdx_InvSign'] = None
    dZETA['dblUseMaxDur'] = None

    # fill dRate
    dZETA['vecRate'] = None
    dZETA['vecT'] = None
    dZETA['vecM'] = None
    dZETA['vecScale'] = None
    dZETA['matMSD'] = None
    dZETA['vecV'] = None
    dZETA['dblPeakTime'] = None
    dZETA['dblPeakWidth'] = None
    dZETA['vecPeakStartStop'] = None
    dZETA['intPeakLoc'] = None
    dZETA['vecPeakStartStopIdx'] = None
    dZETA['dblOnset'] = None

    # %% prep data and assert inputs are correct

    # arrSpikeTimes must be [S by 1] array
    assert (len(arrSpikeTimes.shape) == 1 or arrSpikeTimes.shape[1] == 1) and issubclass(
        arrSpikeTimes.dtype.type, np.floating), "Input arrSpikeTimes is not a 1D float np.array with >2 spike times"

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
    arrEventStarts = arrEventTimes[:, 0]

    # check if number of events and spikes is sufficient
    if arrSpikeTimes.size < 3 or arrEventTimes.size < 3:
        if arrSpikeTimes.size < 3:
            strMsg1 = f"Number of spikes ({arrSpikeTimes.size}) is too few to calculate zeta; "
        else:
            strMsg1 = ""
        if arrEventStarts.size < 3:
            strMsg2 = f"Number of events ({arrEventStarts.size}) is too few to calculate zeta; "
        else:
            strMsg2 = ""
        logging.warning("zetatest: " + strMsg1 + strMsg2 + "defaulting to p=1.0")

        return dblZetaP, dZETA, dRate, arrLatencies

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
        boolDirectQuantile = True
    else:
        assert isinstance(boolDirectQuantile, bool), "boolDirectQuantile is not a boolean"

    # return dRate
    if boolReturnRate is None:
        boolReturnRate = False
    else:
        assert isinstance(boolReturnRate, bool), "boolReturnRate is not a boolean"

    # return dZETA
    if boolReturnZETA is None:
        boolReturnZETA = False
    else:
        assert isinstance(boolReturnZETA, bool), "boolReturnZETA is not a boolean"

    # verbosity
    if boolVerbose is None:
        boolVerbose = False
    else:
        assert isinstance(boolVerbose, bool), "boolVerbose is not a boolean"

    # to do: parallel computing
    boolParallel = False

    # %% calculate zeta
    arrSpikeT, arrRealDiff, arrRealFrac, arrRealFracLinear, cellRandT, cellRandDiff, dblZetaP, dblZETA, intZETALoc = calcZetaOne(
        arrSpikeTimes, arrEventStarts, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch, boolParallel)

    # check if calculation is valid, otherwise return empty values
    if intZETALoc is None:
        logging.warning("zetatest: calculation failed, defaulting to p=1.0")
        return dblZetaP, dZETA, dRate, arrLatencies

    # %% extract real outputs
    # get location
    dblMaxDTime = arrSpikeT[intZETALoc]
    dblD = arrRealDiff[intZETALoc]

    # find peak of inverse sign
    intPeakLocInvSign = np.argmax(-np.sign(dblD)*arrRealDiff)
    dblMaxDTimeInvSign = arrSpikeT(intPeakLocInvSign)
    dblD_InvSign = arrRealDiff(intPeakLocInvSign)
    
    # %% calculate mean-rate difference with t-test
    if boolStopSupplied:
        # calculate spike counts and durations during baseline and stimulus times
        arrRespBinsDur = np.sort(np.reshape(arrEventTimes, -1))
        arrR, arrBins = np.histogram(arrSpikeTimes, bins=arrRespBinsDur)
        arrD = np.diff(arrRespBinsDur)
        
        #mean rate during on-time
        arrMu_Dur = np.divide(np.float64(arrR[0:len(arrR):2]), arrD[0:len(arrD):2])
        
        #calculate mean rates during off-times
        dblStart1 = np.min(arrRespBinsDur)
        dblFirstPreDur = dblStart1 - np.max(dblStart1 - np.median(arrD[1:len(arrD):2]), initial=0)
        dblR1 = np.sum(np.logical_and(arrSpikeTimes > (dblStart1 - dblFirstPreDur), arrSpikeTimes < dblStart1))
        arrMu_Pre = np.divide(np.concatenate([[dblR1], arrR[1:len(arrR):2]]),
                              np.concatenate([[dblFirstPreDur], arrD[1:len(arrD):2]]))

        # get metrics
        dblMeanP = stats.ttest_rel(arrMu_Dur, arrMu_Pre)[1]
        dblMeanZ = -stats.norm.ppf(dblMeanP/2)


# %% calculate instantaneous firing rates
# 	if intLatencyPeaks > 0 || nargout > 2 || intPlot > 0
# 		%get average of multi-scale derivatives, and rescaled to instantaneous spiking rate
# 		dblMeanRate = (numel(vecSpikeT)/(dblUseMaxDur*numel(vecEventStarts)));
# 		[vecRate,sRate] = getMultiScaleDeriv(vecSpikeT,vecRealDiff,[],[],[],intPlot,dblMeanRate,dblUseMaxDur);
# 	end

# %% calculate IFR statistics
# 	if ~isempty(sRate) && intLatencyPeaks > 0
# 		%get IFR peak
# 		[dblPeakRate,dblPeakTime,dblPeakWidth,vecPeakStartStop,intPeakLoc,vecPeakStartStopIdx] = getPeak(vecRate,vecSpikeT,vecRestrictRange);
# 		sRate.dblPeakRate = dblPeakRate;
# 		sRate.dblPeakTime = dblPeakTime;
# 		sRate.dblPeakWidth = dblPeakWidth;
# 		sRate.vecPeakStartStop = vecPeakStartStop;
# 		sRate.intPeakLoc = intPeakLoc;
# 		sRate.vecPeakStartStopIdx = vecPeakStartStopIdx;
#
#
# 		if ~isnan(dblPeakTime)
# 			%assign array data
# 			if intLatencyPeaks > 3
# 				%get onset
# 				[dblOnset,dblOnsetVal] = getOnset(vecRate,vecSpikeT,dblPeakTime,vecRestrictRange);
# 				sRate.dblOnset = dblOnset;
# 				vecLatencies = [dblMaxDTime dblMaxDTimeInvSign dblPeakTime dblOnset];
# 				vecLatencyVals = [vecRate(intZETALoc) vecRate(intPeakLocInvSign) vecRate(intPeakLoc) dblOnsetVal];
# 			else
# 				sRate.dblOnset = [nan];
# 				vecLatencies = [dblMaxDTime dblMaxDTimeInvSign dblPeakTime];
# 				vecLatencyVals = [vecRate(intZETALoc) vecRate(intPeakLocInvSign) vecRate(intPeakLoc)];
# 			end
# 			intLatencyPeaks = min([intLatencyPeaks numel(vecLatencies)]);
# 			vecLatencies = vecLatencies(1:intLatencyPeaks);
# 			vecLatencyVals = vecLatencyVals(1:intLatencyPeaks);
# 			if intPlot > 0
# 				hold on
# 				scatter(dblPeakTime,vecRate(intPeakLoc),'gx');
# 				scatter(dblMaxDTime,vecRate(intZETALoc),'bx');
# 				scatter(dblMaxDTimeInvSign,vecRate(intPeakLocInvSign),'b*');
# 				if intLatencyPeaks > 3
# 					scatter(dblOnset,dblOnsetVal,'rx');
# 					title(sprintf('ZETA=%.0fms,-ZETA=%.0fms,Pk=%.0fms,On=%.2fms',dblMaxDTime*1000,dblMaxDTimeInvSign*1000,dblPeakTime*1000,dblOnset*1000));
# 				else
# 					title(sprintf('ZETA=%.0fms,-ZETA=%.0fms,Pk=%.0fms',dblMaxDTime*1000,dblMaxDTimeInvSign*1000,dblPeakTime*1000));
# 				end
# 				hold off
# 				fixfig;
#
# 				if intPlot > 3
# 					vecHandles = get(gcf,'children');
# 					ptrFirstSubplot = vecHandles(find(contains(get(vecHandles,'type'),'axes'),1,'last'));
# 					axes(ptrFirstSubplot);
# 					vecY = get(gca,'ylim');
# 					hold on;
# 					if intLatencyPeaks > 3,plot(dblOnset*[1 1],vecY,'r--');end
# 					plot(dblPeakTime*[1 1],vecY,'g--');
# 					plot(dblMaxDTime*[1 1],vecY,'b--');
# 					plot(dblMaxDTimeInvSign*[1 1],vecY,'b-.');
# 					hold off
# 				end
# 			end
# 		else
# 			%placeholder peak data
# 			sRate.dblOnset = [nan];
# 			vecLatencies = [nan nan nan nan];
# 			vecLatencyVals = [nan nan nan nan];
# 		end
# 	else
# 		vecLatencies = [];
# 		vecLatencyVals = [];
# 	end

# check number of latencies
# if numel(vecLatencies) < intLatencyPeaks
# vecLatencies(end+1:intLatencyPeaks) = nan;
# vecLatencyVals(end+1:intLatencyPeaks) = nan;
# end

# %% build optional output structure
# 	if nargout > 1
# 		sZETA = struct;
# 		sZETA.dblZETA = dblZETA;
# 		sZETA.dblD = dblD;
# 		sZETA.dblP = dblZetaP;
# 		sZETA.dblPeakT = dblMaxDTime;
# 		sZETA.intPeakIdx = intZETALoc;
# 		if boolStopSupplied
# 			sZETA.dblMeanZ = dblMeanZ;
# 			sZETA.dblMeanP = dblMeanP;
# 			sZETA.vecMu_Dur = vecMu_Dur;
# 			sZETA.vecMu_Pre = vecMu_Pre;
# 		end
# 		sZETA.vecSpikeT = vecSpikeT;
# 		sZETA.vecD = vecRealDiff;
# 		sZETA.cellRandT = cellRandT;
# 		sZETA.cellRandDiff = cellRandDiff;
#
# 		sZETA.dblD_InvSign = dblD_InvSign;
# 		sZETA.dblPeakT_InvSign = dblMaxDTimeInvSign;
# 		sZETA.intPeakIdx_InvSign = intPeakLocInvSign;
# 		sZETA.dblUseMaxDur = dblUseMaxDur;
# 		sZETA.vecLatencyVals = vecLatencyVals;
# 	end

    # %% plot
    if intPlot > 0:
        plotzeta(dZETA,dRate,intPlot)
        
    
    # %% return outputs
    return dblZetaP, dZETA, dRate, arrLatencies

