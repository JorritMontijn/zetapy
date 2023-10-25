# -*- coding: utf-8 -*-
import numpy as np
import time
import logging
import math
import matplotlib.pyplot as plt
import tkinter as tk
# from zetapy import msd
from scipy import stats
from zetapy.dependencies import (calcZetaOne, calcZetaTwo, getTempOffsetOne, flatten, findfirst)
from zetapy.ifr_dependencies import (getMultiScaleDeriv, getPeak, getOnset)
from zetapy.plot_dependencies import calculatePeths
from zetapy.ts_dependencies import calcTsZetaOne, calcTsZetaTwo, getPseudoTimeSeries, getTsRefT, getInterpolatedTimeSeries, 

# %% two-sample zeta test


def zetatest2(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2,
              dblUseMaxDur=None, intResampNum=250, boolPlot=False, boolDirectQuantile=False):
    """
    Calculates two-sample zeta-test

    Heimel, J.A., Meijer, G.T., Montijn, J.S. (2023). A new family of statistical tests for responses
    in point-event and time-series data for one- and two-sample comparisons. bioRxiv

    Montijn, J.S., Seignette, K., Howlett, M.H., Cazemier, J.L., Kamermans, M., Levelt, C.N.,
    and Heimel, J.A. (2021). A parameter-free statistical test for neuronal responsiveness.
    eLife 10, e71969.

    Syntax:
    dblZeta2P,dZETA2 = zetatest2(vecSpikeTimes1,arrEventTimes1,vecSpikeTimes2,arrEventTimes2,
                                                  dblUseMaxDur=None, intResampNum=250, boolPlot=False, boolDirectQuantile=False)


    Parameters
    ----------
    vecSpikeTimes1 : 1D array (float)
        spike times (in seconds) for condition 1.
    arrEventTimes1 : 1D or 2D array (float)
        event on times (s) for condition 1, or [T x 2] including event off times to calculate mean-rate difference.
    vecSpikeTimes2 : 1D array (float)
        spike times (in seconds) for condition 2.
    arrEventTimes2 : 1D or 2D array (float)
        event on times (s) for condition 2, or [T x 2] including event off times to calculate mean-rate difference.

    dblUseMaxDur : float
        window length for calculating ZETA: ignore all spikes beyond this duration after event onset
        (default: minimum of event onset to event onset)
    intResampNum : integer
        number of resamplings (default: 250)
        [Note: if your p-value is close to significance, you should increase this number to enhance the precision]
    boolPlot : boolean switch
        plotting switch (False: no plot, True: plot figure) (default: False)
    boolDirectQuantile: boolean
         switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
         [Note: requires many resamplings!]

    Returns
    -------
    dblZeta2P : float
        p-value based on Zenith of Event-based Time-locked Anomalies for two-sample comparison
    dZETA2 : dict
        additional information of ZETA test
            dblZetaP; p-value based on Zenith of Event-based Time-locked Anomalies (same as above)
            dblZETA; responsiveness z-score (i.e., >2 is significant)
            dblMeanZ; z-score for mean-rate stim/base difference (i.e., >2 is significant)
            dblMeanP; p-value based on mean-rate stim/base difference
            dblZETADeviation; temporal deviation value underlying ZETA
            dblZetaT; time corresponding to ZETA
            intZetaIdx; entry corresponding to ZETA
            vecMu1; average spiking rate values per event underlying t-test for condition 1
            vecMu2; average spiking rate values per event underlying t-test for condition 2
            vecSpikeT: timestamps of spike times (corresponding to vecRealDiff)
            vecRealDiff; difference between condition 1 and 2
            vecRealFrac1; cumulative spike vector of condition 1
            vecRealFrac2; cumulative spike vector of condition 2
            dblD_InvSign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            dblZetaT_InvSign; time corresponding to -ZETA
            intZetaIdx_InvSign; entry corresponding to -ZETA
            cellRandTime; timestamps for null-hypothesis resampled data
            cellRandDiff; null-hypothesis temporal deviation vectors of resampled data
            dblUseMaxDur; window length used to calculate ZETA

    Code by Jorrit Montijn

    Version history:
    1.0 - 25 Oct 2023 Translated to python [by Jorrit Montijn]

    """

    # %% build placeholder outputs
    dblZetaP = 1.0
    dZETA = dict()

    # fill dZETA
    # ZETA significance
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = None
    # mean-rate significance
    dZETA['dblMeanZ'] = None
    dZETA['dblMeanP'] = None
    # data on ZETA peak
    dZETA['dblZETADeviation'] = None
    dZETA['dblZetaT'] = None
    dZETA['intZetaIdx'] = None
    # data underlying mean-rate test
    dZETA['vecMu1'] = None
    dZETA['vecMu2'] = None

    # inverse-sign ZETA
    dZETA['dblD_InvSign'] = None
    dZETA['dblZetaT_InvSign'] = None
    dZETA['intZetaIdx_InvSign'] = None

    # derived from calcZetaOne
    dZETA['vecSpikeT'] = None
    dZETA['vecRealDiff'] = None
    dZETA['vecRealFrac1'] = None
    dZETA['vecRealFrac2'] = None
    dZETA['cellRandTime'] = None
    dZETA['cellRandDiff'] = None
    # dZETA['dblZetaP'] = None #<-updates automatically
    # dZETA['dblZETA'] = None #<-updates automatically
    # dZETA['intZETAIdx'] = None #<-updates automatically

    # window used for analysis
    dZETA['dblUseMaxDur'] = None

    # %% prep data and assert inputs are correct

    # vecSpikeTimes1 must be [S by 1] array
    assert (len(vecSpikeTimes1.shape) == 1 or vecSpikeTimes1.shape[1] == 1) and issubclass(
        vecSpikeTimes1.dtype.type, np.floating), "Input vecSpikeTimes1 is not a 1D float np.array with >2 spike times"
    vecSpikeTimes1 = np.sort(vecSpikeTimes1.flatten(), axis=0)

    # vecSpikeTimes2 must be [S by 1] array
    assert (len(vecSpikeTimes2.shape) == 1 or vecSpikeTimes2.shape[1] == 1) and issubclass(
        vecSpikeTimes2.dtype.type, np.floating), "Input vecSpikeTimes2 is not a 1D float np.array with >2 spike times"
    vecSpikeTimes2 = np.sort(vecSpikeTimes2.flatten(), axis=0)

    # ensure orientation and assert that arrEventTimes1 is a 1D or N-by-2 array of floats
    assert len(arrEventTimes1.shape) < 3 and issubclass(
        arrEventTimes1.dtype.type, np.floating), "Input arrEventTimes1 is not a 1D or 2D float np.array"
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
    # define event starts
    vecEventStarts1 = arrEventTimes1[:, 0]

    # ensure orientation and assert that arrEventTimes2 is a 1D or N-by-2 array of floats
    assert len(arrEventTimes2.shape) < 3 and issubclass(
        arrEventTimes2.dtype.type, np.floating), "Input arrEventTimes2 is not a 1D or 2D float np.array"
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
    # define event starts
    vecEventStarts2 = arrEventTimes2[:, 0]

    # is stop supplied?
    if len(arrEventTimes1.shape) > 1 and arrEventTimes1.shape[1] > 1 and len(arrEventTimes2.shape) > 1 and arrEventTimes2.shape[1] > 1:
        boolStopSupplied = True
        arrEventOnDur1 = arrEventTimes1[:, 1] - arrEventTimes1[:, 0]
        assert np.all(arrEventOnDur1 > 0), "at least one event in arrEventTimes1 has a negative duration"

        arrEventOnDur2 = arrEventTimes2[:, 1] - arrEventTimes2[:, 0]
        assert np.all(arrEventOnDur2 > 0), "at least one event in arrEventTimes2 has a negative duration"

        # trial dur
        if dblUseMaxDur is None:
            dblUseMaxDur = min(np.min(arrEventOnDur1), np.min(arrEventOnDur2))
    else:
        boolStopSupplied = False
        dblMeanZ = np.nan
        dblMeanP = np.nan

    # trial dur
    if dblUseMaxDur is None:
        dblUseMaxDur = min(np.min(np.diff(arrEventTimes1[:, 0])), np.min(np.diff(arrEventTimes2[:, 0])))
    else:
        dblUseMaxDur = np.float64(dblUseMaxDur)
        assert dblUseMaxDur.size == 1 and dblUseMaxDur > 0, "dblUseMaxDur is not a positive scalar float"

    # get resampling num
    if intResampNum is None:
        intResampNum = np.int64(250)
    else:
        intResampNum = np.int64(intResampNum)
        assert intResampNum.size == 1 and intResampNum > 1, "intResampNum is not a positive integer"

    # plotting
    if boolPlot is None:
        boolPlot = False
    else:
        assert isinstance(boolPlot, bool), "boolPlot is not a boolean"

    # direct quantile comnputation
    if boolDirectQuantile is None:
        boolDirectQuantile = False
    else:
        assert isinstance(boolDirectQuantile, bool), "boolDirectQuantile is not a boolean"

    # to do: parallel computing
    boolParallel = False

    # %% calculate zeta
    vecEventStarts1 = arrEventTimes1[:, 0]
    vecEventStarts2 = arrEventTimes2[:, 0]
    # if len(vecEventStarts1) > 1 and (len(vecSpikeTimes1)+len(vecSpikeTimes2)) > 0 and dblUseMaxDur is not None and dblUseMaxDur > 0:
    dZETA_Two = calcZetaTwo(vecSpikeTimes1, vecEventStarts1, vecSpikeTimes2,
                            vecEventStarts2, dblUseMaxDur, intResampNum, boolDirectQuantile)

    # %% calculate zeta
    # update and unpack
    dZETA.update(dZETA_Two)
    vecSpikeT = dZETA['vecSpikeT']
    vecRealDiff = dZETA['vecRealDiff']
    dblZetaP = dZETA['dblZetaP']
    intZetaIdx = dZETA['intZetaIdx']

    # check if calculation is valid, otherwise return empty values
    if intZetaIdx is None:
        logging.warning("zetatest2: calculation failed, defaulting to p=1.0")
        return dblZetaP, dZETA

    # %% extract real outputs
    # get location
    dblZetaT = vecSpikeT[intZetaIdx]
    dblZETADeviation = vecRealDiff[intZetaIdx]

    # find peak of inverse sign
    intZetaIdx_InvSign = np.argmax(-np.sign(dblZETADeviation)*vecRealDiff)
    dblZetaT_InvSign = vecSpikeT[intZetaIdx_InvSign]
    dblD_InvSign = vecRealDiff[intZetaIdx_InvSign]

    # %% calculate mean-rate difference with t-test
    if boolStopSupplied:
        # calculate spike counts and durations during baseline and stimulus times
        vecRespBinsDur = np.sort(np.reshape(arrEventTimes1, -1))
        vecR, arrBins = np.histogram(vecSpikeTimes1, bins=vecRespBinsDur)
        vecD = np.diff(vecRespBinsDur)
        vecMu1 = np.divide(np.float64(vecR[0:len(vecR):2]), vecD[0:len(vecD):2])

        # calculate mean rates during off-times
        vecRespBinsDur = np.sort(np.reshape(arrEventTimes2, -1))
        vecR, arrBins = np.histogram(vecSpikeTimes2, bins=vecRespBinsDur)
        vecD = np.diff(vecRespBinsDur)
        vecMu2 = np.divide(np.float64(vecR[0:len(vecR):2]), vecD[0:len(vecD):2])

        # get metrics
        dblMeanP = stats.ttest_rel(vecMu1, vecMu2)[1]
        dblMeanZ = -stats.norm.ppf(dblMeanP/2)

    # %% build output dictionary
    # fill dZETA
    dZETA['dblZETADeviation'] = dblZETADeviation
    dZETA['dblZetaT'] = dblZetaT
    if boolStopSupplied:
        dZETA['dblMeanZ'] = dblMeanZ
        dZETA['dblMeanP'] = dblMeanP
        dZETA['vecMu1'] = vecMu1
        dZETA['vecMu2'] = vecMu2

    # inverse-sign ZETA
    dZETA['dblD_InvSign'] = dblD_InvSign
    dZETA['dblZetaT_InvSign'] = dblZetaT_InvSign
    dZETA['intZetaIdx_InvSign'] = intZetaIdx_InvSign
    # window used for analysis
    dZETA['dblUseMaxDur'] = dblUseMaxDur

    # %% plot
    if boolPlot:
        plotzeta2(vecSpikeTimes1, vecEventStarts1, vecSpikeTimes2, vecEventStarts2, dZETA)

    # %% return outputs
    return dblZetaP, dZETA

# %% time-series zeta


def zetatstest(vecTime, vecValue, arrEventTimes,
               dblUseMaxDur=None, intResampNum=100, boolPlot=False, dblJitterSize=2.0, boolDirectQuantile=False, boolStitch=True):
    """
    Calculates responsiveness index zeta for timeseries data

    Montijn, J.S., Seignette, K., Howlett, M.H., Cazemier, J.L., Kamermans, M., Levelt, C.N.,
    and Heimel, J.A. (2021). A parameter-free statistical test for neuronal responsiveness.
    eLife 10, e71969.

    Montijn J.S. and Heimel J.A. (2022). A novel and highly sensitive statistical test for calcium imaging.
   FENS meeting 2022, Poster S03-480

    Syntax:
    dblZetaP,dZETA = zetatstest(vecTime, vecValue, arrEventTimes,
                 dblUseMaxDur=None, intResampNum=100, boolPlot=False, dblJitterSize=2.0, boolDirectQuantile=False, boolStitch=True):

    Parameters
    ----------
    vecTime [N x 1]: 1D array (float)
        timestamps in seconds corresponding to entries in vecValue.
    vecValue [N x 1] : 1D array (float)
        values (e.g., dF/F0 activity).
    arrEventTimes : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate mean-rate difference.
    dblUseMaxDur : float
        window length for calculating ZETA: ignore all entries beyond this duration after event onset
        (default: minimum of event onset to event onset)
    intResampNum : integer
        number of resamplings (default: 100)
        [Note: if your p-value is close to significance, you should increase this number to enhance the precision]
    boolPlot : boolean switch
        plotting switch (False: no plot, True: plot figure) (default: False)
    dblJitterSize; float
        sets the temporal jitter window relative to dblUseMaxDur (default: 2.0)
    boolDirectQuantile: boolean
         switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
         [Note: requires many resamplings!]
    boolStitch: boolean
        switch to perform data stitching (default: True)


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
            dblLatencyZETA; time corresponding to ZETA
            intZETAIdx; entry corresponding to ZETA
            vecMu_Dur; mean activity per trial during stim (used for mean-rate test)
            vecMu_Base; mean activity per trial during baseline (used for mean-rate test)
            dblD_InvSign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            dblLatencyInvZETA; time corresponding to -ZETA
            intIdx_InvSign; entry corresponding to -ZETA
            vecRealTime: timestamps of event-centered time-series values (corresponding to vecRealDeviation)
            vecRealDeviation; temporal deviation vector of data
            vecRealFrac; cumulative distribution of spike times
            vecRealFracLinear; linear baseline of cumulative distribution
            matRandDeviation; baseline temporal deviation matrix of jittered data
            dblUseMaxDur; window length used to calculate ZETA


    Code by Jorrit Montijn

    Version history:
    1.0 - 24 August 2023 Translation to python by Jorrit Montijn
    1.1 - 25 Oct 2023 Removed jitter distro switch [by Jorrit Montijn]

    """
    # %% build placeholder outputs
    dblZetaP = 1.0
    dZETA = dict()

    # fill dZETA
    # ZETA significance
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = None
    # mean-rate significance
    dZETA['dblMeanZ'] = None
    dZETA['dblMeanP'] = None
    # data on ZETA peak
    dZETA['dblZETADeviation'] = None
    dZETA['dblLatencyZETA'] = None
    dZETA['intZETAIdx'] = None
    # data underlying mean-rate test
    dZETA['vecMu_Dur'] = None
    dZETA['vecMu_Base'] = None
    # inverse-sign ZETA
    dZETA['dblD_InvSign'] = None
    dZETA['dblLatencyInvZETA'] = None
    dZETA['intIdx_InvSign'] = None

    # derived from calcZetaOne
    dZETA['vecRealTime'] = None
    dZETA['vecRealDeviation'] = None
    dZETA['vecRealFrac'] = None
    dZETA['vecRealFracLinear'] = None
    dZETA['cellRandTime'] = None
    dZETA['cellRandDeviation'] = None
    # dZETA['dblZetaP'] = None #<-updates automatically
    # dZETA['dblZETA'] = None #<-updates automatically
    # dZETA['intZETAIdx'] = None #<-updates automatically

    # window used for analysis
    dZETA['dblUseMaxDur'] = None

    # %% prep data and assert inputs are correct

    # vecTime and vecValue must be [N by 1] arrays
    assert len(vecTime.shape) == len(
        vecValue.shape) and vecTime.shape == vecValue.shape, "vecTime and vecValue have different shapes"
    assert (len(vecTime.shape) == 1 or vecTime.shape[1] == 1) and issubclass(
        vecTime.dtype.type, np.floating), "Input vecTime is not a 1D float np.array with >2 spike times"
    vecTime = vecTime.flatten()
    vecValue = vecValue.flatten()
    vecReorder = np.argsort(vecTime, axis=0)
    vecTime = vecTime[vecReorder]
    vecValue = vecValue[vecReorder]

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

    # check if number of events and values is sufficient
    if vecTime.size < 3 or vecEventStarts.size < 3:
        if vecTime.size < 3:
            strMsg1 = f"Number of entries in time-series ({vecTime.size}) is too few to calculate zeta; "
        else:
            strMsg1 = ""
        if vecEventStarts.size < 3:
            strMsg2 = f"Number of events ({vecEventStarts.size}) is too few to calculate zeta; "
        else:
            strMsg2 = ""
        logging.warning("zetatstest: " + strMsg1 + strMsg2 + "defaulting to p=1.0")

        return dblZetaP, dZETA

    # is stop supplied?
    if len(arrEventTimes.shape) > 1 and arrEventTimes.shape[1] > 1:
        boolStopSupplied = True
        vecEventStops = arrEventTimes[:, 1]
        vecEventOnDur = arrEventTimes[:, 1] - arrEventTimes[:, 0]
        assert np.all(vecEventOnDur > 0), "at least one event in arrEventTimes has a non-positive duration"
        vecMu_Dur = np.zeros(vecEventStops.shape)
        vecMu_Base = np.zeros(vecEventStops.shape)

    else:
        boolStopSupplied = False
        dblMeanZ = np.nan
        dblMeanP = np.nan
        vecMu_Dur = []
        vecMu_Base = []

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
    if boolPlot is None:
        boolPlot = False
    else:
        assert isinstance(boolPlot, bool), "boolPlot is not a boolean"

    # jitter
    if dblJitterSize is None:
        dblJitterSize = np.float64(2.0)
    else:
        dblJitterSize = np.float64(dblJitterSize)
        assert dblJitterSize.size == 1 and dblJitterSize > 0, "dblJitterSize is not a postive scalar float"

    # direct quantile computation
    if boolDirectQuantile is None:
        boolDirectQuantile = False
    else:
        assert isinstance(boolDirectQuantile, bool), "boolDirectQuantile is not a boolean"

    # which jitter distro to use?
    if boolStitch is None:
        boolDirectQuantile = True
    else:
        assert isinstance(boolStitch, bool), "boolStitch is not a boolean"

    # sampling interval
    dblSamplingInterval = np.median(np.diff(vecTime))

    # %% check data length
    dblDataT0 = np.min(vecTime)
    dblReqT0 = np.min(vecEventStarts) - dblJitterSize*dblUseMaxDur
    if dblDataT0 > dblReqT0:
        logging.warning("zetatstest: leading data preceding first event is insufficient for maximal jittering")

    dblDataT_end = np.max(vecTime)
    dblReqT_end = np.max(vecEventStarts) + dblJitterSize*dblUseMaxDur + dblUseMaxDur
    if dblDataT_end < dblReqT_end:
        logging.warning("zetatstest: lagging data after last event is insufficient for maximal jittering")

    # %% calculate zeta
    dZETA_One = calcTsZetaOne(vecTime, vecValue, vecEventStarts, dblUseMaxDur, intResampNum,
                              boolDirectQuantile, dblJitterSize, boolStitch)

    # update and unpack
    dZETA.update(dZETA_One)
    vecRealTime = dZETA['vecRealTime']
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
        logging.warning("zetatstest: calculation failed, defaulting to p=1.0")
        return dblZetaP, dZETA

    # %% extract real outputs
    # get location
    dblLatencyZETA = vecRealTime[intZETAIdx]
    dblZETADeviation = vecRealDeviation[intZETAIdx]

    # find peak of inverse sign
    intIdx_InvSign = np.argmax(-np.sign(dblZETADeviation)*vecRealDeviation)
    dblLatencyInvZETA = vecRealTime[intIdx_InvSign]
    dblD_InvSign = vecRealDeviation[intIdx_InvSign]

    # %% calculate mean-rate difference
    if boolStopSupplied:
        # pre-allocate
        intTimeNum = len(vecTime)-1

        # go through trials to build spike time vector
        for intEvent, dblStimStartT in enumerate(vecEventStarts):
            # %% get original times
            dblStimStopT = vecEventStops[intEvent]
            dblBaseStopT = dblStimStartT + dblUseMaxDur
            if (dblBaseStopT - dblStimStopT) <= 0:
                raise Exception(
                    "Input error: event stop times do not precede the next stimulus' start time")

            intStartT = np.max([0, findfirst(vecTime > dblStimStartT) - 1])
            intStopT = np.min([intTimeNum, findfirst(vecTime > dblStimStopT)+1])
            intEndT = np.min([intTimeNum, findfirst(vecTime > dblBaseStopT)+1])
            vecSelectFramesBase = np.arange(intStopT, intEndT)
            vecSelectFramesStim = np.arange(intStartT, intStopT)

           #  %% get data
            vecUseBaseTrace = vecValue[vecSelectFramesBase]
            vecUseStimTrace = vecValue[vecSelectFramesStim]

            # %% get activity
            vecMu_Base[intEvent] = np.mean(vecUseBaseTrace)
            vecMu_Dur[intEvent] = np.mean(vecUseStimTrace)

        # get metrics
        indUseTrials = np.logical_and(~np.isnan(vecMu_Dur), ~np.isnan(vecMu_Base))
        vecMu_Dur = vecMu_Dur[indUseTrials]
        vecMu_Base = vecMu_Base[indUseTrials]
        dblMeanP = stats.ttest_rel(vecMu_Dur, vecMu_Base)[1]
        dblMeanZ = -stats.norm.ppf(dblMeanP/2)

    # %% build output structure
    # fill dZETA
    dZETA['dblZETADeviation'] = dblZETADeviation
    dZETA['dblLatencyZETA'] = dblLatencyZETA
    if boolStopSupplied:
        dZETA['dblMeanZ'] = dblMeanZ
        dZETA['dblMeanP'] = dblMeanP
        dZETA['vecMu_Dur'] = vecMu_Dur
        dZETA['vecMu_Base'] = vecMu_Base

    # inverse-sign ZETA
    dZETA['dblD_InvSign'] = dblD_InvSign
    dZETA['dblLatencyInvZETA'] = dblLatencyInvZETA
    dZETA['intIdx_InvSign'] = intIdx_InvSign
    # window used for analysis
    dZETA['dblUseMaxDur'] = dblUseMaxDur

    # %% plot
    if boolPlot:
        plottszeta(vecTime, vecValue, vecEventStarts, dZETA)

    # %% return
    return dblZetaP, dZETA

# %% zetatest


def zetatest(vecSpikeTimes, arrEventTimes,
             dblUseMaxDur=None, intResampNum=100, boolPlot=False, dblJitterSize=2.0,
             tplRestrictRange=(-np.inf, np.inf), boolStitch=True,
             boolDirectQuantile=False, boolReturnRate=False):
    """
    Calculates neuronal responsiveness index ZETA.

    Montijn, J.S., Seignette, K., Howlett, M.H., Cazemier, J.L., Kamermans, M., Levelt, C.N.,
    and Heimel, J.A. (2021). A parameter-free statistical test for neuronal responsiveness.
    eLife 10, e71969.

    Syntax:
    dblZetaP,dZETA,dRate,vecLatencies = zetatest(vecSpikeTimes,arrEventTimes,
                                                   dblUseMaxDur=None, intResampNum=100, boolPlot=False, dblJitterSize=2.0,
                                                   tplRestrictRange=(-np.inf, np.inf), boolStitch=True,
                                                   boolDirectQuantile=False, boolReturnRate=False):

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
    boolPlot : boolean switch
        plotting switch (False: no plot, True: plot figure) (default: False)
    dblJitterSize; float
        sets the temporal jitter window relative to dblUseMaxDur (default: 2.0)
    tplRestrictRange : 2-element tuple
        temporal range within which to restrict onset/peak latencies (default: [-inf inf])
    boolStitch; boolean
        switch to use data-stitching to ensure continuous time (default: True)
    boolDirectQuantile: boolean
         switch to use the empirical null-distribution rather than the Gumbel approximation (default: False)
         [Note: requires many resamplings!]
    boolReturnRate : boolean
        switch to return dictionary with spiking rate features [note: return-time is much faster if this is False]

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
            dblLatencyZETA; time corresponding to ZETA
            intZETAIdx; entry corresponding to ZETA
            vecMu_Dur; spiking rate per trial during stim (used for mean-rate test)
            vecMu_Pre; spiking rate per trial during baseline (used for mean-rate test)
            dblD_InvSign; largest deviation of inverse sign to ZETA (i.e., -ZETA)
            dblLatencyInvZETA; time corresponding to -ZETA
            intIdx_InvSign; entry corresponding to -ZETA
            vecSpikeT: timestamps of spike times (corresponding to vecRealDeviation)
            vecRealDeviation; temporal deviation vector of data
            vecRealFrac; cumulative distribution of spike times
            vecRealFracLinear; linear baseline of cumulative distribution
            cellRandTime; jittered spike times corresponding to cellRandDeviation
            cellRandDeviation; baseline temporal deviation matrix of jittered data
            dblUseMaxDur; window length used to calculate ZETA
            vecLatencies; 4-element array with latency times for different events:
                1) Latency of ZETA [same as dblZETADeviation]
                2) Latency of largest z-score with inverse sign to ZETA (same as dblLatencyInvZETA])
                3) Peak time of instantaneous firing rate (same as dRate['dblLatencyPeak'])
                4) Onset time, defined as the first crossing of peak half-height (same as dRate['dblLatencyPeakOnset'])
            vecLatencyVals; values corresponding to above latencies (ZETA, -ZETA, rate at peak, rate at onset)

    dRate : dict (empty if boolReturnRate was not set to True)
        additional parameters of the firing rate, return with boolReturnRate
            vecRate; instantaneous spiking rates (like a PSTH)
            vecT; time-points corresponding to vecRate (same as dZETA.vecSpikeT)
            vecM; Mean of multi-scale derivatives
            vecScale; timescales used to calculate derivatives
            matMSD; multi-scale derivatives matrix
            vecV; values on which vecRate is calculated (same as dZETA.vecZ)
        Data on the peak and onset:
            dblLatencyPeak; time of peak (in seconds) [vecLatencies entry #3]
            dblPeakWidth; duration of peak (in seconds) [vecLatencies entry #3]
            vecPeakStartStop; start and stop time of peak (in seconds) [vecLatencies entry #3]
            intPeakLoc; spike index of peak (corresponding to dZETA.vecSpikeT) [vecLatencies entry #3]
            vecPeakStartStopIdx; spike indices of peak start/stop (corresponding to dZETA.vecSpikeT) [vecLatencies entry #3]
            dblLatencyPeakOnset: latency for peak onset [vecLatencies entry #4]
    Code by Jorrit Montijn, Guido Meijer & Alexander Heimel

    Version history:
    2.5 - 17 June 2020 Translated to python [Alexander Heimel]
    2.5.1 - 18 February 2022 Bugfix of 1D arrEventTimes [by Guido Meijer]
    2.6 - 20 February 2022 Refactoring of python code [by Guido Meijer]
    3.0 - 16 Aug 2023 New port of zetatest to Python [by Jorrit Montijn]
    3.1 - 15 Sept 2023 Changed latency variable names, added intUseJitterDistro switch [by Jorrit Montijn]
    3.2 - 25 Oct 2023 Removed intUseJitterDistro switch [by Jorrit Montijn]

    """

    # %% build placeholder outputs
    dblZetaP = 1.0
    dZETA = dict()
    dRate = dict()
    vecLatencies = np.empty((4, 1))
    vecLatencies.fill(np.nan)
    vecLatencyVals = vecLatencies

    # fill dZETA
    # ZETA significance
    dZETA['dblZetaP'] = dblZetaP
    dZETA['dblZETA'] = None
    # mean-rate significance
    dZETA['dblMeanZ'] = None
    dZETA['dblMeanP'] = None
    # data on ZETA peak
    dZETA['dblZETADeviation'] = None
    dZETA['dblLatencyZETA'] = None
    dZETA['intZETAIdx'] = None
    # data underlying mean-rate test
    dZETA['vecMu_Dur'] = None
    dZETA['vecMu_Pre'] = None
    # inverse-sign ZETA
    dZETA['dblD_InvSign'] = None
    dZETA['dblLatencyInvZETA'] = None
    dZETA['intIdx_InvSign'] = None

    # derived from calcZetaOne
    dZETA['vecSpikeT'] = None
    dZETA['vecRealDeviation'] = None
    dZETA['vecRealFrac'] = None
    dZETA['vecRealFracLinear'] = None
    dZETA['cellRandTime'] = None
    dZETA['cellRandDeviation'] = None
    # dZETA['dblZetaP'] = None #<-updates automatically
    # dZETA['dblZETA'] = None #<-updates automatically
    # dZETA['intZETAIdx'] = None #<-updates automatically

    # window used for analysis
    dZETA['dblUseMaxDur'] = None
    # copy of latency vectors
    dZETA['vecLatencies'] = vecLatencies
    dZETA['vecLatencyVals'] = vecLatencyVals

    # fill dRate
    dRate['vecRate'] = None
    dRate['vecT'] = None
    dRate['vecM'] = None
    dRate['vecScale'] = None
    dRate['matMSD'] = None
    dRate['vecV'] = None
    dRate['dblLatencyPeak'] = None
    dRate['dblPeakWidth'] = None
    dRate['vecPeakStartStop'] = None
    dRate['intPeakLoc'] = None
    dRate['vecPeakStartStopIdx'] = None
    dRate['dblLatencyPeakOnset'] = None

    # %% prep data and assert inputs are correct

    # vecSpikeTimes must be [S by 1] array
    assert (len(vecSpikeTimes.shape) == 1 or vecSpikeTimes.shape[1] == 1) and issubclass(
        vecSpikeTimes.dtype.type, np.floating), "Input vecSpikeTimes is not a 1D float np.array with >2 spike times"
    vecSpikeTimes = np.sort(vecSpikeTimes.flatten(), axis=0)

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
    if vecSpikeTimes.size < 3 or vecEventStarts.size < 3:
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
    if boolPlot is None:
        boolPlot = False
    else:
        assert isinstance(boolPlot, bool), "boolPlot is not a boolean"

    # jitter
    if dblJitterSize is None:
        dblJitterSize = np.float64(2.0)
    else:
        dblJitterSize = np.float64(dblJitterSize)
        assert dblJitterSize.size == 1 and dblJitterSize > 0, "dblJitterSize is not a postive scalar float"

    # restrict range
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
    if boolPlot is True and boolReturnRate is False:
        boolReturnRate = True
        logging.warning(
            "zetatest: boolReturnRate was False, but you requested plotting, so boolReturnRate is now set to True")

    # to do: parallel computing
    boolParallel = False

    # %% calculate zeta
    dZETA_One = calcZetaOne(vecSpikeTimes, vecEventStarts, dblUseMaxDur, intResampNum,
                            boolDirectQuantile, dblJitterSize, boolStitch, boolParallel)

    # update and unpack
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
        return dblZetaP, dZETA, dRate

    # %% extract real outputs
    # get location
    dblLatencyZETA = vecSpikeT[intZETAIdx]
    dblZETADeviation = vecRealDeviation[intZETAIdx]

    # find peak of inverse sign
    intIdx_InvSign = np.argmax(-np.sign(dblZETADeviation)*vecRealDeviation)
    dblLatencyInvZETA = vecSpikeT[intIdx_InvSign]
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
        dblFirstPreDur = dblStart1 - np.max(dblStart1 - np.median(vecD[1:len(vecD):2]), initial=0) + np.finfo(float).eps
        dblR1 = np.sum(np.logical_and(vecSpikeTimes > (dblStart1 - dblFirstPreDur), vecSpikeTimes < dblStart1))
        vecCounts = np.concatenate([[dblR1], vecR[1:len(vecR):2]])
        vecDurs = np.concatenate([[dblFirstPreDur], vecD[1:len(vecD):2]])
        vecMu_Pre = np.divide(vecCounts, vecDurs)

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
        if vecRate is not None:
            # get IFR peak
            dPeak = getPeak(vecRate, dRate['vecT'], tplRestrictRange=tplRestrictRange)
            dRate.update(dPeak)
            if dRate['dblLatencyPeak'] is not None and ~np.isnan(dRate['dblLatencyPeak']):
                # assign array data
                intZetaIdxRate = min(max(0, intZETAIdx-1), len(vecRate)-1)
                intZetaIdxInvRate = min(max(0, intIdx_InvSign-1), len(vecRate)-1)

                # get onset
                dOnset = getOnset(vecRate, dRate['vecT'], dRate['dblLatencyPeak'], tplRestrictRange)
                dRate['dblLatencyPeakOnset'] = dOnset['dblLatencyPeakOnset']
                vecLatencies = [dblLatencyZETA, dblLatencyInvZETA,
                                dRate['dblLatencyPeak'], dOnset['dblLatencyPeakOnset']]
                vecLatencyVals = [vecRate[intZetaIdxRate], vecRate[intZetaIdxInvRate],
                                  vecRate[dPeak['intPeakLoc']], dOnset['dblValue']]

    # %% build output dictionary
    # fill dZETA
    dZETA['dblZETADeviation'] = dblZETADeviation
    dZETA['dblLatencyZETA'] = dblLatencyZETA
    if boolStopSupplied:
        dZETA['dblMeanZ'] = dblMeanZ
        dZETA['dblMeanP'] = dblMeanP
        dZETA['vecMu_Dur'] = vecMu_Dur
        dZETA['vecMu_Pre'] = vecMu_Pre

    # inverse-sign ZETA
    dZETA['dblD_InvSign'] = dblD_InvSign
    dZETA['dblLatencyInvZETA'] = dblLatencyInvZETA
    dZETA['intIdx_InvSign'] = intIdx_InvSign
    # window used for analysis
    dZETA['dblUseMaxDur'] = dblUseMaxDur
    # copy of latency vectors
    dZETA['vecLatencies'] = vecLatencies
    dZETA['vecLatencyVals'] = vecLatencyVals

    # %% plot
    if boolPlot:
        plotzeta(vecSpikeTimes, vecEventStarts, dZETA, dRate)

    # %% return outputs
    return dblZetaP, dZETA, dRate

# %% IFR


def ifr(vecSpikeTimes, vecEventTimes,
        dblUseMaxDur=None, dblSmoothSd=2.0, dblMinScale=None, dblBase=1.5, boolParallel=False):
    """Returns instantaneous firing rates. Syntax:
        ifr(vecSpikeTimes,vecEventTimes,
               dblUseMaxDur=None, dblSmoothSd=2, dblMinScale=None, dblBase=1.5)

    Required input:
        - vecSpikeTimes [S x 1]: spike times (s)
        - vecEventTimes [T x 1]: event on times (s), or [T x 2] including event off times

    Optional inputs:
        - dblUseMaxDur: float (s), ignore all spikes beyond this duration after stimulus onset
                                    [default: median of trial start to trial start]
        - dblSmoothSd: float, Gaussian SD of smoothing kernel (in # of bins) [default: 2]
        - dblMinScale: minimum derivative scale in seconds [default: round(log(1/1000) / log(dblBase))]
        - dblBase: critical value for locally dynamic derivative [default: 1.5]

    Outputs:
        - vecTime: array with timestamps
        - vecRate: array with instantaneous firing rates
        - dIFR; dictionary with entries:
            - vecTime: same as first output
            - vecRate: same as second output
            - vecDeviation: ZETA's deviation vector
            - vecScale: temporal scales used to calculate the multi-scale derivative

    Version history:
    1.0 - June 24, 2020 Created by Jorrit Montijn, translated to python by Alexander Heimel
    2.0 - August 22, 2023 Updated translation [by JM]
   """
    # %% prep data and assert inputs are correct
    # pre-allocate outputs
    vecTime = np.empty(0)
    vecRate = np.empty(0)
    dIFR = dict()
    dIFR['vecTime'] = vecTime
    dIFR['vecRate'] = vecRate
    dIFR['vecDeviation'] = np.empty(0)
    dIFR['vecScale'] = np.empty(0)

    # vecSpikeTimes must be [S by 1] array
    assert (len(vecSpikeTimes.shape) == 1 or vecSpikeTimes.shape[1] == 1) and issubclass(
        vecSpikeTimes.dtype.type, np.floating), "Input vecSpikeTimes is not a 1D float np.array with >2 spike times"
    vecSpikeTimes = np.sort(vecSpikeTimes.flatten(), axis=0)

    # ensure orientation and assert that arrEventTimes is a 1D or N-by-2 array of floats
    assert len(vecEventTimes.shape) < 3 and issubclass(
        vecEventTimes.dtype.type, np.floating), "Input vecEventTimes is not a 1D or 2D float np.array"
    if len(vecEventTimes.shape) > 1:
        if vecEventTimes.shape[1] < 3:
            pass
        elif vecEventTimes.shape[0] < 3:
            vecEventTimes = vecEventTimes.T
        else:
            raise Exception(
                "Input error: vecEventTimes must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        vecEventTimes = np.reshape(vecEventTimes, (-1, 1))
    # define event starts
    vecEventStarts = vecEventTimes[:, 0]

    # check if number of events and spikes is sufficient
    if vecSpikeTimes.size < 3 or vecEventStarts.size < 3:
        if vecSpikeTimes.size < 3:
            strMsg1 = f"Number of spikes ({vecSpikeTimes.size}) is too few to calculate zeta; "
        else:
            strMsg1 = ""
        if vecEventStarts.size < 3:
            strMsg2 = f"Number of events ({vecEventStarts.size}) is too few to calculate zeta; "
        else:
            strMsg2 = ""
        logging.warning("zetatest: " + strMsg1 + strMsg2 + "defaulting to p=1.0")

        return vecTime, vecRate, dIFR

    # trial dur
    if dblUseMaxDur is None:
        dblUseMaxDur = np.min(np.diff(vecEventStarts))
    else:
        dblUseMaxDur = np.float64(dblUseMaxDur)
        assert dblUseMaxDur.size == 1 and dblUseMaxDur > 0, "dblUseMaxDur is not a positive scalar float"

    # parallel processing: to do
    boolUseParallel = False

    # %% get difference from uniform
    vecThisDeviation, vecThisSpikeFracs, vecThisFracLinear, vecThisSpikeTimes = getTempOffsetOne(
        vecSpikeTimes, vecEventStarts, dblUseMaxDur)
    intSpikeNum = vecThisSpikeTimes.size

    # check if sufficient spikes are present
    if vecThisDeviation.size < 3:
        logging.warning("ifr: too few spikes, returning empty variables")

        return vecTime, vecRate, dIFR

    # %% get multi-scale derivative
    intMaxRep = vecEventStarts.size
    dblMeanRate = (intSpikeNum/(dblUseMaxDur*intMaxRep))
    [vecRate, dMSD] = getMultiScaleDeriv(vecThisSpikeTimes, vecThisDeviation,
                                         dblSmoothSd=dblSmoothSd, dblMinScale=dblMinScale, dblBase=dblBase, dblMeanRate=dblMeanRate, dblUseMaxDur=dblUseMaxDur, boolParallel=boolParallel)

    # %% build output
    vecTime = dMSD['vecT']
    vecRate = vecRate  # unnecessary, but for clarity of output building
    dIFR = dict()
    dIFR['vecTime'] = vecTime
    dIFR['vecRate'] = vecRate
    dIFR['vecDeviation'] = vecThisDeviation
    dIFR['vecScale'] = dMSD['vecScale']
    return vecTime, vecRate, dIFR

# %% plotzeta2
def plotzeta2(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2, dZETA,
              intPlotRandSamples=50, intPlotSpikeNum=10000):
    '''
    Creates figure for two-sample ZETA-test

    Syntax:
    plotzeta2(vecSpikeTimes1, arrEventTimes1, vecSpikeTimes2, arrEventTimes2, dZETA,
              intPlotRandSamples=50, intPlotSpikeNum=10000)

    Parameters
    ----------
    vecSpikeTimes1 : 1D array (float)
        spike times (in seconds) for condition 1.
    arrEventTimes1 : 1D or 2D array (float)
        event on times (s) for condition 1, or [T x 2] including event off times to calculate mean-rate difference.
    vecSpikeTimes2 : 1D array (float)
        spike times (in seconds) for condition 2.
    arrEventTimes2 : 1D or 2D array (float)
        event on times (s) for condition 2, or [T x 2] including event off times to calculate mean-rate difference.
    dZETA : dict
        Output of zetatest2.
    intPlotRandSamples : int, optional
        Maximum number of random resampling to plot. The default is 50.
    intPlotSpikeNum : int, optional
        Maximum number of spikes to plot. The default is 10000.


    Code by Jorrit Montijn

    Version history:
    1.0 - 25 October 2023 Created by Jorrit Montijn
    '''
    
    # %% check input
    
    # vecSpikeTimes1 must be [S by 1] array
    assert (len(vecSpikeTimes1.shape) == 1 or vecSpikeTimes1.shape[1] == 1) and issubclass(
        vecSpikeTimes1.dtype.type, np.floating), "Input vecSpikeTimes1 is not a 1D float np.array with >2 spike times"
    vecSpikeTimes1 = np.sort(vecSpikeTimes1.flatten(), axis=0)

    # vecSpikeTimes2 must be [S by 1] array
    assert (len(vecSpikeTimes2.shape) == 1 or vecSpikeTimes2.shape[1] == 1) and issubclass(
        vecSpikeTimes2.dtype.type, np.floating), "Input vecSpikeTimes2 is not a 1D float np.array with >2 spike times"
    vecSpikeTimes2 = np.sort(vecSpikeTimes2.flatten(), axis=0)

    # ensure orientation and assert that arrEventTimes1 is a 1D or N-by-2 array of floats
    assert len(arrEventTimes1.shape) < 3 and issubclass(
        arrEventTimes1.dtype.type, np.floating), "Input arrEventTimes1 is not a 1D or 2D float np.array"
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
    # define event starts
    vecEventStarts1 = arrEventTimes1[:, 0]

    # ensure orientation and assert that arrEventTimes2 is a 1D or N-by-2 array of floats
    assert len(arrEventTimes2.shape) < 3 and issubclass(
        arrEventTimes2.dtype.type, np.floating), "Input arrEventTimes2 is not a 1D or 2D float np.array"
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
    # define event starts
    vecEventStarts2 = arrEventTimes2[:, 0]

    # unpack dZETA
    try:
        dblUseMaxDur = dZETA['dblUseMaxDur']
        dblZETA = dZETA['dblZETA']
        dblZetaP = dZETA['dblZetaP']
        dblZETADeviation = dZETA['dblZETADeviation']
        dblZetaT = dZETA['dblZetaT']
        intZetaIdx = dZETA['intZetaIdx']
        
        dblD_InvSign = dZETA['dblD_InvSign']
        dblZetaT_InvSign = dZETA['dblZetaT_InvSign']
        intZetaIdx_InvSign = dZETA['intZetaIdx_InvSign']

        dblMeanZ = dZETA['dblMeanZ']
        dblMeanP = dZETA['dblMeanP']

        vecSpikeT = dZETA['vecSpikeT']
        vecRealDiff = dZETA['vecRealDiff']
        vecRealFrac1 = dZETA['vecRealFrac1']
        vecRealFrac2 = dZETA['vecRealFrac2']
        cellRandTime = dZETA['cellRandTime']
        cellRandDiff = dZETA['cellRandDiff']

    except:
        raise Exception(
            "plotzeta2 error: information is missing from dZETA dictionary")

    # %% plot
    # Plot maximally 50 traces (or however man y are requested)
    intPlotRandSamples = np.min([len(cellRandTime), intPlotRandSamples])

    # Calculate optimal DPI depending on the monitor size
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 15

    # Create figure
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=dpi)

    # reduce spikes
    if vecSpikeTimes1.size > intPlotSpikeNum or vecSpikeTimes2.size > intPlotSpikeNum:
        dblReduceSpikesBy = min(vecSpikeTimes1.size / intPlotSpikeNum, vecSpikeTimes2.size / intPlotSpikeNum)
        intPlotSpikeNum1 = np.round(dblReduceSpikesBy * vecSpikeTimes1.size)
        intPlotSpikeNum2 = np.round(dblReduceSpikesBy * vecSpikeTimes2.size)
        vecSpikeT1_reduced = vecSpikeTimes1[np.round(np.linspace(0, vecSpikeTimes1.size-1, intPlotSpikeNum1))]
        vecSpikeT2_reduced = vecSpikeTimes1[np.round(np.linspace(0, vecSpikeTimes2.size-1, intPlotSpikeNum2))]
    else:
        vecSpikeT1_reduced = vecSpikeTimes1
        vecSpikeT2_reduced = vecSpikeTimes2

    # top left: raster 1
    for i, t in enumerate(vecEventStarts1):
        idx = np.bitwise_and(vecSpikeT1_reduced >= t, vecSpikeT1_reduced <= t + dblUseMaxDur)
        event_spks = vecSpikeT1_reduced[idx]
        ax1.vlines(event_spks - t, i + 1, i, color='k', lw=0.3)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial #', title='Spike raster plot 1')

    # bottom left: raster 2
    for i, t in enumerate(vecEventStarts2):
        idx = np.bitwise_and(vecSpikeT2_reduced >= t, vecSpikeT2_reduced <= t + dblUseMaxDur)
        event_spks = vecSpikeT2_reduced[idx]
        ax3.vlines(event_spks - t, i + 1, i, color='k', lw=0.3)
    ax3.set(xlabel='Time after event (s)', ylabel='Trial #', title='Spike raster plot 2')

    
    # top right: cumulative sums
    ax3.plot(vecSpikeT, vecRealFrac1)
    ax3.plot(vecSpikeT, vecRealFrac2)
    ax3.set(xlabel='Time after event (s)', ylabel='Scaled cumulative spiking density (s)')

    # bottom right: deviation with random jitters
    for i in range(intPlotRandSamples-1):
        ax4.plot(cellRandTime[i], cellRandDiff[i], color=[0.7, 0.7, 0.7])
    ax4.plot(vecSpikeT, vecRealDiff)
    ax4.plot(dblZetaT, dblZETADeviation, 'bx')
    ax4.plot(dblZetaT_InvSign, dblD_InvSign, 'b*')
    ax4.set(xlabel='Time after event (s)', ylabel='Difference in cumulative density (s)')
    if dblMeanZ is not None:
        ax4.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f}), z(Hz)={dblMeanZ:.3f} (p={dblMeanP:.3f})')
    else:
        ax4.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f})')

    f.tight_layout()
    plt.show()

# %% plotzeta
def plotzeta(vecSpikeTimes, arrEventTimes, dZETA, dRate,
             intPlotRandSamples=50, intPlotSpikeNum=10000):
    '''
    Creates figure for ZETA-test analysis

    Syntax:
    plotzeta(vecSpikeTimes, arrEventTimes, dZETA, dRate, intPlotRandSamples=50, intPlotSpikeNum=10000)

    Parameters
    ----------
    vecSpikeTimes : 1D array (float)
        spike times (in seconds).
    arrEventTimes : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate on/off difference.
    dZETA : dict
        Output of zetatest.
    dRate : dict
        Output of zetatest.
    intPlotRandSamples : int, optional
        Maximum number of random resampling to plot. The default is 50.
    intPlotSpikeNum : int, optional
        Maximum number of spikes to plot. The default is 10000.


    Code by Jorrit Montijn

    Version history:
    1.0 - 07 September 2023 Created by Jorrit Montijn
    '''

    # %% check input
    # vecSpikeTimes must be [S by 1] array
    assert (len(vecSpikeTimes.shape) == 1 or vecSpikeTimes.shape[1] == 1) and issubclass(
        vecSpikeTimes.dtype.type, np.floating), "Input vecSpikeTimes is not a 1D float np.array with >2 spike times"
    vecSpikeTimes = np.sort(vecSpikeTimes.flatten(), axis=0)

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

    # unpack dZETA
    try:
        dblUseMaxDur = dZETA['dblUseMaxDur']
        dblZETA = dZETA['dblZETA']
        dblZetaP = dZETA['dblZetaP']
        dblZETADeviation = dZETA['dblZETADeviation']
        dblLatencyZETA = dZETA['dblLatencyZETA']

        dblD_InvSign = dZETA['dblD_InvSign']
        dblLatencyInvZETA = dZETA['dblLatencyInvZETA']
        intIdx_InvSign = dZETA['intIdx_InvSign']

        dblMeanZ = dZETA['dblMeanZ']
        dblMeanP = dZETA['dblMeanP']

        vecSpikeT = dZETA['vecSpikeT']
        vecRealDeviation = dZETA['vecRealDeviation']
        vecRealFrac = dZETA['vecRealFrac']
        vecRealFracLinear = dZETA['vecRealFracLinear']
        cellRandTime = dZETA['cellRandTime']
        cellRandDeviation = dZETA['cellRandDeviation']
        intZETAIdx = dZETA['intZETAIdx']

    except:
        raise Exception(
            "plotzeta error: information is missing from dZETA dictionary")

    # unpack dRate
    try:
        vecRate = dRate['vecRate']
        vecRateT = dRate['vecT']
        vecM = dRate['vecM']
        vecScale = dRate['vecScale']
        matMSD = dRate['matMSD']
        vecV = dRate['vecV']
        dblSmoothSd = dRate['dblSmoothSd']
        dblMeanRate = dRate['dblMeanRate']
    except:
        raise Exception(
            "plotzeta error: information is missing from dRate dictionary")

    # %% plot
    # Plot maximally 50 traces (or however man y are requested)
    intPlotRandSamples = np.min([len(cellRandTime), intPlotRandSamples])

    # Calculate optimal DPI depending on the monitor size
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 15

    # Create figure
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=dpi)

    # top left: raster
    if vecSpikeTimes.size > intPlotSpikeNum:
        vecSpikeT_reduced = vecSpikeTimes[np.round(np.linspace(0, vecSpikeTimes.size-1, intPlotSpikeNum))]
    else:
        vecSpikeT_reduced = vecSpikeTimes

    for i, t in enumerate(vecEventStarts):
        idx = np.bitwise_and(vecSpikeT_reduced >= t, vecSpikeT_reduced <= t + dblUseMaxDur)
        event_spks = vecSpikeT_reduced[idx]
        ax1.vlines(event_spks - t, i + 1, i, color='k', lw=0.3)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial #', title='Spike raster plot')

    # top right: psth
    peth, binned_spikes = calculatePeths(vecSpikeTimes, np.ones(vecSpikeTimes.shape), [1],
                                         vecEventStarts, pre_time=0, post_time=dblUseMaxDur,
                                         bin_size=dblUseMaxDur/25, smoothing=0)
    ax2.errorbar(peth['tscale'], peth['means'][0, :], yerr=peth['sems'])
    ax2.set(xlabel='Time after event (s)', ylabel='spks/s',
            title='Mean spiking over trials')

    # bottom left: deviation with random jitters
    for i in range(intPlotRandSamples-1):
        ax3.plot(cellRandTime[i], cellRandDeviation[i], color=[0.7, 0.7, 0.7])
    ax3.plot(vecSpikeT, vecRealDeviation)
    ax3.plot(dblLatencyZETA, dblZETADeviation, 'bx')
    ax3.plot(dblLatencyInvZETA, dblD_InvSign, 'b*')
    ax3.set(xlabel='Time after event (s)', ylabel='Spiking density anomaly (s)')
    if dblMeanZ is not None:
        ax3.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f}), z(Hz)={dblMeanZ:.3f} (p={dblMeanP:.3f})')
    else:
        ax3.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f})')

    # bottom right: ifr
    if len(vecRateT) > 1000:
        vecSubset = np.round(np.linspace(0, len(vecRateT)-1, 1000)).astype(int)
        ax4.plot(vecRateT[vecSubset], vecRate[vecSubset])
    else:
        ax4.plot(vecRateT, vecRate)

    if dblMeanRate == 1:
        strLabelY = 'Time-locked activation (a.u.)'
    else:
        strLabelY = 'Spiking rate (Hz)'
    ax4.set(xlabel='Time after event (s)', ylabel=strLabelY, title='IFR (instantaneous firing rate)')

    # plot onsets
    vecLatencies = dZETA['vecLatencies']
    vecLatencyVals = dZETA['vecLatencyVals']
    if len(vecLatencies) > 2 and vecLatencies[2] is not None and ~np.isnan(vecLatencies[2]):
        # plot peak time
        ax4.plot(vecLatencies[2], vecLatencyVals[2], 'gx')

    if len(vecLatencies) > 3 and vecLatencies[3] is not None and ~np.isnan(vecLatencies[3]):
        # plot onset time
        ax4.plot(vecLatencies[3], vecLatencyVals[3], 'rx')

    f.tight_layout()
    plt.show()

# %% plottszeta


def plottszeta(vecTime, vecData, arrEventTimes, dZETA, intPlotRandSamples=50):
    '''


    Parameters
    ----------
    vecTime [N x 1]: 1D array (float)
        timestamps in seconds corresponding to entries in vecValue.
    vecData [N x 1] : 1D array (float)
        values (e.g., dF/F0 activity).
    arrEventTimes : 1D or 2D array (float)
        event on times (s), or [T x 2] including event off times to calculate on/off difference.
    dZETA : dict
        Output of zetatstest.
    intPlotRandSamples : int, optional
        Maximum number of random resampling to plot. The default is 50.


    Code by Jorrit Montijn

    Version history:
    1.0 - 07 September 2023 Created by Jorrit Montijn
    '''

    # %% prep data and assert inputs are correct

    # vecTime and vecValue must be [N by 1] arrays
    assert len(vecTime.shape) == len(
        vecData.shape) and vecTime.shape == vecData.shape, "vecTime and vecValue have different shapes"
    assert (len(vecTime.shape) == 1 or vecTime.shape[1] == 1) and issubclass(
        vecTime.dtype.type, np.floating), "Input vecTime is not a 1D float np.array with >2 spike times"
    vecTime = vecTime.flatten()
    vecData = vecData.flatten()
    vecReorder = np.argsort(vecTime, axis=0)
    vecTime = vecTime[vecReorder]
    vecData = vecData[vecReorder]

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
    vecEventTimes = arrEventTimes[:, 0]

    # check if number of events and values is sufficient
    if vecTime.size < 3 or vecEventTimes.size < 3:
        if vecTime.size < 3:
            strMsg1 = f"Number of entries in time-series ({vecTime.size}) is too few; "
        else:
            strMsg1 = ""
        if vecEventTimes.size < 3:
            strMsg2 = f"Number of events ({vecEventTimes.size}) is too few; "
        else:
            strMsg2 = ""
            logging.warning("plottszeta: " + strMsg1 + strMsg2 + "defaulting to p=1.0")

    # unpack dZETA
    try:
        # ZETA significance
        dblZetaP = dZETA['dblZetaP']
        dblZETA = dZETA['dblZETA']
        # mean-rate significance
        dblMeanZ = dZETA['dblMeanZ']
        dblMeanP = dZETA['dblMeanP']
        # data on ZETA peak
        dblZETADeviation = dZETA['dblZETADeviation']
        dblLatencyZETA = dZETA['dblLatencyZETA']
        intZETAIdx = dZETA['intZETAIdx']
        # data underlying mean-rate test
        vecMu_Dur = dZETA['vecMu_Dur']
        vecMu_Base = dZETA['vecMu_Base']
        # inverse-sign ZETA
        dblD_InvSign = dZETA['dblD_InvSign']
        dblLatencyInvZETA = dZETA['dblLatencyInvZETA']
        intIdx_InvSign = dZETA['intIdx_InvSign']

        # derived from calcZetaOne
        vecRealTime = dZETA['vecRealTime']
        vecRealDeviation = dZETA['vecRealDeviation']
        vecRealFrac = dZETA['vecRealFrac']
        vecRealFracLinear = dZETA['vecRealFracLinear']
        cellRandTime = dZETA['cellRandTime']
        cellRandDeviation = dZETA['cellRandDeviation']
        dblUseMaxDur = dZETA['dblUseMaxDur']

    except:
        raise Exception(
            "plottszeta error: information is missing from dZETA dictionary")

    # %% calculate heat map
    # sampling interval
    dblSamplingInterval = np.median(np.diff(vecTime))
    vecRefT = np.arange(dblSamplingInterval/2, dblUseMaxDur, dblSamplingInterval)
    vecRefT, matAct = getInterpolatedTimeSeries(vecTime, vecData, vecEventTimes, vecRefT)

    # %% plot
    # Plot maximally 50 traces (or however man y are requested)
    intPlotRandSamples = np.min([len(cellRandTime), intPlotRandSamples])

    # Calculate optimal DPI depending on the monitor size
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 15

    # Create figure
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=dpi)

    # top left: heat map
    x0 = vecRefT[1]
    x1 = vecRefT[-1]
    xw = x1-x0
    intTrialNum = matAct.shape[0]
    yh = intTrialNum-1
    pos = ax1.imshow(matAct, interpolation='none', extent=[x0, x1, 1, intTrialNum])
    ax1.set_aspect((xw/yh)/2)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial number',
            title='Color indicates data value')
    f.colorbar(pos, ax=ax1)

    # top right: mean +/- SEM
    vecMean = np.mean(matAct, axis=0)
    vecSem = np.std(matAct, axis=0)/np.sqrt(intTrialNum)
    ax2.errorbar(vecRefT, vecMean, yerr=vecSem)
    ax2.set(xlabel='Time after event (s)', ylabel='Data value',
            title='Mean +/- SEM over trials')

    # bottom left: cumulative plots
    vecRealTime = dZETA['vecRealTime']
    vecRealFrac = dZETA['vecRealFrac']
    vecRealFracLinear = dZETA['vecRealFracLinear']
    ax3.plot(vecRealTime, vecRealFrac)
    ax3.plot(vecRealTime, vecRealFracLinear, color=[0.7, 0.7, 0.7])
    ax3.set(xlabel='Time after event (s)', ylabel='Cumulative data', title='Time-series zeta-test')

    # bottom right: deviation with random jitters
    for i in range(intPlotRandSamples-1):
        ax4.plot(cellRandTime[i], cellRandDeviation[i], color=[0.7, 0.7, 0.7])
    ax4.plot(vecRealTime, vecRealDeviation)
    ax4.plot(dblLatencyZETA, dblZETADeviation, 'bx')
    ax4.plot(dblLatencyInvZETA, dblD_InvSign, 'b*')
    ax4.set(xlabel='Time after event (s)', ylabel='Data amplitude anomaly')
    if dblMeanZ is not None:
        ax4.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f}), z(Hz)={dblMeanZ:.3f} (p={dblMeanP:.3f})')
    else:
        ax4.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f})')

    f.tight_layout()
    plt.show()
