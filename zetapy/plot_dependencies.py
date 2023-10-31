# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:08:44 2023

@author: Jorrit
"""
import numpy as np
import time
import logging
import math
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.signal import convolve, gaussian
from zetapy.ts_dependencies import getInterpolatedTimeSeries

# %% plottszeta2
def plottszeta2(dZETA, intPlotRandSamples=50):
    '''
    Creates figure for two-sample time-series ZETA-test

    Syntax:
    plottszeta2(dZETA, intPlotRandSamples=50)
    '''
    
    # unpack dZETA
    try:
        dblZetaP = dZETA['dblZetaP']
        dblZETA = dZETA['dblZETA']
        dblZETADeviation = dZETA['dblZETADeviation']
        dblZETATime = dZETA['dblZETATime']
        intZETAIdx = dZETA['intZETAIdx']
        dblMeanZ   = dZETA['dblMeanZ']
        dblMeanP  = dZETA['dblMeanP']
        vecMu1 = dZETA['vecMu1']
        vecMu2  = dZETA['vecMu2']
        dblZETADeviation_InvSign = dZETA['dblZETADeviation_InvSign']
        dblZETATime_InvSign = dZETA['dblZETATime_InvSign']
        intZETAIdx_InvSign = dZETA['intZETAIdx_InvSign']
        vecRefTime = dZETA['vecRefTime']
        vecRealDiff = dZETA['vecRealDiff']
        matRandDiff = dZETA['matRandDiff']
        vecRealFrac1 = dZETA['vecRealFrac1']
        vecRealFrac2 = dZETA['vecRealFrac2']
        dblUseMaxDur = dZETA['dblUseMaxDur']
        matTracePerTrial1 = dZETA['matTracePerTrial1']
        matTracePerTrial2 = dZETA['matTracePerTrial2']
    except:
        raise Exception(
            "plotzeta2 error: information is missing from dZETA dictionary")

    # %% plot
    # Plot maximally 50 traces (or however man y are requested)
    intPlotRandSamples = np.min([matRandDiff.shape[0], intPlotRandSamples])

    # Calculate optimal DPI depending on the monitor size
    screen_width = tk.Tk().winfo_screenwidth()
    dpi = screen_width / 15

    # Create figure
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 6), dpi=dpi)
    
    # top left: heat map 1
    x0 = vecRefTime[1]
    x1 = vecRefTime[-1]
    xw = x1-x0
    intTrialNum1 = matTracePerTrial1.shape[0]
    yh = intTrialNum1-1
    pos = ax1.imshow(matTracePerTrial1, interpolation='none', extent=[x0, x1, 1, intTrialNum1])
    ax1.set_aspect((xw/yh)/2)
    ax1.set(xlabel='Time after event (s)', ylabel='Trial number',
            title='Cond1; Color indicates data value')
    f.colorbar(pos, ax=ax1)

    # bottom left: heat map 2
    x0 = vecRefTime[1]
    x1 = vecRefTime[-1]
    xw = x1-x0
    intTrialNum2 = matTracePerTrial2.shape[0]
    yh = intTrialNum2-1
    pos = ax2.imshow(matTracePerTrial2, interpolation='none', extent=[x0, x1, 1, intTrialNum2])
    ax2.set_aspect((xw/yh)/2)
    ax2.set(xlabel='Time after event (s)', ylabel='Trial number',
            title='Cond2; Color indicates data value')
    f.colorbar(pos, ax=ax2)
    
    # top right: cumulative sums
    ax3.plot(vecRefTime, vecRealFrac1)
    ax3.plot(vecRefTime, vecRealFrac2)
    ax3.set(xlabel='Time after event (s)', ylabel='Scaled cumulative data (s)')

    # bottom right: deviation with random bootstraps
    for i in range(intPlotRandSamples-1):
        ax4.plot(vecRefTime, matRandDiff[i,:], color=[0.7, 0.7, 0.7])
    ax4.plot(vecRefTime, vecRealDiff)
    ax4.plot(dblZETATime, dblZETADeviation, 'bx')
    ax4.plot(dblZETATime_InvSign, dblZETADeviation_InvSign, 'b*')
    ax4.set(xlabel='Time after event (s)', ylabel='Difference in cumulative density')
    if dblMeanZ is not None:
        ax4.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f}), z(Hz)={dblMeanZ:.3f} (p={dblMeanP:.3f})')
    else:
        ax4.set(title=f'ZETA={dblZETA:.3f} (p={dblZetaP:.3f})')

    f.tight_layout()
    plt.show()

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
    ax2.plot(vecSpikeT, vecRealFrac1)
    ax2.plot(vecSpikeT, vecRealFrac2)
    ax2.set(xlabel='Time after event (s)', ylabel='Scaled cumulative spiking density (s)')

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