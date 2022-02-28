import numpy as np
import time
import logging
import math
from zetapy import msd
from scipy import stats
from zetapy.dependencies import flatten, getTempOffset, getGumbel, getPeak, getOnset


def getZeta(arrSpikeTimes, arrEventTimes, dblUseMaxDur=None, intResampNum=100, intPlot=0,
            intLatencyPeaks=2, tplRestrictRange=(-np.inf,np.inf),
            boolReturnRate=False, boolReturnZETA=False, boolVerbose=False):
    """
    Calculates neuronal responsiveness index ZETA.

    Montijn, J.S., Seignette, K., Howlett, M.H., Cazemier, J.L., Kamermans, M., Levelt, C.N.,
    and Heimel, J.A. (2021). A parameter-free statistical test for neuronal responsiveness.
    eLife 10, e71969.

    Parameters
    ----------
    arrSpikeTimes : 1D array
        spike times (in seconds).
    arrEventTimes : 1D or 2D array
        event on times (s), or [T x 2] including event off times to calculate mean-rate difference.
    dblUseMaxDur : float
        window length for calculating ZETA: ignore all spikes beyond this duration after event onset
        (default: median of event onset to event onset)
    intResampNum : integer
        number of resamplings (default: 100)
    intPlot : integer
        plotting switch (0=none, 1=inst. rate only, 2=traces only, 3=raster plot as well,
                         4=adds latencies in raster plot) (default: 0)
    intLatencyPeaks : integer
        maximum number of latency peaks to return (1-4) (default: 2)
    tplRestrictRange : 2 element tuple
        temporal range within which to restrict onset/peak latencies (default: [-inf inf])
    boolReturnRate : boolean
        switch to return dictionary with spiking rate features
    boolReturnZETA : boolean
        switch to return dictionary with additional ZETA parameters
    boolVerbose : boolean
        switch to print progress messages (default: false)

    Returns
    -------
    dblZetaP : float
        p-value based on Zenith of Event-based Time-locked Anomalies
    arrLatencies : 1D array
        different latency estimates, number determined by intLatencyPeaks.
        If no peaks are detected, it returns NaNs
            1) Latency of ZETA
            2) Latency of largest z-score with inverse sign to ZETA
            3) Peak time of instantaneous firing rate
            4) Onset time of above peak, defined as the first crossing of peak half-height
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
            Additionally, it will return peak onset latency (first crossing of peak half-height)
            dblOnset: latency for peak onset

    Original code by Jorrit Montijn, ported to python by Alexander Heimel & Guido Meijer

    Version history:
    2.5 - 17 June 2020 Jorrit Montijn, translated to python by Alexander Heimel
    2.5.1 - 18 February 2022 Bugfix by Guido Meijer of 1D arrEventTimes
    2.6 - 20 February 2022 Refactoring of python code by Guido Meijer
    """

    # ensure arrEventTimes is a N x 2 array
    if len(arrEventTimes.shape) > 1:
        boolStopSupplied = True
        if np.shape(arrEventTimes)[1] > 2:
            arrEventTimes = np.transpose(arrEventTimes)
    else:
        boolStopSupplied = False
        arrEventTimes = np.vstack((arrEventTimes, np.zeros(arrEventTimes.shape))).T

    # trial dur
    if dblUseMaxDur is None:
        dblUseMaxDur = np.median(np.diff(arrEventTimes[:,0]))

    ## build onset/offset vectors
    arrEventStarts = arrEventTimes[:,0]

    ## prepare interpolation points
    intMaxRep = np.shape(arrEventTimes)[0]
    cellSpikeTimesPerTrial = [None] * intMaxRep

    # go through trials to build spike time vector
    for intEvent in range(intMaxRep):
        # get times
        dblStartT = arrEventStarts[intEvent]
        dblStopT = dblStartT + dblUseMaxDur

        # build trial assignment
        cellSpikeTimesPerTrial[intEvent] = arrSpikeTimes[(arrSpikeTimes < dblStopT)
                                                         & (arrSpikeTimes > dblStartT)] - dblStartT

    # get spikes in fold
    vecSpikeT = np.array(sorted(flatten([0,cellSpikeTimesPerTrial, dblUseMaxDur])))
    intSpikes = vecSpikeT.shape[0]

    ## run normal
    vecOrigDiff, vecRealFrac, vecRealFracLinear = getTempOffset(vecSpikeT, arrSpikeTimes,
                                                                arrEventStarts, dblUseMaxDur)

    # mean subtract difference
    vecRealDiff = vecOrigDiff - np.mean(vecOrigDiff)

    ## run bootstraps
    hTic = time.time()
    matRandDiff = np.empty((intSpikes, intResampNum))
    matRandDiff[:] = np.nan
    for intResampling in range(intResampNum):
        ## msg
        if boolVerbose and ((time.time()-hTic) > 5):
            print('Now at resampling %d/%d' % (intResampling,intResampNum))
            hTic = time.time()

        ## get random subsample
        vecStimUseOnTime = (arrEventStarts + 2 * dblUseMaxDur
                            * ((np.random.rand(arrEventStarts.shape[0]) - 0.5) * 2))

        # get temp offset
        vecRandDiff, vecRandFrac, vecRandFracLinear = getTempOffset(vecSpikeT, arrSpikeTimes,
                                                                    vecStimUseOnTime, dblUseMaxDur)

        # assign data
        matRandDiff[:,intResampling] = vecRandDiff - np.mean(vecRandDiff)

    ## calculate measure of effect size (for equal n, d' equals Cohen's d)
    if (len(vecRealDiff) < 3) | (arrSpikeTimes.shape[0] < 10):
        if boolVerbose:
            logging.warning('Insufficient samples to calculate zeta')
        dblZetaP = 1
        arrLatencies = np.array([np.nan] * intLatencyPeaks)
        dZETA = dict()
        dRate = dict()
        if (boolReturnZETA & boolReturnRate):
            return dblZetaP, arrLatencies, dZETA, dRate
        elif boolReturnZETA:
            return dblZetaP, arrLatencies, dZETA
        elif boolReturnRate:
            return dblZetaP, arrLatencies, dRate
        else:
            return dblZetaP, arrLatencies

    # find highest peak and retrieve value
    vecMaxRandD = np.max(np.abs(matRandDiff), 0)
    dblRandMu = np.mean(vecMaxRandD)
    dblRandVar = np.var(vecMaxRandD, ddof=1)
    intZETALoc = np.argmax(np.abs(vecRealDiff))
    dblPosD = np.max(np.abs(vecRealDiff)) # Can be combined with line above

    # get location
    dblMaxDTime = vecSpikeT[intZETALoc]
    dblD = vecRealDiff[intZETALoc]

    # calculate statistical significance using Gumbel distribution
    if boolVerbose:
        print('Python: Gumbel %0.7f, %0.7f, %0.7f' % (dblRandMu, dblRandVar, dblPosD))
    dblZetaP, dblZETA = getGumbel(dblRandMu, dblRandVar, dblPosD)

    # find peak of inverse sign
    intPeakLocInvSign = np.argmax(-np.sign(dblD)*vecRealDiff)
    dblMaxDTimeInvSign = vecSpikeT[intPeakLocInvSign]
    dblD_InvSign = vecRealDiff[intPeakLocInvSign]

    if boolStopSupplied:
        ## calculate mean-rate difference
        vecEventStops = arrEventTimes[:,1]
        vecStimHz = np.zeros(intMaxRep)
        vecBaseHz = np.zeros(intMaxRep)
        dblMedianBaseDur = np.median(arrEventStarts[1:] - vecEventStops[0:-1])

        # go through trials to build spike time vector
        for intEvent in range(intMaxRep):
            # get times
            dblStartT = arrEventStarts[intEvent]
            dblStopT = dblStartT + dblUseMaxDur
            dblPreT = dblStartT - dblMedianBaseDur

            # build trial assignment
            vecStimHz[intEvent] = (np.sum((arrSpikeTimes < dblStopT) & (arrSpikeTimes > dblStartT))
                                   / (dblStopT - dblStartT))
            vecBaseHz[intEvent] = (np.sum((arrSpikeTimes < dblStartT) & (arrSpikeTimes > dblPreT))
                                   / dblMedianBaseDur)

        # get metrics
        dblMeanD = np.mean(vecStimHz - vecBaseHz) / ((np.std(vecStimHz) + np.std(vecBaseHz)) / 2)
        dblMeanP = stats.ttest_rel(vecStimHz, vecBaseHz)

    ## plot
    if intPlot > 1:
        logging.warning('Plotting is not translated to python yet')
        """
        %plot maximally 50 traces
        intPlotIters = min([size(matRandDiff,2) 50]);

        %make maximized figure
        figure
        drawnow;
        jFig = get(handle(gcf), 'JavaFrame');
        jFig.setMaximized(true);
        figure(gcf);
        drawnow;

        if intPlot > 2
            subplot(2,3,1)
            plotRaster(arrSpikeTimes,arrEventStarts(:,1),dblUseMaxDur,10000);
            xlabel('Time from event (s)');
            ylabel('Trial #');
            title('Spike raster plot');
            fixfig;
            grid off;
        end

        %plot
        subplot(2,3,2)
        sOpt = struct;
        sOpt.handleFig =-1;
        [vecMean,vecSEM,vecWindowBinCenters] = doPEP(arrSpikeTimes,0:0.025:dblUseMaxDur,arrEventStarts(:,1),sOpt);
        errorbar(vecWindowBinCenters,vecMean,vecSEM);
        ylim([0 max(get(gca,'ylim'))]);
        title(sprintf('Mean spiking over trials'));
        xlabel('Time from event (s)');
        ylabel('Mean spiking rate (Hz)');
        fixfig

        subplot(2,3,3)
        plot(vecSpikeT,vecRealFrac)
        hold on
        plot(vecSpikeT,vecRealFracLinear,'color',[0.5 0.5 0.5]);
        title(sprintf('Real data'));
        xlabel('Time from event (s)');
        ylabel('Fractional position of spike in trial');
        fixfig

        subplot(2,3,4)
        cla;
        hold all
        for intOffset=1:intPlotIters
            plot(vecSpikeT,matRandDiff(:,intOffset),'Color',[0.5 0.5 0.5]);
        end
        plot(vecSpikeT,vecRealDiff,'Color',lines(1));
        scatter(dblMaxDTime,vecRealDiff(intZETALoc),'bx');
        scatter(dblMaxDTimeInvSign,vecRealDiff(intPeakLocInvSign),'b*');
        hold off
        xlabel('Time from event (s)');
        ylabel('Offset of data from linear (s)');
        if boolStopSupplied
            title(sprintf('ZETA=%.3f (p=%.3f), d(Hz)=%.3f (p=%.3f)',dblZETA,dblZetaP,dblMeanD,dblMeanP));
        else
            title(sprintf('ZETA=%.3f (p=%.3f)',dblZETA,dblZetaP));
        end
        fixfig
    """

    ## calculate MSD if significant
    if intLatencyPeaks > 0:
        # get average of multi-scale derivatives, and rescaled to instantaneous spiking rate
        dblMeanRate =  intSpikes / (dblUseMaxDur * intMaxRep)
        vecRate, dRate = msd.getMultiScaleDeriv(vecSpikeT, vecRealDiff, intPlot=intPlot,
                                                dblMeanRate=dblMeanRate, dblUseMaxDur=dblUseMaxDur)
    else:
        dRate = None

    ## calculate MSD statistics
    if dRate is not None and intLatencyPeaks > 0:
        # get MSD peak
        (dblPeakRate, dblPeakTime, dblPeakWidth, vecPeakStartStop,
         intPeakLoc, vecPeakStartStopIdx) = getPeak(vecRate, vecSpikeT, tplRestrictRange)

        dRate['dblPeakRate'] = dblPeakRate
        dRate['dblPeakTime'] = dblPeakTime
        dRate['dblPeakWidth'] = dblPeakWidth
        dRate['vecPeakStartStop'] = vecPeakStartStop
        dRate['intPeakLoc'] = intPeakLoc
        dRate['vecPeakStartStopIdx'] = vecPeakStartStopIdx

        if not math.isnan(dblPeakTime):
            # assign array data
            if intLatencyPeaks > 3:
                # get onset
                dblOnset, dblOnsetVal = getOnset(vecRate, vecSpikeT, dblPeakTime, tplRestrictRange)[:2]
                dRate['dblOnset'] = dblOnset
                arrLatencies = np.array([dblMaxDTime, dblMaxDTimeInvSign, dblPeakTime, dblOnset])
                vecLatencyVals = np.array([vecRate[intZETALoc], vecRate[intPeakLocInvSign],
                                           vecRate[intPeakLoc], dblOnsetVal], dtype=object)
            else:
                dRate['dblOnset'] = np.nan
                arrLatencies = np.array([dblMaxDTime, dblMaxDTimeInvSign, dblPeakTime])
                vecLatencyVals = np.array([vecRate[intZETALoc], vecRate[intPeakLocInvSign],
                                           vecRate[intPeakLoc]], dtype=object)
            arrLatencies = arrLatencies[0:intLatencyPeaks]
            vecLatencyVals = vecLatencyVals[0:intLatencyPeaks]
            if intPlot > 0:
                logging.warning('Plot not translated to python yet')
                """
                hold on
                scatter(dblPeakTime,vecRate(intPeakLoc),'gx');
                scatter(dblMaxDTime,vecRate(intZETALoc),'bx');
                scatter(dblMaxDTimeInvSign,vecRate(intPeakLocInvSign),'b*');
                if intLatencyPeaks > 3
                    scatter(dblOnset,dblOnsetVal,'rx');
                    title(sprintf('ZETA=%.0fms,-ZETA=%.0fms,Pk=%.0fms,On=%.2fms',dblMaxDTime*1000,dblMaxDTimeInvSign*1000,dblPeakTime*1000,dblOnset*1000));
                else
                    title(sprintf('ZETA=%.0fms,-ZETA=%.0fms,Pk=%.0fms',dblMaxDTime*1000,dblMaxDTimeInvSign*1000,dblPeakTime*1000));
                end
                hold off
                fixfig;

                if intPlot > 3
                    vecHandles = get(gcf,'children');
                    ptrFirstSubplot = vecHandles(find(contains(get(vecHandles,'type'),'axes'),1,'last'));
                    axes(ptrFirstSubplot);
                    vecY = get(gca,'ylim');
                    hold on;
                    if intLatencyPeaks > 3,plot(dblOnset*[1 1],vecY,'r--');end
                    plot(dblPeakTime*[1 1],vecY,'g--');
                    plot(dblMaxDTime*[1 1],vecY,'b--');
                    plot(dblMaxDTimeInvSign*[1 1],vecY,'b-.');
                    hold off
                end
                """
        else:
            #placeholder peak data
            dRate['dblOnset'] = np.nan
            arrLatencies = np.array([np.nan] * intLatencyPeaks)
            vecLatencyVals = np.array([np.nan] * intLatencyPeaks)
    else:
        arrLatencies = []
        vecLatencyVals = []

    ## build optional output dictionary
    dZETA = dict()
    dZETA['dblZeta'] = dblZETA
    dZETA['dblD'] = dblD
    dZETA['dblP'] = dblZetaP
    dZETA['dblPeakT'] = dblMaxDTime
    dZETA['intPeakIdx'] = intZETALoc
    if boolStopSupplied:
        dZETA['dblMeanD'] = dblMeanD
        dZETA['dblMeanP'] = dblMeanP
    dZETA['vecSpikeT'] = vecSpikeT
    dZETA['vecRealFrac'] = vecRealFrac
    dZETA['vecRealFracLinear'] = vecRealFracLinear
    dZETA['vecD'] = vecRealDiff
    dZETA['vecNoNormD'] = vecOrigDiff
    dZETA['matRandD'] = matRandDiff
    dZETA['dblD_InvSign'] = dblD_InvSign
    dZETA['dblPeakT_InvSign'] = dblMaxDTimeInvSign
    dZETA['intPeakIdx_InvSign'] = intPeakLocInvSign
    dZETA['dblUseMaxDur'] = dblUseMaxDur

    if (boolReturnZETA & boolReturnRate):
        return dblZetaP, arrLatencies, dZETA, dRate
    elif boolReturnZETA:
        return dblZetaP, arrLatencies, dZETA
    elif boolReturnRate:
        return dblZetaP, arrLatencies, dRate
    else:
        return dblZetaP, arrLatencies


def getIFR(arrSpikeTimes, arrEventStarts, dblUseMaxDur=None, intSmoothSd=5, dblMinScale=None,
           dblBase=1.5, intPlot=0, boolVerbose=True):
    """Returns multi-scale derivative. Syntax:
       [vecMSD,sMSSD] = getMultiScaleSpikeDeriv(arrSpikeTimes,arrEventStarts,dblUseMaxDur,intSmoothSd,dblMinScale,dblBase,intPlot,boolVerbose)
    Required input:
        - arrSpikeTimes [S x 1]: spike times (s)
        - arrEventStarts [T x 1]: event on times (s), or [T x 2] including event off times
        - dblUseMaxDur: float (s), ignore all spikes beyond this duration after stimulus onset
                                    [default: median of trial start to trial start]

    Optional inputs:
        - intSmoothSd: Gaussian SD of smoothing kernel (in # of bins) [default: 3]
        - dblMinScale: minimum derivative scale in seconds [default: 1/1000]
        - dblBase: critical value for locally dynamic derivative [default: 4]
        - intPlot: integer, plotting switch (0=none, 1=plot)
        - boolVerbose: boolean, switch to print messages

    Outputs:
        - vecMSprime; Multi-scale derivative
        - sMSSD; structure with fields:
            - vecMSD;
            - vecSpikeT;
            - vecFracs;
            - vecLinear;
            - vecDiff;
            - vecScale;
            - matSmoothMSprime;
            - matMSprime;

    Version history:
    1.0 - June 24, 2020 Created by Jorrit Montijn, translated to python by Alexander Heimel
    """

    if dblMinScale==None:
        dblMinScale = np.round(np.log(1/1000) / np.log(dblBase))

    if dblUseMaxDur == None:
        dblUseMaxDur = np.median(np.diff(arrEventStarts[:,0]))

    ## prepare normalized spike times
    # pre-allocate
    intMaxRep = np.shape(arrEventStarts)[0]
    cellSpikeTimesPerTrial = [None] * intMaxRep

    # go through trials to build spike time vector
    for intEvent in range(intMaxRep):
        # get times
        dblStartT = arrEventStarts[intEvent,0]
        dblStopT = dblStartT + dblUseMaxDur

        # build trial assignment
        cellSpikeTimesPerTrial[intEvent] = arrSpikeTimes[(arrSpikeTimes < dblStopT)
                                                         & (arrSpikeTimes > dblStartT)] - dblStartT

    # get spikes in fold
    vecSpikeT = np.array(sorted(flatten(cellSpikeTimesPerTrial)))

    ## get difference from uniform
    vecFracs = np.linspace(0, 1, vecSpikeT.shape[0])
    vecLinear = vecSpikeT / np.max(vecSpikeT)
    vecDiff = vecFracs - vecLinear
    vecDiff = vecDiff - np.mean(vecDiff)

    ## get multi-scale derivative
    vecMSD, sMSD = msd.getMultiScaleDeriv(vecSpikeT, vecDiff, intSmoothSd, dblMinScale, dblBase, intPlot)

    sMSD['vecSpikeT'] = vecSpikeT
    sMSD['vecFracs'] = vecFracs
    sMSD['vecLinear'] = vecLinear
    sMSD['vecDiff'] = vecDiff

    return vecMSD, sMSD

