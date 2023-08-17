# -*- coding: utf-8 -*-
import numpy as np
import logging
from scipy import stats
from math import pi, sqrt, exp


def calcZetaOne(vecSpikeTimes, vecEventStarts, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch, boolParallel):
    """
   Calculates neuronal responsiveness index zeta
    dZETA = calcZetaOne(
        vecSpikeTimes, vecEventStarts, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch,boolParallel)
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
    # ensure orientation and assert that vecEventStarts is a 1D array of floats
    assert len(vecEventStarts.shape) < 3 and issubclass(
        vecEventStarts.dtype.type, np.floating), "Input vecEventStarts is not a 1D or 2D float np.array"
    if len(vecEventStarts.shape) > 1:
        if vecEventStarts.shape[1] < 3:
            pass
        elif vecEventStarts.shape[0] < 3:
            vecEventStarts = vecEventStarts.T
        else:
            raise Exception(
                "Input error: arrEventTimes must be T-by-1 or T-by-2; with T being the number of trials/stimuli/events")
    else:
        # turn into T-by-1 array
        vecEventStarts = np.reshape(vecEventStarts, (-1, 1))
    # define event starts
    vecEventT = vecEventStarts[:, 0]

    dblMinPreEventT = np.min(vecEventT)-dblUseMaxDur*5*dblJitterSize
    dblStartT = max([vecSpikeTimes[0], dblMinPreEventT])
    dblStopT = max(vecEventT)+dblUseMaxDur*5*dblJitterSize
    vecSpikeTimes = vecSpikeTimes[np.logical_or(vecSpikeTimes < dblStartT, vecSpikeTimes > dblStopT)]

    if vecSpikeTimes.size < 3:
        logging.warning("calcZetaOne:vecSpikeTimes: too few spikes around events to calculate zeta")
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
        logging.warning("calcZetaOne:vecRealDeviation: too few spikes around events to calculate zeta")
        return dZETA

    vecRealDeviation = vecRealDeviation - np.mean(vecRealDeviation)
    intZETALoc = np.argmax(np.abs(vecRealDeviation))
    dblMaxD = vecRealDeviation[intZETALoc]

    # %% run bootstraps; try parallel, otherwise run normal loop
    # run pre-set number of iterations
    cellRandTime = np.empty((intResampNum, 1))
    cellRandDeviation = np.empty((intResampNum, 1))
    vecMaxRandD = np.empty((intResampNum, 1))
    vecMaxRandD.fill(np.nan)

    vecStartOnly = np.reshape(vecPseudoEventT, (-1, 1))
    intTrials = vecStartOnly.size
    vecJitterPerTrial = np.multiply(dblJitterSize, np.linspace(-dblUseMaxDur, dblUseMaxDur, num=intTrials))
    matJitterPerTrial = np.empty((intTrials, intResampNum))
    matJitterPerTrial.fill(np.nan)
    for intResampling in range(intResampNum):
        matJitterPerTrial[:, intResampling] = vecJitterPerTrial[np.random.default_rng().permutation(intTrials)]

    for intResampling in range(intResampNum):
        # get random subsample
        vecStimUseOnTime = vecStartOnly + matJitterPerTrial[:, intResampling]

        # get temp offset
        vecRandDiff, vecThisSpikeFracs, vecThisFracLinear, vecThisSpikeTimes = getTempOffsetOne(
            vecPseudoSpikeTimes, vecStimUseOnTime, dblUseMaxDur)

        # assign data
        cellRandTime[intResampling] = vecThisSpikeTimes
        cellRandDeviation[intResampling] = vecRandDiff - np.mean(vecRandDiff)
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


def getZetaP(arrMaxD, vecMaxRandD, boolDirectQuantile):

    # %% calculate significance
    # find highest peak and retrieve value
    vecMaxRandD = np.sort(np.unique(vecMaxRandD), axis=0)

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
                dblValue = np.interp(d, vecMaxRandD, np.arange(0, vecMaxRandD.size)+1)

            arrZetaP[i] = 1 - (dblValue/(1+vecMaxRandD.size))

        # transform to output z-score
        arrZETA = -stats.norm.ppf(arrZetaP/2)
    else:
        # calculate statistical significance using Gumbel distribution
        dblZetaP, dblZETA = getGumbel(np.mean(vecMaxRandD), np.var(vecMaxRandD), arrMaxD)

    # return
    return arrZetaP, arrZETA


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
        New translation to Python by Jorrit Montijn: Now supports np.array input of arrX

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


def getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblUseMaxDur):
    return vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT


# 	%% get temp diff vector
# 	%pre-allocate
# 	vecSpikesInTrial = getSpikeT(vecSpikeTimes,vecStimUseOnTime,dblUseMaxDur);
# 	[vecThisSpikeTimes,ia,ic] = unique(vecSpikesInTrial);
# 	%introduce minimum jitter to identical spikes
# 	vecNotUnique = vecSpikesInTrial(ia(diff(ia)>1));
# 	if ~isempty(vecNotUnique)
# 		dblUniqueOffset = max(eps(vecSpikesInTrial));
# 		for intNotUnique=1:numel(vecNotUnique)
# 			vecIdx = find(vecNotUnique(intNotUnique)==vecSpikesInTrial);
# 			vecSpikesInTrial(vecIdx) = vecSpikesInTrial(vecIdx) + dblUniqueOffset*((1:numel(vecIdx))'-numel(vecIdx)/2);
# 		end
# 		[vecThisSpikeTimes,ia,ic] = unique(vecSpikesInTrial);
# 	end
# 	vecThisSpikeFracs = linspace(1/numel(vecThisSpikeTimes),1,numel(vecThisSpikeTimes))';
#
# 	%get linear fractions
# 	vecThisFracLinear = (vecThisSpikeTimes./dblUseMaxDur);
#
# 	%calc difference
# 	vecThisDiff = vecThisSpikeFracs - vecThisFracLinear;
# 	vecThisDiff = vecThisDiff - mean(vecThisDiff);
# end


def getPseudoSpikeVectors(vecSpikeTimes, vecEventT, dblUseMaxDur, boolDiscardEdges=False):
    #     %ensure sorting and alignment
    # 	vecSpikeTimes = sort(vecSpikeTimes(:));
    # 	vecEventT = sort(vecEventT(:));
    #
    # 	if ~exist('boolDiscardEdges','var') || isempty(boolDiscardEdges)
    # 		boolDiscardEdges = false;
    # 	end
    #
    # 	%% pre-allocate
    # 	intSamples = numel(vecSpikeTimes);
    # 	intTrials = numel(vecEventT);
    # 	dblMedianDur = median(diff(vecSpikeTimes));
    # 	cellPseudoSpikeT = cell(1,intTrials);
    # 	vecPseudoStartT = nan(intTrials,1);
    # 	dblPseudoEventT = 0;
    # 	intLastUsedSample = 0;
    # 	intFirstSample = [];
    # 	%run
    # 	for intTrial=1:intTrials
    # 		dblEventT = vecEventT(intTrial);
    # 		intStartSample = (find(vecSpikeTimes >= dblEventT,1));
    # 		intEndSample = (find(vecSpikeTimes > (dblEventT+dblWindowDur),1)-1);
    # 		if intStartSample > intEndSample
    # 			intEndSample = [];
    # 			intStartSample = [];
    # 		end
    # 		if isempty(intEndSample)
    # 			intEndSample = intStartSample;
    # 		end
    # 		vecEligibleSamples = intStartSample:intEndSample;
    # 		indRemSamples = (vecEligibleSamples <= 0) | (vecEligibleSamples > intSamples);
    # 		vecUseSamples = vecEligibleSamples(~indRemSamples);
    #
    # 		%check if beginning or end
    # 		if ~isempty(vecUseSamples)
    # 			if intTrial==1 && ~boolDiscardEdges
    # 				vecUseSamples = 1:vecUseSamples(end);
    # 			elseif intTrial==intTrials && ~boolDiscardEdges
    # 				vecUseSamples = vecUseSamples(1):intSamples;
    # 			end
    # 		end
    # 		vecAddT = vecSpikeTimes(vecUseSamples);
    # 		indOverlap = (vecUseSamples <= intLastUsedSample);
    #
    # 		%get event t
    # 		if intTrial == 1
    # 			dblPseudoEventT = 0;
    # 		else
    # 			if intTrial > 1 && dblWindowDur > (dblEventT - vecEventT(intTrial-1))
    # 				vecUseSamples = vecUseSamples(~indOverlap);
    # 				vecAddT = vecSpikeTimes(vecUseSamples);
    # 				dblPseudoEventT = dblPseudoEventT + dblEventT - vecEventT(intTrial-1);
    # 			else
    # 				dblPseudoEventT = dblPseudoEventT + dblWindowDur;
    # 			end
    # 		end
    #
    # 		%% MAKE LOCAL TO EVENT
    # 		if isempty(vecUseSamples)
    # 			vecLocalPseudoT = [];
    # 		else
    # 			intLastUsedSample = vecUseSamples(end);
    # 			vecLocalPseudoT = vecAddT - dblEventT + dblPseudoEventT;
    # 		end
    # 		if isempty(intFirstSample) && ~isempty(vecUseSamples)
    # 			intFirstSample = vecUseSamples(1);
    # 			dblPseudoT0 = dblPseudoEventT;
    # 		end
    #
    #
    # 		cellPseudoSpikeT{intTrial} = vecLocalPseudoT;
    # 		vecPseudoStartT(intTrial) = dblPseudoEventT;
    #
    # 	end
    #
    # 	%% add beginning
    # 	if ~boolDiscardEdges && ~isempty(intFirstSample) && intFirstSample > 1
    # 		dblStepBegin = vecSpikeTimes(intFirstSample) - vecSpikeTimes(intFirstSample-1);
    # 		vecSampAddBeginning = 1:(intFirstSample-1);
    # 		cellPseudoSpikeT = cat(2,{vecSpikeTimes(vecSampAddBeginning) - vecSpikeTimes(vecSampAddBeginning(1)) + dblPseudoT0 - dblStepBegin - range(vecSpikeTimes(vecSampAddBeginning))},cellPseudoSpikeT);
    # 	end
    #
    # 	%% add end
    # 	intTn = numel(vecSpikeTimes);
    # 	intLastUsedSample = find(vecSpikeTimes>(vecEventT(end)+dblWindowDur),1);
    # 	if ~boolDiscardEdges && ~isempty(intLastUsedSample) && intTn > intLastUsedSample
    # 		vecSampAddEnd = intLastUsedSample:intTn;
    # 		cellPseudoSpikeT = cat(2,cellPseudoSpikeT,{vecSpikeTimes(vecSampAddEnd) - dblEventT + dblPseudoEventT + dblWindowDur});
    # 	end
    #
    # 	%% recombine into vector
    # 	vecPseudoSpikeTimes = cell2vec(cellPseudoSpikeT);
    return vecPseudoSpikeTimes, vecPseudoEventT


def plotzeta(dZETA, dRate,
             intPlot=1):

    print("to do")

    # %% plot
    # 	if intPlot > 1
    # 		%plot maximally 50 traces
    # 		intPlotIters = min([numel(cellRandDeviation) 50]);
    #
    # 		%maximize figure
    # 		figure;
    # 		drawnow;
    # 		try
    # 			try
    # 				%try new method
    # 				h = handle(gcf);
    # 				h.WindowState = 'maximized';
    # 			catch
    # 				%try old method with javaframe (deprecated as of R2021)
    # 				sWarn = warning('off','MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame');
    # 				drawnow;
    # 				jFig = get(handle(gcf), 'JavaFrame');
    # 				jFig.setMaximized(true);
    # 				drawnow;
    # 				warning(sWarn);
    # 			end
    # 		catch
    # 		end
    # 		if intPlot > 2
    # 			subplot(2,3,1)
    # 			plotRaster(vecSpikeTimes,vecEventStarts(:,1),dblUseMaxDur,10000);
    # 			xlabel('Time after event (s)');
    # 			ylabel('Trial #');
    # 			title('Spike raster plot');
    # 			fixfig;
    # 			grid off;
    # 		end
    #
    # 		%plot
    # 		subplot(2,3,2)
    # 		sOpt = struct;
    # 		sOpt.handleFig =-1;
    # 		if dblUseMaxDur < 0.5
    # 			dblBinSize = dblUseMaxDur/40;
    # 		else
    # 			dblBinSize = 0.025;
    # 		end
    # 		vecBins = 0:dblBinSize:dblUseMaxDur;
    # 		[vecMean,vecSEM,vecWindowBinCenters] = doPEP(vecSpikeTimes,vecBins,vecEventStarts(:,1),sOpt);
    # 		errorbar(vecWindowBinCenters,vecMean,vecSEM);
    # 		ylim([0 max(get(gca,'ylim'))]);
    # 		title(sprintf('Mean spiking over trials'));
    # 		xlabel('Time after event (s)');
    # 		ylabel('Mean spiking rate (Hz)');
    # 		fixfig
    #
    # 		subplot(2,3,3)
    # 		plot(vecSpikeT,vecRealFrac)
    # 		hold on
    # 		plot(vecSpikeT,vecRealFracLinear,'color',[0.5 0.5 0.5]);
    # 		title(sprintf('Real data'));
    # 		xlabel('Time after event (s)');
    # 		ylabel('Fractional position of spike in trial');
    # 		fixfig
    #
    # 		subplot(2,3,4)
    # 		cla;
    # 		hold all
    # 		for intIter=1:intPlotIters
    # 			plot(cellRandTime{intIter},cellRandDeviation{intIter},'Color',[0.5 0.5 0.5]);
    # 		end
    # 		plot(vecSpikeT,vecRealDeviation,'Color',lines(1));
    # 		scatter(dblMaxDTime,vecRealDeviation(intZETALoc),'bx');
    # 		scatter(dblMaxDTimeInvSign,vecRealDeviation(intPeakLocInvSign),'b*');
    # 		hold off
    # 		xlabel('Time after event (s)');
    # 		ylabel('Offset of data from linear (s)');
    # 		if boolStopSupplied
    # 			title(sprintf('ZETA=%.3f (p=%.3f), z(Hz)=%.3f (p=%.3f)',dblZETA,dblZetaP,dblMeanZ,dblMeanP));
    # 		else
    # 			title(sprintf('ZETA=%.3f (p=%.3f)',dblZETA,dblZetaP));
    # 		end
    # 		fixfig
    # 	end
    # 	%% plot
    # 	if intPlot == 1
    # 		if ~isempty(get(gca,'Children'))
    # 			figure;
    # 		end
    # 		stairs(vecT,vecRate)
    # 		xlabel('Time after event (s)');
    # 		ylabel(strLabelY);
    # 		title(sprintf('Peri Event Plot (PEP)'));
    # 		fixfig
    # 	elseif intPlot > 1
    # 		subplot(2,3,5);
    # 		imagesc(matMSD');
    # 		set(gca,'ytick',[]);
    # 		ylabel(sprintf('Scale (s) (%.1es - %.1es)',vecScale(1),vecScale(end)));
    # 		xlabel('Timestamp index (#)');
    # 		title(strTitle);
    # 		fixfig
    # 		grid off
    # 		subplot(2,3,6);
    # 		if numel(vecT) > 10000
    # 			vecSubset = round(linspace(1,numel(vecT),10000));
    # 			plot(vecT(vecSubset),vecRate(vecSubset));
    # 		else
    # 			stairs(vecT,vecRate);
    # 		end
    # 		xlabel('Time after event (s)');
    # 		ylabel(strLabelY);
    # 		title(sprintf('Peri Event Plot (PEP)'));
    # 		fixfig
    # 	end
    #         if intPlot > 0
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


def getMultiScaleDeriv(vecT, vecV,
                       intSmoothSd=0, dblMinScale=None, dblBase=1.5, dblMeanRate=1, dblUseMaxDur=None, boolParallel=False):

    # 	%getMultiScaleDeriv Returns multi-scale derivative. Syntax:
    # 	%   [vecRate,sMSD] = getMultiScaleDeriv(vecT,vecV,intSmoothSd,dblMinScale,dblBase,intPlot,dblMeanRate,dblUseMaxDur,boolUseParallel)
    # 	%Required input:
    # 	%	- vecT [N x 1]: timestamps (e.g., spike times)
    # 	%	- vecV [N x 1]: values (e.g., z-scores)
    # 	%
    # 	%Optional inputs:
    # 	%	- intSmoothSd: Gaussian SD of smoothing kernel (in # of samples) [default: 0]
    # 	%	- dblMinScale: minimum derivative scale in seconds [default: 1/1000]
    # 	%	- dblBase: base for exponential scale step size [default: 1.5]
    # 	%	- intPlot: integer, plotting switch (0=none, 1=plot rates, 2=subplot 5&6 of [2 3]) [default: 0].
    # 	%						If set to 1, it will plot into the current axes if empty, or create a new figure if ~isempty(get(gca,'Children'))
    # 	%	- dblMeanRate: mean spiking rate to normalize vecRate (optional)
    # 	%	- dblUseMaxDur: trial duration to normalize vecRate (optional)
    # 	%	- boolUseParallel: use parallel processing (optional) [default: true if pool is active, otherwise false; can decrease performance, so be cautious!]
    # 	%
    # 	%Outputs:
    # 	%	- vecRate; Instantaneous spiking rate
    # 	%	- sMSD; structure with fields:
    # 	%		- vecRate; instantaneous spiking rates (like a PSTH)
    # 	%		- vecT; time-points corresponding to vecRate (same as input vecT)
    # 	%		- vecM; Mean of multi-scale derivatives
    # 	%		- vecScale; timescales used to calculate derivatives
    # 	%		- matMSD; multi-scale derivatives matrix
    # 	%		- vecV; values on which vecRate is calculated (same as input vecV)
    # 	%
    # 	%Version history:
    # 	%1.0 - January 24 2020
    # 	%	Created by Jorrit Montijn - split from previous getMultiScaleDeriv.
    # 	%1.1 - February 26 2020
    # 	%	Added instantaneous spiking rate rescaling [by JM]
    # 	%1.1.1 - January 10 2022
    # 	%	Changed plotting behavior to create new figure when intPlot==1 if gca is not empty [by JM]
    # 	%1.1.2 - May 17 2023
    # 	%	Compiled calcMSD() as mex-file & modified parfor to increase computation speed [by JM]
    # 	%1.1.3 - May 26 2023
    # 	%	Changed default parallel-processing behaviour & compiled calcSingleMSD() as mex-file.
    # 	%	GPU computation is now within try-catch block, so CPU-only pipeline works as well [by JM]
    # 	%1.1.4 - May 30 2023
    # 	%	Removed artificial points at t=0 and t=dblUseMaxDur [by JM]
    # 	%1.2 - June 6 2023
    # 	%	Fixed small temporal asymmetry of MSD calculation [by JM]
    # 	%1.2.1 - July 24 2023
    # 	%	Fixed crash if mex-file is unusable (i.e., non-windows systems) and parallel processing is
    # 	%	requested [by JM]

    # 	%% set default values
    # 	if ~exist('intSmoothSd','var') || isempty(intSmoothSd)
    # 		intSmoothSd = 0;
    # 	end
    # 	if ~exist('dblBase','var') || isempty(dblBase)
    # 		dblBase = 1.5;
    # 	end
    # 	if ~exist('dblMinScale','var') || isempty(dblMinScale)
    # 		dblMinScale = round(log(1/1000) / log(dblBase));
    # 	end
    # 	if ~exist('intPlot','var') || isempty(intPlot)
    # 		intPlot = 0;
    # 	end
    # 	if ~exist('dblMeanRate','var') || isempty(dblMeanRate)
    # 		dblMeanRate = 1;
    # 		strLabelY = 'Time-locked activation (a.u.)';
    # 	else
    # 		strLabelY = 'Spiking rate (Hz)';
    # 	end
    # 	if ~exist('dblUseMaxDur','var') || isempty(dblUseMaxDur)
    # 		dblUseMaxDur = range(vecT);
    # 	end
    # 	if ~exist('boolUseParallel','var') || isempty(boolUseParallel)
    # 		objPool = gcp('nocreate');
    # 		if isempty(objPool) || ~isprop(objPool,'NumWorkers') || objPool.NumWorkers < 4
    # 			boolUseParallel = false;
    # 		else
    # 			boolUseParallel = true;
    # 		end
    # 	end

    # 	%% reorder just in case
    # 	[vecT,vecReorder] = sort(vecT(:),'ascend');
    # 	vecV = vecV(vecReorder);
    # 	vecV = vecV(:);
    # 	indRem = vecT==0 | vecT==dblUseMaxDur;%points at 0 and 1 are artificial
    # 	vecT(indRem) = [];
    # 	vecV(indRem) = [];

    # 	%% get multi-scale derivative
    # 	dblMaxScale = log(max(vecT)/10) / log(dblBase);
    # 	vecExp = dblMinScale:dblMaxScale;
    # 	vecScale=dblBase.^vecExp;
    # 	intScaleNum = numel(vecScale);
    # 	intN = numel(vecT);
    # 	matMSD = zeros(intN,intScaleNum);
    # 	if boolUseParallel
    # 		try
    # 			parfor intScaleIdx=1:intScaleNum
    # 				dblScale = vecScale(intScaleIdx);
    # 				%run through all points
    # 				matMSD(:,intScaleIdx) = calcSingleMSD_mex(dblScale,vecT,vecV);
    # 			end
    # 		catch
    # 			parfor intScaleIdx=1:intScaleNum
    # 				dblScale = vecScale(intScaleIdx);
    # 				%run through all points
    # 				matMSD(:,intScaleIdx) = calcSingleMSD(dblScale,vecT,vecV);
    # 			end
    # 		end
    # 	else
    # 		try
    # 			for intScaleIdx=1:intScaleNum
    # 				dblScale = vecScale(intScaleIdx);
    # 				%run through all points
    # 				matMSD(:,intScaleIdx) = calcSingleMSD_mex(dblScale,vecT,vecV);
    # 			end
    # 		catch
    # 			for intScaleIdx=1:intScaleNum
    # 				dblScale = vecScale(intScaleIdx);
    # 				%run through all points
    # 				matMSD(:,intScaleIdx) = calcSingleMSD(dblScale,vecT,vecV);
    # 			end
    # 		end
    # 	end

    # 	%% smoothing
    # 	if intSmoothSd > 0
    # 		vecFilt = normpdf(-2*(intSmoothSd):2*intSmoothSd,0,intSmoothSd)';
    # 		vecFilt = vecFilt./sum(vecFilt);
    # 		%pad array
    # 		matMSD = padarray(matMSD,floor(size(vecFilt)/2),'replicate');

    # 		%filter
    # 		try
    # 			matMSD = conv2(gpuArray(matMSD),gpuArray(vecFilt),'valid');
    # 		catch
    # 			matMSD = conv2(matMSD,vecFilt,'valid');
    # 		end

    # 		%title
    # 		strTitle = 'Smoothed MSDs';
    # 	else
    # 		%title
    # 		strTitle = 'MSDs';
    # 	end
    # 	%mean
    # 	vecM = mean(gather(matMSD),2);

    # 	%weighted average of vecM by inter-spike intervals
    # 	dblMeanM = (1/dblUseMaxDur)*sum(((vecM(1:(end-1)) + vecM(2:end))/2).*diff(vecT));

    # 	%rescale to real firing rates
    # 	vecRate = dblMeanRate * ((vecM + 1/dblUseMaxDur)/(dblMeanM + 1/dblUseMaxDur));

    # 	%% build output
    # 	if nargout > 1
    # 		sMSD = struct;
    # 		sMSD.vecRate = vecRate;
    # 		sMSD.vecT = vecT;
    # 		sMSD.vecM = vecM;
    # 		sMSD.vecScale = vecScale;
    # 		sMSD.matMSD = matMSD;
    # 		sMSD.vecV = vecV;
    # 	end
    vecRate = np.array()
    dMSD = dict()
    return vecRate, dMSD


def calcSingleMSD(dblScale, vecT, vecV):
    # 	intN = numel(vecT);
    # 	vecMSD = zeros(intN,1);
    # 	%run through all points
    # 	for intS=1:intN
    # 		%select points within window
    # 		dblT = vecT(intS);
    # 		dblMinEdge = dblT - dblScale/2;
    # 		dblMaxEdge = dblT + dblScale/2;
    # 		indCompMin = vecT > dblMinEdge;
    # 		intIdxMinT = find(indCompMin,1,'first');
    # 		if isempty(intIdxMinT)
    # 			intIdxMinT=1;
    # 		else
    # 			intIdxMinT = intIdxMinT(1);
    # 		end
    # 		indCompMax = vecT > dblMaxEdge;
    # 		intIdxMaxT = find(indCompMax,1,'first');
    # 		if isempty(intIdxMaxT)
    # 			intIdxMaxT=intN;
    # 		else
    # 			intIdxMaxT = intIdxMaxT(1)-1;
    # 		end
    # 		if (intIdxMinT > intIdxMaxT)
    # 			dblD=0;
    # 		else
    # 			if (intIdxMinT == intIdxMaxT) && (intIdxMinT > 1) && (intIdxMinT < intN)
    # 				intIdxMaxT=intIdxMinT+1;
    # 				intIdxMinT=intIdxMinT-1;
    # 			end
    # 			dbl_dT = max([dblScale (vecT(intIdxMaxT) - vecT(intIdxMinT))]);
    # 			dblD = (vecV(intIdxMaxT) - vecV(intIdxMinT))/dbl_dT;
    # 		end
    # 		%select points within window
    # 		vecMSD(intS) = dblD;
    # 	end
    vecMSD = np.array()
    return vecMSD


def getPeak(vecData, vecT, tplRestrictRange=(-np.inf, np.inf), intSwitchZ=1):
    """Returns highest peak time, width, and location. Syntax:
        [dblPeakValue,dblPeakTime,dblPeakWidth,vecPeakStartStop,intPeakLoc,vecPeakStartStopIdx] = getPeak(vecData,vecT,tplRestrictRange=(-np.inf,np.inf), intSwitchZ=1):

    Required input:
        - vecData [N x 1]: values

    Optional inputs:
        - vecT [N x 1]: timestamps corresponding to vecData (default: [1:N])
        - tplRestrictRange: restrict peak to lie within tplRestrictRange[0] and tplRestrictRange[1]

    Outputs:
        - dblPeakTime: time of peak
        - dblPeakWidth: width of peak
        - vecPeakStartStop: start/stop times of peak
        - intPeakLoc: index of peak
        - vecPeakStartStopIdx: start/stop indices of peak

    Version history:
    1.0 - June 19, 2020, Created by Jorrit Montijn, Translated to python by Alexander Heimel
    """

    dPeak = dict()
    dPeak['dblPeakRate'] = None
    dPeak['dblPeakTime'] = None
    dPeak['dblPeakWidth'] = None
    dPeak['vecPeakStartStop'] = None
    dPeak['intPeakLoc'] = None
    dPeak['vecPeakStartStopIdx'] = None
    return dPeak


def getOnset(vecData, vecT, dblPeakT, vecRestrictRange, intSwitchZ=1):
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

    return dblOnset, dblValue, dblBaseVal, dblPeakT, dblPeakVal
