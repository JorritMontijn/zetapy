# -*- coding: utf-8 -*-


def calcZetaOne(arrSpikeTimes, arrEventStarts, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch, boolUseParallel):
    """
   Calculates neuronal responsiveness index zeta
    arrSpikeT, arrRealDiff, arrRealFrac, arrRealFracLinear, cellRandT, cellRandDiff, dblZetaP, dblZETA, intZETALoc = calcZetaOne(
        arrSpikeTimes, arrEventStarts, dblUseMaxDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch,boolUseParallel)

    """

    # %% pre-allocate output
    arrSpikeT = None
    arrRealDiff = None
    arrRealFrac = None
    arrRealFracLinear = None
    cellRandT = None
    cellRandDiff = None
    dblZetaP = 1.0
    dblZETA = 0.0
    intZETALoc = None

#     %% check inputs and pre-allocate error output

# 	if ~exist('boolDirectQuantile','var') || isempty(boolDirectQuantile)
# 		boolDirectQuantile = false;
# 	end
# 	if ~exist('boolStitch','var') || isempty(boolStitch)
# 		boolStitch = true;
# 	end
# 	if ~exist('boolUseParallel','var') || isempty(boolUseParallel)
# 		objPool = gcp('nocreate');
# 		if isempty(objPool) || ~isprop(objPool,'NumWorkers') || objPool.NumWorkers < 4
# 			boolUseParallel = false;
# 		else
# 			boolUseParallel = true;
# 		end
# 	end
#
# 	%% reduce spikes
# 	if size(vecEventStarts,2)>2,error([mfilename ':IncorrectMatrixForm'],'Incorrect input form for vecEventStarts; size must be [m x 1] or [m x 2]');end
# 	vecEventT = vecEventStarts(:,1);
# 	dblStartT = max([vecSpikeTimes(1) min(vecEventT)-dblUseMaxDur*5*dblJitterSize]);
# 	dblStopT = max(vecEventT)+dblUseMaxDur*5*dblJitterSize;
# 	vecSpikeTimes(vecSpikeTimes < dblStartT | vecSpikeTimes > dblStopT) = [];
# 	if numel(vecSpikeTimes) < 3
# 		return;
# 	end
#
# 	%% build pseudo data, stitching stimulus periods
# 	if boolStitch
# 		[vecPseudoSpikeTimes,vecPseudoEventT] = getPseudoSpikeVectors(vecSpikeTimes,vecEventT,dblUseMaxDur);
# 	else
# 		vecPseudoSpikeTimes = vecSpikeTimes;
# 		vecPseudoEventT = vecEventT;
# 	end
#
# 	%% run normal
# 	%get data
# 	[vecRealDiff,vecRealFrac,vecRealFracLinear,vecSpikeT] = ...
# 		getTempOffsetOne(vecPseudoSpikeTimes,vecPseudoEventT,dblUseMaxDur);
# 	if numel(vecRealDiff) < 3
# 		return
# 	end
# 	vecRealDiff = vecRealDiff - mean(vecRealDiff);
# 	[dblMaxD,intZETALoc]= max(abs(vecRealDiff));
#
# 	%% run bootstraps; try parallel, otherwise run normal loop
# 	% run pre-set number of iterations
# 	cellRandT = cell(1,intResampNum);
# 	cellRandDiff = cell(1,intResampNum);
# 	vecMaxRandD = nan(1,intResampNum);
# 	vecStartOnly = vecPseudoEventT(:);
# 	intTrials = numel(vecStartOnly);
# 	%vecJitterPerTrial = dblJitterSize*dblUseMaxDur*((rand(size(vecStartOnly))-0.5)*2); %original zeta
# 	vecJitterPerTrial = dblJitterSize*linspace(-dblUseMaxDur,dblUseMaxDur,intTrials)'; %new
# 	matJitterPerTrial = nan(intTrials,intResampNum);
# 	for intResampling=1:intResampNum
# 		matJitterPerTrial(:,intResampling) = vecJitterPerTrial(randperm(numel(vecJitterPerTrial)));
# 	end
# 	if boolUseParallel
# 		parfor intResampling=1:intResampNum
# 			%% get random subsample
# 			vecStimUseOnTime = vecStartOnly + matJitterPerTrial(:,intResampling);
#
# 			%get temp offset
# 			[vecRandDiff,vecThisSpikeFracs,vecThisFracLinear,vecThisSpikeTimes] = ...
# 				getTempOffsetOne(vecPseudoSpikeTimes,vecStimUseOnTime,dblUseMaxDur);
#
# 			%assign data
# 			cellRandT{intResampling} = vecThisSpikeTimes;
# 			cellRandDiff{intResampling} = vecRandDiff - mean(vecRandDiff);
# 			vecMaxRandD(intResampling) = max(abs(cellRandDiff{intResampling}));
# 		end
# 	else
# 		for intResampling=1:intResampNum
# 			%% get random subsample
# 			vecStimUseOnTime = vecStartOnly + matJitterPerTrial(:,intResampling);
#
# 			%get temp offset
# 			[vecRandDiff,vecThisSpikeFracs,vecThisFracLinear,vecThisSpikeTimes] = ...
# 				getTempOffsetOne(vecPseudoSpikeTimes,vecStimUseOnTime,dblUseMaxDur);
#
# 			%assign data
# 			cellRandT{intResampling} = vecThisSpikeTimes;
# 			cellRandDiff{intResampling} = vecRandDiff - mean(vecRandDiff);
# 			vecMaxRandD(intResampling) = max(abs(cellRandDiff{intResampling}));
# 		end
# 	end
#
# 	%% calculate significance
# 	[dblZetaP,dblZETA] = getZetaP(dblMaxD,vecMaxRandD,boolDirectQuantile);

    return arrSpikeT, arrRealDiff, arrRealFrac, arrRealFracLinear, cellRandT, cellRandDiff, dblZetaP, dblZETA, intZETALoc

def plotzeta(dZETA,dRate,
             intPlot=1):
    
    
    print("to do")
    
    # %% plot
    # 	if intPlot > 1
    # 		%plot maximally 50 traces
    # 		intPlotIters = min([numel(cellRandDiff) 50]);
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
    # 			plot(cellRandT{intIter},cellRandDiff{intIter},'Color',[0.5 0.5 0.5]);
    # 		end
    # 		plot(vecSpikeT,vecRealDiff,'Color',lines(1));
    # 		scatter(dblMaxDTime,vecRealDiff(intZETALoc),'bx');
    # 		scatter(dblMaxDTimeInvSign,vecRealDiff(intPeakLocInvSign),'b*');
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
