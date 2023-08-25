
%% set random seed
rng(100,'mt19937ar')
dblEndT = 12.9;
dblStartT = -2.9;
dblTotDur = dblEndT-dblStartT;
dblWindowDur = 1.0;
dblSamplingRate = 25.0; %Hz
dblSampleDur = 1/dblSamplingRate;

vecSpikeTimes = dblTotDur*sort(rand(1000,1)) + dblStartT;
vecEventTimes = (0:dblWindowDur:9)';

%transform to time-series
vecTimestamps =  dblStartT:dblSampleDur:dblEndT;
vecSpikesBinned = histcounts(vecSpikeTimes, vecTimestamps);
vecTimestamps = vecTimestamps(1:(end-1));
dblSmoothSd = 2.0;
intSmoothRange = 2*ceil(dblSmoothSd);
vecFilt = normpdf(-intSmoothRange:intSmoothRange, 0, dblSmoothSd);
vecFilt = vecFilt / sum(vecFilt);

% pad array
intPadSize = floor(numel(vecFilt)/2);
vecData = padarray(vecSpikesBinned, [0 intPadSize],'replicate');

% filter
vecData = conv(vecData, vecFilt, 'valid');

%% test getPseudoSpikeVectors
%passes, same as matlab

[vecPseudoTime, vecPseudoData, vecPseudoEventT] = ...
	getPseudoTimeSeries(vecTimestamps, vecData, vecEventTimes, dblWindowDur);


%% test getTsRefT
%passes
vecTime = getTsRefT(vecPseudoTime,vecPseudoEventT,dblWindowDur);

%% getInterpolatedTimeSeries
[vecUsedTime,matDataPerTrial] = getInterpolatedTimeSeries(vecTimestamps,vecData,vecEventTimes,dblWindowDur,vecTime)

%% getTraceOffsetOne
[vecThisDiff,vecThisFrac,vecThisFracLinear,vecRefT] = ...
	getTraceOffsetOne(vecTimestamps,vecData, vecEventTimes, dblWindowDur)

%% calcTsZetaOne
intResampNum =1000;
boolDirectQuantile = false;
dblJitterSize = 2;
%[vecRefT,vecRealDiff,vecRealFrac,vecRealFracLinear,cellRandT,cellRandDiff,dblZetaP,dblZETA,intZETALoc] = ...
%		calcTsZetaOne(vecTimestamps,vecData,vecEventTimes,dblWindowDur,intResampNum,boolDirectQuantile,dblJitterSize)
	

%% zetatstest
intPlot = 0;
[dblZetaP,sZETA] = zetatstest(vecTimestamps,vecData,vecEventTimes,dblWindowDur,intResampNum,intPlot,boolDirectQuantile,dblJitterSize)




