
%% set random seed
intSeed = 0;
rng(intSeed,'mt19937ar')
dblEndT = 12.9;
dblStartT = -2.9;
dblTotDur = dblEndT-dblStartT;
dblWindowDur = 1.0;
dblStimDur = 0.5;
dblSamplingRate = 25.0; %Hz
dblSampleDur = 1/dblSamplingRate;

vecSpikeTimes = dblTotDur*sort(rand(1000,1)) + dblStartT;
vecEventTimes = (0:dblWindowDur:9)';

%add stimulus-induced spikes as an elevated rate during stimulus presentation; literally the most
%advantageous situation for a t-test to pick up
vecSpikeTimesOn = dblTotDur*sort(rand(1000,1)) + dblStartT;
indKeepSpikesOn = false(size(vecSpikeTimesOn));
vecEventTimesOff = vecEventTimes + dblStimDur;
for intTrial = 1:numel(vecEventTimes)
    dblTrialStartT = vecEventTimes(intTrial);
    dblTrialStopT = vecEventTimesOff(intTrial);
    indKeepSpikesOn(vecSpikeTimesOn > dblTrialStartT & vecSpikeTimesOn < dblTrialStopT) = true;
end
vecSpikeTimesOn = vecSpikeTimesOn(indKeepSpikesOn);
vecSpikeTimes = sort([vecSpikeTimes; vecSpikeTimesOn]);

%% transform to time-series
rng(intSeed,'mt19937ar')
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
rng(intSeed,'mt19937ar')
[vecPseudoTime, vecPseudoData, vecPseudoEventT] = ...
	getPseudoTimeSeries(vecTimestamps, vecData, vecEventTimes, dblWindowDur);


%% test getTsRefT
%passes
rng(intSeed,'mt19937ar')
vecTime = getTsRefT(vecPseudoTime,vecPseudoEventT,dblWindowDur);

%% getInterpolatedTimeSeries
rng(intSeed,'mt19937ar')
[vecUsedTime,matDataPerTrial] = getInterpolatedTimeSeries(vecTimestamps,vecData,vecEventTimes,dblWindowDur,vecTime)

%% getTraceOffsetOne
rng(intSeed,'mt19937ar')
[vecThisDiff,vecThisFrac,vecThisFracLinear,vecRefT] = ...
	getTraceOffsetOne(vecTimestamps,vecData, vecEventTimes, dblWindowDur)

%% calcTsZetaOne
rng(intSeed,'mt19937ar')
intResampNum =1000;
boolDirectQuantile = false;
dblJitterSize = 2;
%[vecRefT,vecRealDiff,vecRealFrac,vecRealFracLinear,cellRandT,cellRandDiff,dblZetaP,dblZETA,intZETALoc] = ...
%		calcTsZetaOne(vecTimestamps,vecData,vecEventTimes,dblWindowDur,intResampNum,boolDirectQuantile,dblJitterSize)
	

%% zetatstest
rng(intSeed,'mt19937ar')
intPlot = 0;
[dblZetaP,sZETA] = zetatstest(vecTimestamps,vecData,vecEventTimes,dblWindowDur,intResampNum,intPlot,boolDirectQuantile,dblJitterSize)

%% zetatstest
%with ttest
rng(intSeed,'mt19937ar')
intPlot = 4;
matEventTimes = [vecEventTimes vecEventTimes+dblStimDur];
[dblZetaP2,sZETA2] = zetatstest(vecTimestamps, vecData, matEventTimes,dblWindowDur,intResampNum,intPlot,boolDirectQuantile,dblJitterSize);

