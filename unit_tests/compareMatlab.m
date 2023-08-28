
%% set random seed
rng(1,'mt19937ar')
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

%% test getPseudoSpikeVectors
%passes; same as python
rng(1,'mt19937ar')
dblWindowDur = 1.0;
boolDiscardEdges = false;

[vecPseudoSpikeTimes,vecPseudoEventT] = getPseudoSpikeVectors(vecSpikeTimes,vecEventTimes,dblWindowDur,boolDiscardEdges);

%% getspiket
%passes
rng(1,'mt19937ar')
[vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT] = getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblWindowDur); %#ok<ASGLU> 

%% calculate zetaone
%passes
rng(1,'mt19937ar')
intResampNum = 100;
boolDirectQuantile = false;
dblJitterSize = 2;
boolStitch = true;
[vecSpikeT,vecRealDiff,vecRealFrac,vecRealFracLinear,cellRandT,cellRandDiff,dblZetaP,dblZETA,intZETALoc] = calcZetaOne(vecSpikeTimes, vecEventTimes, dblWindowDur, intResampNum, boolDirectQuantile, dblJitterSize, boolStitch);
plot(vecSpikeT,vecRealDiff)

[dblMaxD1,intZETAIdx] = max(abs(vecRealDiff))
dblMaxD = abs(vecRealDiff(intZETAIdx))

%% test zetatest
%passes
rng(1,'mt19937ar')
[p,sZETA] = zetatest(vecSpikeTimes,vecEventTimes);

%% get deviation vector
%passes
rng(1,'mt19937ar')
dblWindowDur = 1.0;
boolDiscardEdges = false;
[vecPseudoSpikeTimes,vecPseudoEventT] = getPseudoSpikeVectors(vecSpikeTimes,vecEventTimes,dblWindowDur,boolDiscardEdges)
[vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT] = getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblWindowDur)


%% test getmsd
%passes
rng(1,'mt19937ar')
[vecRate,sMSD] = getMultiScaleDeriv(vecSpikeT,vecRealDeviation);

%with smoothing, also passes
dblSmoothSd = 1;
dblMinScale = -20;
dblBase = 1.3;
intPlot = 2;
dblMeanRate = 1;
dblUseMaxDur = 1;
[vecRate2,sMSD2] = getMultiScaleDeriv(vecSpikeT,vecRealDeviation,dblSmoothSd,dblMinScale,dblBase,intPlot,dblMeanRate,dblUseMaxDur);

%% test getpeak
%passes
rng(1,'mt19937ar')
[dblPeakValue,dblPeakTime,dblPeakWidth,vecPeakStartStop,intPeakLoc,vecPeakStartStopIdx] = getPeak(vecRate2, sMSD2.vecT);
dblPeakValue
dblPeakTime
dblPeakWidth

%% test getonset
%passes
rng(1,'mt19937ar')
[dblOnset,dblValue,dblBaseVal,dblPeakTime,dblPeakValue] = getOnset(vecRate2,sMSD2.vecT,dblPeakTime);

%% test ifr
%passes
rng(1,'mt19937ar')
[vecTime, vecRate,dIFR] = getIFR(vecSpikeTimes, vecEventTimes);
plot(vecTime,vecRate)

%% test plotting
%passes
rng(1,'mt19937ar')
intResampNum = 100;
intPlot = 4;
intLatencyPeaks = 4;
vecRestrictRange = [-inf inf];
boolDirectQuantile = false;
dblJitterSize = 2;
boolStitch = true;
matEventTimes = [vecEventTimes vecEventTimes+0.5];
[dblZetaP,sZETA,sRate,vecLatencies] = zetatest(vecSpikeTimes,matEventTimes,dblUseMaxDur,...
    intResampNum,intPlot,intLatencyPeaks,vecRestrictRange,boolDirectQuantile,dblJitterSize,boolStitch);
