%% random numbers
%passes; same as python
rng(1,'mt19937ar')
dblEndT = 10;
vecSpikeTimes = dblEndT*sort(rand(100,1)); %the same
vecEventTimes = (0:(dblEndT-1))'; %the same

%% test getPseudoSpikeVectors
%passes; same as python
dblWindowDur = 1.0;
boolDiscardEdges = false;

[vecPseudoSpikeTimes,vecPseudoEventT] = getPseudoSpikeVectors(vecSpikeTimes,vecEventTimes,dblWindowDur,boolDiscardEdges);

%% getspiket
%passes
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
dblWindowDur = 1.0;
boolDiscardEdges = false;
[vecPseudoSpikeTimes,vecPseudoEventT] = getPseudoSpikeVectors(vecSpikeTimes,vecEventTimes,dblWindowDur,boolDiscardEdges)
[vecRealDeviation, vecRealFrac, vecRealFracLinear, vecSpikeT] = getTempOffsetOne(vecPseudoSpikeTimes, vecPseudoEventT, dblWindowDur)


%% test getmsd
%passes
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
[dblPeakValue,dblPeakTime,dblPeakWidth,vecPeakStartStop,intPeakLoc,vecPeakStartStopIdx] = getPeak(vecRate2, sMSD2.vecT);
dblPeakValue
dblPeakTime
dblPeakWidth

%% test getonset
%passes
[dblOnset,dblValue,dblBaseVal,dblPeakTime,dblPeakValue] = getOnset(vecRate2,sMSD2.vecT,dblPeakTime);

%% test ifr
%passes
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
