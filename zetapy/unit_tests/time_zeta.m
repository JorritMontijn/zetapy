%% random numbers
%passes; same as python
rng(1,'mt19937ar')
intN = 100;
vecP = nan(1,intN);
for i=1:intN
    dblEndT = 10;
    vecSpikeTimes = dblEndT*sort(rand(100,1));
    vecEventTimes = (0:(dblEndT-1))';
    vecP(i) = zetatest(vecSpikeTimes,vecEventTimes);
end

%N=100 takes 0.822 s in matlab
%N=100 takes 2.17 s in python