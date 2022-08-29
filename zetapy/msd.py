import logging
from math import log
from scipy.stats import norm
from scipy.signal import convolve2d
import numpy as np


def getMultiScaleDeriv(vecT, vecV, intSmoothSd=0, dblMinScale=None, dblBase=1.5, intPlot=0,
                       dblMeanRate=1, dblUseMaxDur=None, axs=None):
    """Returns multi-scale derivative. Syntax:
       [vecRate,sMSD] = getMultiScaleDeriv(vecT,vecV,intSmoothSd,dblMinScale,dblBase,intPlot,dblMeanRate,dblUseMaxDur)
    Required input:
        - vecT [N x 1]: timestamps (e.g., spike times)
        - vecV [N x 1]: values (e.g., z-scores)

    Optional inputs:
        - intSmoothSd: Gaussian SD of smoothing kernel (in # of samples) [default: 0]
        - dblMinScale: minimum derivative scale in seconds [default: 1/1000]
        - dblBase: base for exponential scale step size [default: 1.5]
        - intPlot: integer, plotting switch 
        - dblMeanRate: mean spiking rate to normalize vecRate (optional)
        - dblUseMaxDur: trial duration to normalize vecRate (optional)
        - axs: two element vector with the handles to the two subplots that are to be plotted

    Outputs:
        - vecRate; Instantaneous spiking rate
        - sMSD; structure with fields:
            - vecRate; instantaneous spiking rates (like a PSTH)
            - vecT; time-points corresponding to vecRate (same as input vecT)
            - vecM; Mean of multi-scale derivatives
            - vecScale; timescales used to calculate derivatives
            - matMSD; multi-scale derivatives matrix
            - vecV; values on which vecRate is calculated (same as input vecV)

    Version history:
    %1.1 - June 18, 2020 created by Jorrit Montijn, translated to python by Alexander Heimel
    """

    ## set default values
    if dblMinScale==None:
        dblMinScale = round(log(1/1000) / log(dblBase))

    if dblUseMaxDur == None:
        dblUseMaxDur = np.max(vecT) - np.min(vecT)

    ## reorder just in case
    ### [vecT,vecReorder] = sort(vecT(:),'ascend');
    ### vecV = vecV(vecReorder);
    ### vecV = vecV(:);
    vecReorder = np.argsort(vecT)
    vecT = np.array(vecT)[vecReorder]
    vecV = np.array(vecV)[vecReorder]

    ## prepare data
    dblMaxScale = log(np.max(vecT)/10) / log(dblBase)
    intN = len(vecT)

    ## get multi-scale derivative
    ### vecExp = dblMinScale:dblMaxScale;
    vecExp = np.arange(dblMinScale,dblMaxScale+1)
    vecScale = [dblBase ** x for x in vecExp]
    intScaleNum = len(vecScale)
    matMSD = np.zeros( (intN,intScaleNum) )

    #logging.warning('Loop parallelization is not yet translated to python.')
    # try %try parallel
    #     parfor intScaleIdx=1:intScaleNum
    #         dblScale = vecScale(intScaleIdx);

    #         %run through all points
    #         for intS=1:intN
    #             %select points within window
    #             matMSD(intS,intScaleIdx) = getD(dblScale,intS,intN,vecT,vecV);
    #         end
    #     end
    # catch %otherwise try normal loop
    for intScaleIdx in range(intScaleNum):
        dblScale = vecScale[intScaleIdx]
        # run through all points
        for intS in range(intN):
            # select points within window
            matMSD[intS,intScaleIdx] = getD(dblScale, intS, intN, vecT, vecV)

    ## smoothing
    if intSmoothSd > 0:
        ### vecFilt = normpdf(-2*(intSmoothSd):2*intSmoothSd,0,intSmoothSd)';
        vecFilt = norm.pdf( range(-2*(intSmoothSd),2*intSmoothSd+1), 0, intSmoothSd)
        vecFilt = vecFilt / sum(vecFilt)
        # pad array
        ### matMSD = padarray(matMSD,floor(size(vecFilt)/2),'replicate');
        x = int(np.floor(len(vecFilt)/2))
        matMSD = np.pad(matMSD,( (x,x),(0,0)),'edge')

        # filter
        ### matMSD = conv2(matMSD,vecFilt,'valid');
        matMSD = convolve2d(matMSD, np.transpose([vecFilt]), 'valid')

    # mean
    ### vecM = mean(matMSD,2);
    vecM = np.mean(matMSD,1)

    # weighted average of vecM by inter-spike intervals
    ### dblMeanM = (1/dblUseMaxDur) * sum(((vecM(1:(end-1)) + vecM(2:end))/2).*diff(vecT));
    dblMeanM = (1/dblUseMaxDur) * sum(((vecM[:-1] + vecM[1:])/2) * np.diff(vecT))

    # rescale to real firing rates
    ### vecRate = dblMeanRate * ((vecM + 1/dblUseMaxDur)/(dblMeanM + 1/dblUseMaxDur));
    vecRate = dblMeanRate * ((vecM + 1/dblUseMaxDur) / (dblMeanM + 1/dblUseMaxDur))

    ## plot
    if intPlot == 1:
        if intSmoothSd > 0:
            strTitle = 'Smoothed MSDs'
        else:
            strTitle = 'MSDs'
        axs[0].imshow(matMSD.T, aspect='auto')
        axs[0].set(yticks=[], ylabel='Scale (s)',
                   xlabel='Timestamp index (#)', title=strTitle)    
        
        if len(vecT) > 1000:
            vecSubset = np.round(np.linspace(0, len(vecT)-1, 1000)).astype(int)
            axs[1].plot(vecT[vecSubset], vecRate[vecSubset])
        else:
            h, edges = np.histogram(vecT, bins=np.linspace(0, vecT[-1], num=1000))
            axs[1].stairs(h, edges)
        
        #vecSubset = np.round(np.linspace(1, len(vecT), 1000))
        #axs[1].plot(vecT[vecSubset], vecRate[vecSubset])
       
        if dblMeanRate == 1:
            strLabelY = 'Time-locked activation (a.u.)'
        else:
            strLabelY = 'Spiking rate (Hz)'
        axs[1].set(xlabel='Time (s)', ylabel=strLabelY, title='Peri Event Plot (PEP)')
        

    ## build output
    sMSD = dict()
    sMSD['vecRate'] = vecRate
    sMSD['vecT'] = vecT
    sMSD['vecM'] = vecM
    sMSD['vecScale'] = vecScale
    sMSD['matMSD'] = matMSD
    sMSD['vecV'] = vecV

    return vecRate, sMSD

def getD(dblScale,intS,intN,vecT,vecV):
    # select points within window
    dblT = vecT[intS]
    dblMinEdge = dblT - dblScale/2
    dblMaxEdge = dblT + dblScale/2
    ### intIdxMinT = find(vecT > dblMinEdge,1);
    ### if isempty(intIdxMinT)
    ###     intIdxMinT=1
    ### end
    intIdxMinT = np.where(vecT > dblMinEdge)[0]
    if len(intIdxMinT) > 0:
        intIdxMinT = intIdxMinT[0]
    else:
        intIdxMinT = 0

    ### intIdxMaxT = find(vecT > dblMaxEdge,1);
    ### if isempty(intIdxMaxT)
    ###     intIdxMaxT=intN
    ### end
    intIdxMaxT = np.where(vecT > dblMaxEdge)[0]
    if len(intIdxMaxT) > 0 :
        intIdxMaxT = intIdxMaxT[0]
    else:
        intIdxMaxT = intN - 1

    ### if intIdxMinT == intIdxMaxT && intIdxMinT > 1
    ###     intIdxMinT=intIdxMaxT-1
    ### end
    if intIdxMinT == intIdxMaxT and intIdxMinT > 0 :
        intIdxMinT = intIdxMaxT-1

    ### dbl_dT = max([dblScale (vecT(intIdxMaxT) - vecT(intIdxMinT))]);
    ### dblD = (vecV(intIdxMaxT) - vecV(intIdxMinT))/dbl_dT;
    dbl_dT = max([dblScale, (vecT[intIdxMaxT] - vecT[intIdxMinT])])
    dblD = (vecV[intIdxMaxT] - vecV[intIdxMinT]) / dbl_dT
    return dblD
