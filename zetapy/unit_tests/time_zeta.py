# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 13:43:58 2023

@author: Jorrit
"""
# imports
import timeit

# time
import_module = "import numpy as np; from zetapy import zetatest"
testcode = '''
np.random.seed(1)
intN = 100
vecP = np.zeros(100)
for i in range(intN):
    dblEndT = 10;
    vecSpikeTimes = dblEndT*np.sort(np.random.rand(100,1),axis=0) #the same
    vecEventTimes = np.arange(0,dblEndT,1.0) #the same
    p,dZETA,dRate,vecLatencies = zetatest(vecSpikeTimes,vecEventTimes)
    vecP[i] = p
'''

dblTimeOut = timeit.repeat(stmt=testcode, setup=import_module, repeat=1, number = 1)
print(dblTimeOut)

#N=100 takes 0.822 s in matlab
#N=100 takes 2.17 s in python