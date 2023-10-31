# zetapy
Repository containing ZETA functions and dependencies. For an example of how to use the code, check example.py. 

Now available as pip package! Simply "pip install zetapy".

Note on updates and maintenance: the original code is written and maintained in MATLAB by Jorrit Montijn (https://github.com/JorritMontijn/zetatest). This Python repository and the pip package are maintained by Jorrit Montijn and Guido Meijer.

Our pre-print describing data-stitching, the time-series ZETA-test, and the two-sample tests is now online: xxx!

The article describing the original ZETA-test has been published in eLife: https://elifesciences.org/articles/71969

The ZETA-test for spiking data has been extensively tested on real and artificial data, and has been peer-reviewed. The time-series ZETA-test and two-sample ZETA-tests are also thoroughly tested and are described in our pre-print, which we will submit for peer-review soon. We are confident all ZETA-tests, including the two-sample tests, are reliable and statistically sound, but if you find any bugs, do let us know on the Issues page!

This repository contains five main functions:

 - zetatest: Calculates the Zenith of Event-based Time-locked Anomalies (ZETA) for spike times of a single neuron. Outputs a p-value.
 - zetatstest: Calculates the time-series version of ZETA, for data such as calcium imaging or EEG recordings.
 - zetatest2: Same as (1), but for testing whether two neurons respond differently to the same stimulus; or whether one neuron responds differently to two sets of stimuli.
 - zetatstest2: Same as (2), but for testing differences between two time-series data arrays.
 - ifr: Calculates the instantaneous firing rate (IFR) without running the ZETA-test. Use this as you would a PSTH function.

Rationale for ZETA

Neurophysiological studies depend on a reliable quantification of whether and when a neuron responds to stimulation, be it sensory, optogenetically or otherwise. However, current statistical analysis methods to determine a neuron’s responsiveness require arbitrary parameter choices, such as a binning size. This choice can change the results of the analysis, which invites bad statistical practice and reduces the replicability of analyses. Moreover, many methods, such as bin-wise t-tests, only detect classically mean-rate modulated cells. Especially with advent of techniques that yield increasingly large numbers of cells, such as Neuropixels recordings or two-photon calcium imaging, it is important to use tests for cell-inclusion that require no manual curation. Here, we present a new family of statistical tests for responses in point-event and time-series data for one- and two-sample comparisons: the family of ZETA-tests. As shown in our papers, they outperform approaches such as optimally-binned ANOVAs, t-tests and model-based approaches, in the sense that it includes more cells in real neurophysiological data at a similar false-positive rate.

Finally, ZETA’s timescale-, parameter- and binning-free nature allowed us to implement a ZETA-derived algorithm to calculate peak onset and offset latencies in neuronal spike trains with theoretically unlimited temporal resolution.

Please send any questions or comments to j.montijn at nin.knaw.nl.

![zeta_image](https://user-images.githubusercontent.com/15422591/135059690-2d7f216a-726e-4080-a4ec-2b3fae78e10c.png)
