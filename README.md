# Active Learning for Quality Engineering
This repo contains the code for replicating the results from the paper on Active Learning published in Quality Engineering.

The images used in this paper are from the MVTec datasets and they can be requested [here](https://www.mvtec.com/company/research/datasets/mvtec-ad).

This repo contains:
1. `requirements.txt`: file that contains a list of all the packages installed in the environment used, along with their version numbers.
2. `feature_extraction.ipynb`: notebook showing how to obtain preprocessed features from the imagse using a pre-trained ResNet-18.
4. `sampling_strategies.py`: functions for performing random sampling and margin sampling.
5. `active_learning.ipynb`: notebook showing how the implementation of the active learning strategy.
