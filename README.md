# GSP-DML: Generalized Sum Pooling for Metric Learning
Official Tensorflow and PyTorch Implementation of "Generalized Sum Pooling for Metric Learning"

# quick info
**tensorflow:** Implements GSP layer as well as many DML methods and provides tensorflow alternative of [MLRC](https://github.com/KevinMusgrave/pytorch-metric-learning) benchmarking framework.

**pytorch:** Implements GSP layer and applies it to Inta-Batch, building on [official Intra-Batch](https://github.com/dvl-tum/intra_batch) framework.  

A detailed guide will be prepared soon.


# requirements:
tensorflow >= 2.8

yaml, numpy, PIL, sklearn, scipy, matplotlib, imageio, pprint

# instructions for benchmarking framework in tensorflow:
(1) put custom config files in ./metric_learning/configs

(2) run >python trainModel.py command with arguments

(2.1) dataset can be passes as an argument

(2.2) dataset is downloaded automatically

(3) different configurations can be experimented by changing related .yaml files
