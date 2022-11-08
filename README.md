# reweight-imbalance-classification-with-OT

Code for  "Learning to Re-weight Examples with Optimal Transport for Imbalanced Classification", in NeurIPS 2022.


Requirements:

Python 3.6
PyTorch 1.7.1
tqdm 4.19.9
torchvision 0.8.2
numpy 1.19.2



Stage1: 

Pretrain the backbone with the imbalanced training set. See paper for more detailes.

Adjust your file path according to the code.

Stage2:

Learn the weight vector by optimizing OT loss and update the recognition model.
Run: OT_train.py


Abstract: Imbalanced data pose challenges for deep learning based classification models. One
of the most widely-used approaches for tackling imbalanced data is re-weighting,
where training samples are associated with different weights in the loss function.
Most of existing re-weighting approaches treat the example weights as the learnable
parameter and optimize the weights on the meta set, entailing expensive bilevel
optimization. In this paper, we propose a novel re-weighting method based on
optimal transport (OT) from a distributional point of view. Specifically, we view
the training set as an imbalanced distribution over its samples, which is transported
by OT to a balanced distribution obtained from the meta set. The weights of
the training samples are the probability mass of the imbalanced distribution and
learned by minimizing the OT distance between the two distributions. Compared
with existing methods, our proposed one disengages the dependence of the weight
learning on the concerned classifier at each iteration. Experiments on image,
text and point cloud datasets demonstrate that our proposed re-weighting method
has excellent performance, achieving state-of-the-art results in many cases and
providing a promising tool for addressing the imbalanced classification issue.



@inproceedings{Guo2022reweight,
title={Learning to Re-weight Examples with Optimal Transport for Imbalanced Classification},
author={Guo, Dandan and Li, Zhuo and Zheng, Meixi and Zhao, He and Zhou, Mingyuan and Zha, Hongyuan},
booktitle={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)},
year={2022}
}
