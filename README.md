# NINet
### [Project Page](http://www.ok.sc.e.titech.ac.jp/res/DeepSM/acmmm22.html)
This repo is the official implementation of **[Digging Into Normal Incorporated Stereo Matching, ACM MM2022](https://dl.acm.org/doi/abs/10.1145/3503161.3548312)**

### Introduction
In this paper, we propose a normal incorporated joint learning framework consisting of two specific modules named non-local disparity propagation(NDP) and affinity-aware residual learning(ARL). The estimated normal map is first utilized for calculating a non-local affinity matrix and a non-local offset to perform spatial propagation at the disparity level. To enhance geometric consistency, especially in low-texture regions, the estimated normal map is then leveraged to calculate a local affinity matrix, providing the residual learning with information about where the correction should refer and thus improving the residual learning efficiency.

## The code is still under organized, completed version coming soon.
## Prerequisites
- Python 3.9, PyTorch >= 1.8.0
- CUDA ToolKit for DCN-V2 Compile

## Training
- TODO


## Testing
 - TODO
## Citation

```
@inproceedings{liu2022digging,
  title={Digging Into Normal Incorporated Stereo Matching},
  author={Liu, Zihua and Zhang, Songyan and Wang, Zhicheng and Okutomi, Masatoshi},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={6050--6060},
  year={2022}
}
```

