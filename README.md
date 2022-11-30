# PDNet: Toward Better One-Stage Object Detection With Prediction Decoupling
A PyTorch implementation of [PDNet: Toward Better One-Stage Object Detection With Prediction Decoupling](https://ieeexplore.ieee.org/document/9844453). 
*IEEE Transactions on Image Processing*, 2022. 


## Introduction

We propose a **P**rediction-target-**D**ecoupled **Net**work (PDNet) for object detection, which decouples the prediction of different targets (*i.e.*, object category and boundaries) to their respective proper positions. The following shows the maps of decoupled regression predictions for object box localization.

![reg_pred_map](demo/demo_reg_pred_map.png)


## Installation
This implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), so the installation procedure is pretty much the same. Please check [INSTALL.md](INSTALL.md) for details. Additionally, remember to build the CUDA implementation of the prediction collection module with the following commands.
```bash
cd pdnet_core/pred_collect_ops
python setup.py build develop
```


## Model Zoo

We provide the following trained models with different backbones or training settings.

Model | Training iterations | Training scales | AP (minival) | Link
--- |:---:|:---:|:---:|:---:
PDNet_R_50_FPN_1x | 90k | 800 | 41.8 | [cfg](configs/pdnet/pdnet_R_50_FPN_1x.yaml) / [weights]()
PDNet_R_50_FPN_2x | 180k | [640, 800] | 44.3 | [cfg](configs/pdnet/pdnet_R_50_FPN_2x.yaml) / [weights]()
PDNet_R_50_FPN_2xx | 180k | [480, 960] | 45.0 | [cfg](configs/pdnet/pdnet_R_50_FPN_2xx.yaml) / [weights]()
PDNet_R_101_FPN_2x | 180k | [640, 800] | 45.7 | [cfg](configs/pdnet/pdnet_R_101_FPN_2x.yaml) / [weights]()
PDNet_R_101_FPN_2xx | 180k | [480, 960] | 46.6 | [cfg](configs/pdnet/pdnet_R_101_FPN_2xx.yaml) / [weights]()
PDNet_X_101_64x4d_FPN_2x | 180k | [640, 800] | 47.4 | [cfg](configs/pdnet/pdnet_X_101_64x4d_FPN_2x.yaml) / [weights]()
PDNet_X_101_64x4d_FPN_2xx | 180k | [480, 960] | 48.7 | [cfg](configs/pdnet/pdnet_X_101_64x4d_FPN_2xx.yaml) / [weights]()


## Inference
Evaluate the trained model on the minival split of MS COCO dataset. Please change the arguments of `--config-file` and `--ckpt` to evaluate different models.
```bash
python tools/test_net.py --config-file configs/pdnet/pdnet_R_50_FPN_1x.yaml --ckpt PDNet_R_50_FPN_1x.pth
```

## Training
We train our detection models on 4 GPUs with the following command. We prepare the configuration files for training different models in ``configs/pdnet/``.
```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --config-file configs/pdnet/pdnet_R_50_FPN_1x.yaml
```

## Citation
Please cite our paper if this project helps your research.
```
@article{yang2022pdnet,
  title={PDNet: Toward Better One-Stage Object Detection With Prediction Decoupling},
  author={Yang, Li and Xu, Yan and Wang, Shaoru and Yuan, Chunfeng and Zhang, Ziqi and Li, Bing and Hu, Weiming},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={5121--5133},
  year={2022},
  publisher={IEEE}
}
```