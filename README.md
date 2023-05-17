# USSPA: Symmetric Shape-Preserving Autoencoder for Unsupervised Real Scene Point Cloud Completion

This repository contains PyTorch implementation for **Symmetric Shape-Preserving Autoencoder for Unsupervised Real Scene Point Cloud Completion** (CVPR2023).


## Start
### Requirements

```
CUDA                            10.2    ~   11.1
python                          3.7
torch                           1.8.0   ~   1.9.0
numpy
lmdb
msgpack-numpy
ninja                              
termcolor
tqdm
open3d                           
h5py
```
We successfully build the [pointnet2 operation lib](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib) with `CUDA 10.2 + torch 1.9.0` and `CUDA 11.1 + torch 1.8.0`, separately. It should work with PyTorch `1.9.0+`.

## Install 
```
cd code/util/pointnet2_ops_lib
python setup.py install
```

## Pretrained Models
Download ([NJU BOX](https://box.nju.edu.cn/f/56de6132f1ac4621a56b/) code:usspa, [Baidu Yun](https://pan.baidu.com/s/1gWDewT5Pi00fQ5Sf3xriSg?pwd=boqx) code:boqx) and extract our pretrained models as the `weights` folder in `code/network`.
The `weights` folder should be 
```
weights
├── usspa
│   ├── all
│   │   └── model-120.pkl
│   ├── chair
|   |   └── ...
│   └── ...
├── scannet_scanobj
│   └── ...
└── scanobj
    └── ...
```

## Datasets
Download ([NJU Box](https://box.nju.edu.cn/d/4308d5e0f03e48ee9d5c/) code:usspa, [Baidu Yun](https://pan.baidu.com/s/1nFelWBu0V88cEsFeWoHmgQ?pwd=sbo2) code:sbo2) and extract our dataset and ShapeNet dataset to the `data` folder. And download PCN dataset following [PoinTr](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md). The `data` folder should be
```
data
├── PCN
|   └── ...
├── RealComData
└── RealComShapeNetData
```

## Evaluation
```
cd code/network
```
Evaluate completion results of USSPA on our dataset for single-category and multi-category.
```
python test.py --class_name [all, chair, ...]
```
Evaluate completion results of USSPA(classifier) on our dataset for multi-category.
```
python test_classifier.py
```
Evaluate completion results of USSPA on PCN dataset.
```
python test_pcn.py --class_name [chair, table, ...]
```

## Train
```
cd code/network
```
Train USSPA on our dataset for single-category and multi-category.
```
python train.py --class_name [all, chair, ...]
```
Train USSPA(classifier) on our dataset for multi-category.
```
python train_classifier.py
```
Train USSPA on PCN dataset.
```
python train_pcn.py --class_name [chair, table, ...]
```

## License
MIT License

## Acknowledgements
[pointnet2 operation lib](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib)

[Scan2CAD](https://github.com/skanti/Scan2CAD)

[ScanNet](https://github.com/ScanNet/ScanNet)

[ShapeNet](https://shapenet.org/)

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{ma2023usspa,
  title={Symmetric Shape-Preserving Autoencoder for Unsupervised Real Scene Point Cloud Completion},
  author={Ma, Changfeng and Chen, Yinuo and Guo, Pengxiao and Guo, Jie and Wang, Chongjun and Guo, Yanwen},
  booktitle={CVPR},
  year={2023}
}
```