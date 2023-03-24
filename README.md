# CAC
This is the implementation of [**Learning Context-aware Classifier for Semantic Segmentation**](https://arxiv.org/abs/2303.11633) (AAAI 2023, Oral). This repo provides the implementation of CAC for 2D semantic segmentation. 

CAC is also found effective in **3D semantic segmentation**, achieving competitive performance against recent SOTA methods (e.g., boosting SpUNet to 76% mIoU on Scannet val), and the implementation is available at [**PointCept**](https://github.com/Pointcept/Pointcept) that is a powerful and flexible codebase for point cloud perception research. 

![cac](https://user-images.githubusercontent.com/68939582/219602560-2e6d85ef-ce07-48cd-ae76-08c21cdf45d6.png)


# Get Started

### Acknowledgement
This repo is built upon [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation). Many thanks to the contributors.

```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

### Environment
+ mmcv-full 1.5.0
+ mmsegmentation 0.30.0
+ timm 0.5.4
+ numpy 1.21.0
+ torch 1.9.1
+ CUDA 11.1
    
For more details, please refer to the dependancy requirments of [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation).


### Dataset & Training & Testing
The data preparation, training and testing strictly follows that of MMSegmentation, please refer to the [**document**](https://mmsegmentation.readthedocs.io/en/latest/) for more details.

For example, for training with config.py on 4 GPUs, please run:

    ./tool/dist_train.sh config.py 4 
    
For testing with config.py on 4 GPUs, and the weights are loaded from model.pth, please run:

    ./tool/dist_test.sh config.py model.pth 4 --eval mIoU

# Results and models
We reproduce the results on ADE20K with this repo as examples for using the context-aware classifier. For reproducing the results on other benchmarks, please refer to the [**configurations**](https://github.com/open-mmlab/mmsegmentation/tree/master/configs) of mmsegmentation and change the decoder heads accordingly. One example of FCN is:

![image](https://user-images.githubusercontent.com/68939582/220012772-f8a8bbb3-be27-4aa1-8c01-c8ea43ff2f32.png)


We note that, except for UperNet with Swin-B and Swin-L trained with 8 gpus, the other models are trained with 4 gpus due to the limited computational resources, such that the batch-size used for training UperNet with Swin-T becomes 4 instead of the default value 2 for a fair comparison. This does not affect the baseline performance. 

| Method | Backbone |   mIoU | config                                                                                                                | download                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- |   ----: | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN + CAC                     | MobileNet-V2  | 37.42 | [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k_cac.py)  | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EcpUuw52_8VNj5MML2v3UxgBFvOjq1uS_OBG-DWFeS3jOQ?e=aPE2N5)      |
| DeepLabV3Plus + CAC           | R-50-D8       | 46.29 | [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k_cac.py)  | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/ESnimdpoLbtIkQJNEsw-oo8BBIzZJdvRX8wvb0mArmsisQ?e=OKGeCk)      |
|  OCRNet + CAC                 | HRNetV2p-W18  | 44.53 | [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/ocrnet/ocrnet_hr18_512x512_160k_ade20k_cac.py)  | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EbY82J1OOL9NkTaJZ5vtFf8BDrMS9SKt0J16ZougM9T1_A?e=7S87u8)      |
|  UperNet + CAC                 | Swin-T       | 46.91 |  [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_bs4_cac.py)  | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/Ee3esPXw8SxPhJWo3P-l57AB6kETREaO12G6j3ySonXCHQ?e=SkHU79)      |
|  UperNet + CAC        | Swin-B  (IN-22K)      | 52.46 |  [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_cac.py)  | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EW34a-ogeNBLgHyxP8aTlGkBjPB08ynCXoqFi4BLyYgi5g?e=lP9QvE)      |
|  UperNet + CAC        | Swin-L  (IN-22K)      | 52.75 |  [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/swin/upernet_swin_large_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_cac.py)  | [model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EenCUmdIjpBFtlIDt6-7pJcB46zF2S4SxhRzRGSbdvROdQ?e=cgGvhh)     |




# Citation

If you find this project useful, please consider citing:
```
@InProceedings{tian2023cac,
    title={Learning Context-aware Classifier for Semantic Segmentation},
    author={Zhuotao Tian and Jiequan Cui and Li Jiang and Xiaojuan Qi and Xin Lai and Yixin Chen and Shu Liu and Jiaya Jia},
    booktitle={Proceedings of the Thirty-Seventh {AAAI} Conference on Artificial Intelligence},
    year={2023}
}
```
