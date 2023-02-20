# CAC
This is the implementation of [**Learning Context-aware Classifier for Semantic Segmentation**](https://arxiv.org/abs/2010.05210) (AAAI 2023). 

![cac](https://user-images.githubusercontent.com/68939582/219602560-2e6d85ef-ce07-48cd-ae76-08c21cdf45d6.png)


# Get Started

### Acknowledgement
This repo is built upon [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation). Many thanks to the contributors.

### Environment
+ mmcv-full 1.5.0
+ mmsegmentation 0.30.0
+ timm 0.5.4
+ numpy 1.21.0
+ torch 1.9.1
+ CUDA 11.1

You can directly run the following command to create the environment:
 
    pip install -r cac_requirements.txt
    
For more details, please refer to the dependancy requirments of [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation).


### Dataset & Training & Testing
The data preparation, training and testing strictly follows that of MMSegmentation, please refer to the [**document**](https://mmsegmentation.readthedocs.io/en/latest/) for more details.


# Results and models
We reproduce the results on ADE20K with this repo as examples for using the context-aware classifier. For reproducing the results on other benchmarks, please refer to the [**configurations**](https://github.com/open-mmlab/mmsegmentation/tree/master/configs) of mmsegmentation and change the decoder heads accordingly. 

We note that, except for UperNet with Swin-B and Swin-L trained with 8 gpus, the other models are trained with 4 gpus due to the limited computational resources, such that the batch-size used for training UperNet with Swin-T becomes 4 instead of 2. 

| Method | Backbone |   mIoU | config                                                                                                                | download                                                                                                                                                                                                                                                                                                                               |
| ------ | -------- |   ----: | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FCN + CAC                     | MobileNet-V2  | 37.42 | [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/mobilenet_v2/fcn_m-v2-d8_512x512_160k_ade20k_cac.py)  | [model](-)      |
| DeepLabV3Plus + CAC           | R-50-D8       | 46.29 | [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k_cac.py)  | [model](-)      |
|  OCRNet + CAC                 | HRNetV2p-W18  | 44.53 | [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/ocrnet/ocrnet_hr18_512x512_160k_ade20k_cac.py)  | [model](-)      |
|  UperNet + CAC                 | Swin-T       | 46.91 |  [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_bs4_cac.py)  | [model](-)      |
|  UperNet + CAC        | Swin-B  (IN-22K)      | 52.46 |  [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_cac.py)  | [model](-)      |
|  UperNet + CAC        | Swin-L  (IN-22K)      | 52.75 |  [config](https://github.com/tianzhuotao/Context-aware-Classifier/blob/main/configs/swin/upernet_swin_large_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_cac.py)  | [model](-)     |




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
