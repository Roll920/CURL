# PyTorch Implementation of CURL
* Neural Network Pruning with Residual-Connections and Limited-Data, CVPR 2020, Oral.
* [[CVF open acess]](http://openaccess.thecvf.com/content_CVPR_2020/html/Luo_Neural_Network_Pruning_With_Residual-Connections_and_Limited-Data_CVPR_2020_paper.html)

## Requirements 
PyTorch environment:
* Python 3.6.9
* PyTorch 1.2.0
* [torchsummaryX](https://github.com/nmhkahn/torchsummaryX)
* yaml 3.12

## Prune on ImageNet
1. clone this repository.
2. download the ImageNet dataset and organize this dataset with train and val folders.
3. select subfolder:
   ```
   cd ImageNet/CURL
   ```
4. start pruning and fine-tuning:
   ```
   ./run_this.sh
   ```

Note:
The training log files on ImageNet are missing. We provide the pruned model: ``ImageNet/released_model/ResNet50-CURL-1G.pth``. You can run ``ImageNet/released_model/run_this.sh`` to test its accuracy.

## Prune on CUB200
1. clone this repository.
2. download the CUB200 dataset and organize this dataset with train and val folders.
3. expand the small dataset using ``CUB200/expand_dataset.py``
4. select subfolder:
   ```
   cd CUB200/mobilenetv2/
   ```
4. edit the configuration file ``config.yaml``.
5. calculate the importance score for each filter:
   ```
   ./evaluate_importance.sh
   ```
6. fine-tune the pruned model:
   ```
   ./run_this.sh
   ```

Note:
The training log files are provided in corresponding folders.

## Results
We prune the ResNet50 on ImageNet dataset:

| Architecture  | Top-1 Acc.  | Top-5 Acc.  | #MACs   | #Param. |
| ------------- | ------------- | ------------- |  ------------- |  ------------- | 
| ResNet-50 | 76.15% | 92.87% | 4.09G | 25.56M |
| CURL |  73.39%  | 91.46%  | 1.11G  | 6.67M  |

The results of MobileNetV2 on CUB200:

| Architecture  | Top-1 Acc.  | #MACs |
| ------------- | ------------- | ------------- | 
| MobileNetV2-1.0  | 78.77%  | 299.77M  |
| MobileNetV2-0.5 | 73.96% | 96.12M |
| CURL | 78.72% | 96.07M |

The results of ResNet50 on CUB200:

| Architecture  | Top-1 Acc.  | #MACs |
| ------------- | ------------- | ------------- | 
| ResNet50  | 84.76% | 4.09G  |
| ResNet50-CURL | 81.33% | 1.11G |
| CURL | 83.64% | 1.10G |

## Citation
If you find this work is useful for your research, please cite:
```
@InProceedings{Luo_2020_CVPR,
author = {Luo, Jian-Hao and Wu, Jianxin},
title = {Neural Network Pruning With Residual-Connections and Limited-Data},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020},
pages = {1458-1467}
}
```

## Contact
Feel free to contact me if you have any question (Jian-Hao Luo luojh@lamda.nju.edu.cn or jianhao920@gmail.com).

