## S³FD_ATSS_SAPD: Single Shot Scale-invariant Face Detector ##
Application of Adaptive Training Sample Selection and Soft Anchor Point Detection to S3FD

based on the PyTorch Implementation of Single Shot Scale-invariant Face Detector

### Description
Meanwhile train hand and head with S3FD,hand dataset is [Egohands Dataset](http://vision.soic.indiana.edu/projects/egohands/),head dataset is [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release),we can download [hand model](https://pan.baidu.com/s/1_d4HqioBJknGj2ypwtYaXg) and [face model](https://pan.baidu.com/s/1epyTAUc6qSt3oZ7veK4oEw)

### Requirement
* pytorch 1.0.X 
* opencv 
* numpy 
* easydict
* torchvision

### Prepare data 
1. download WIDER face dataset、Egohands dataset and SCUT-HEAD
2. modify data/config.py 
3. ``` python prepare_wider_data.py ```
4 ``` python prepare_handataset.py ```


### Train
I chose face dataset for training and evaluation of ATSS and SAPD applications.
``` 
python train.py --batch_size 4 --dataset face
``` 

### Evalution
According to your dataset path, modify data/config.py 

1. test on WIDER FACE 
```
python wider_test.py
```

### Implementation Details
1. Adaptive Training Sample Selection(ATSS)

        I applied ATSS after the original sample selection part of S3FD in bbox_utils.py (line 193-223)

        Unlike the original ATSS algorithm starting from empty candidate set, I used result positive set from stage 2 of S3FD as starting candidate set. Other details follow the original ATSS algorithm.

2. Soft Anchor Point Detection(SAPD)

        I applied SAPD to the smoothed_L1_loss of S3FD in multibox_loss.py (line 109-107) anchor_weight calculation for generalized centerness function is done in bbox_utils.py (line 293)

        I first multiply anchor_weight to the result of smoothed_L1_loss and devide the total sum of loss with the sum of anchor_weight in multibox_loss.py (line 112-114)

        I tried to preserve the main concept of original SAPD while modifying the generalized centerness function to make it fit to the regression loss of S3FD.


### Result
1. test on WIDER FACE 
```
Easy AP    Baseline= 0.927  ATSS_only= 0.927  SAPD_only= 0.927  ATSS_SAPD= 0.927
    
Medium AP  Baseline= 0.927  ATSS_only= 0.927  SAPD_only= 0.927  ATSS_SAPD= 0.927
    
Hard AP    Baseline= 0.927  ATSS_only= 0.927  SAPD_only= 0.927  ATSS_SAPD= 0.927
```

2. test on FDDB
```
Baseline= 0.927  ATSS_only= 0.927  SAPD_only= 0.927  ATSS_SAPD= 0.927
    
Baseline= 0.927  ATSS_only= 0.927  SAPD_only= 0.927  ATSS_SAPD= 0.927
    
Baseline= 0.927  ATSS_only= 0.927  SAPD_only= 0.927  ATSS_SAPD= 0.927
```    




### References
* [S³FD: Single Shot Scale-invariant Face Detector](https://arxiv.org/abs/1708.05237)
* [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424)
* [Soft Anchor-Point Object Detection](https://arxiv.org/abs/1911.12448)
* [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
* [S3FD.pytorch](https://github.com/yxlijun/S3FD.pytorch)