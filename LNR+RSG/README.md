## LNR vs. RSG

This is a pytorch implementation of LNR to compare the performance of RSG on the source code of CVPR 2021 paper "RSG: A Simple but Effective Module for Learning Imbalanced Datasets". 

1. To reimplement the result of ResNet-32 on long-tailed CIFAR-10 (imbalanced ratio = 100) with LDAM-RSG:

   ```
   python cifar_train.py --imb_type exp --imb_factor 0.01  --loss_type LDAM --train_rule DRW
   ```

2. To reimplement the result of ResNet-32 on step CIFAR-100 (imbalance = 100) with LDAM-DRW:

   ```
   python cifar_train.py --dataset cifar100 --imb_type step --imb_factor 0.01 --loss_type LDAM --train_rule DRW --onlyldam 1
   ```

3. To reimplement the result of ResNet-32 on step CIFAR-10 (imbalance = 100) with LDAM-LNR:

   ```
   python cifar_train.py --imb_type step --imb_factor 0.01 --loss_type LDAM --train_rule DRW --addnoise 1
   ```


Citation
-----------------
   Thanks to the authors.
  ```
  @inproceedings{Jianfeng2021RSG,
    title = {RSG: A Simple but Effective Module for Learning Imbalanced Datasets},
    author = {Jianfeng Wang and Thomas Lukasiewicz and Xiaolin Hu and Jianfei Cai and Zhenghua Xu},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2021}
  }
  ```
