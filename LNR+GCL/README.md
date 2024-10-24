# LNR + GCL 
We adapted LNR on the source code of GCL: [Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Long-Tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment_CVPR_2022_paper.html) based on Pytorch. 

## CIFAR10
Run the GCL on Long-tailed setting
```bash
$ python cifar_train_backbone.py --arch resnet32 /
                                 --dataset cifar10 --data_path './dataset/data_img' /
                                 --loss_type 'GCL' --imb_factor 0.01 -imb_type 'exp'
```

Run the LNR+GCL on Long-tailed setting
```bash
$ python cifar_train_backbone.py --arch resnet32 /
                                 --dataset cifar10 --data_path './dataset/data_img' /
                                 --loss_type 'GCL' --imb_factor 0.01 -imb_type 'exp' --addnoise 1
```

## <a name="Citation"></a>Citation

## Citation
```
@inproceedings{Li2022GCL,
  author    = {Mengke Li, Yiu{-}ming Cheung, Yang Lu},
  title     = {Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment},
  pages = {6929-6938},
  booktitle = {CVPR},
  year      = {2022},
}
```

```
@article{Li2024AGCL,
  author={Li, Mengke and Cheung, Yiu-ming and Lu, Yang and Hu, Zhikai and Lan, Weichao and Huang, Hui},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Adjusting Logit in Gaussian Form for Long-Tailed Visual Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  doi={10.1109/TAI.2024.3401102}}
```


## Acknowledgment
Many thanks to the authors.
