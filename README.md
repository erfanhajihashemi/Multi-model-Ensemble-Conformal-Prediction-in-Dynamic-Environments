This is implmention of paper "multi-model Ensemble Conformal Prediction in Dynamic Enviroments. Implementations of prposed algortihms at this paper has been added to code from [paper](https://openreview.net/pdf?id=qqMcym6AmS). To produce results for SAOCPMM method, adaptive_cp_saocpmm.py and synthetic_saocpmm.py files need to be run.

Produce results for realdatesets:  
```shell
python adaptive_cp.py --dataset <dataset> --distributin_shift <distribution_shift> --model <model>
```

Produce result for synthetic dataset
```shell
python synthetic.py --dataset <dataset>  --model <model>
```

For example to see table 1:
```shell
python adaptive_cp.py --dataset CIFAR100 --distribution_shift gradual --model  densenet121 resnet50 resnet18 googlenet
python adaptive_cp_saocpmm.py --dataset CIFAR100 --distribution_shift gradual --model  densenet121 resnet50 resnet18 googlenet
```
