# Compressed Interaction Graph based Framework for Muti-behavior Recommendation

This repository contains TensorFlow codes and datasets for the paper.

## Environment
The codes of CIGF are implemented and tested under the following development environment:
* python=3.6.12
* tensorflow=1.14.0
* numpy=1.16.0
* scipy=1.5.2

## Datasets
We utilized three datasets to evaluate CIGF: <i>Beibei, Tmall, </i>and <i>IJCAI Contest</i>. The <i>purchase</i> behavior is taken as the target behavior for all datasets. The last target behavior for the test users are left out to compose the testing set. We filtered out users and items with too few interactions.

## Just Run It！

* Beibei
```
python base_framework_origin_final.py --data beibei 
```
* Tmall
```
python base_framework_origin_final.py --data tmall --gnn_layer 4
```
* IJCAI
```
python base_framework_samp_final.py --data ijcai
```

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{cigf,
  title={Compressed Interaction Graph based Framework for Multi-behavior Recommendation},
  author={Guo, Wei and Meng, Chang and Yuan, Enming and He, Zhicheng and Guo, Huifeng and Zhang, Yingxue and Chen, Bo and Hu, Yaochen and Tang, Ruiming and Li, Xiu and others},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={960--970},
  year={2023}
}
```
