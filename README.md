# AutoPruner

This repository contains source code of research paper "AutoPruner: Transformer-based Call Graph Pruning", which is submitted to ESEC/FSE 2022

## Environment Configuration
```
conda env create --file enviroment.yml
```
## Experiments
To replicate the result of AutoPruner, please down the data from this [link](https://zenodo.org/record/6369874#.YjWzmi8RppR) and put in the same folder with this repository, then run following below instructions. Note that, our results may be slightly different when running on different devices. However, this diffences does not affect our findings in the paper. 

### RQ1
To replicate the result of AutoPruner in call graph pruning on Wala (RQ1), please use
```
bash script/rq1_wala.sh
```
To replicate the result of AutoPruner in call graph pruning on Doop (RQ1), please use
```
bash script/rq1_doop.sh
```
To replicate the result of AutoPruner in call graph pruning on Petablox (RQ1), please use
```
bash script/rq1_peta.sh
```

```
### RQ2
#### Null-pointer Analysis
In this analysis, we follow the experimental settings of [cgPruner](http://web.cs.ucla.edu/~akshayutture/papers/icse22_firstPaper_preprint.pdf) including their code of Null-pointer Analysis (NPA). Please refer to cgPruner's [replication package](https://zenodo.org/record/6057691#.YoXA8WBByek) for further instructions. You also can find our manual evaluation in npe_result folder in this [link](https://zenodo.org/record/6369874#.YjWzmi8RppR) 

#### Monomorphic Call-site Detection
To replicate the result of AutoPruner in monomorphic call-site detection on Wala's call graph (RQ1), please use
```
bash script/rq2_wala.sh
```
To replicate the result of AutoPruner in monomorphic call-site detection on Doop's call graph (RQ1), please use
```
bash script/rq2_doop.sh
```
To replicate the result of AutoPruner in monomorphic call-site detection on Petablox's call graph (RQ1), please use
```
bash script/rq2_peta.sh
```

### RQ3
To replicate the ablation study of AutoPruner with strutural features, please use
```
bash script/rq3_structure.sh
```
To replicate the ablation study of AutoPruner with semantic features, please use

```
bash script/rq3_semantic.sh
```
To replicate the ablation study of AutoPruner with caller function, please use

```
bash script/rq3_caller.sh
```
To replicate the ablation study of AutoPruner with callee function, please use

```
bash script/rq3_callee.sh
```
