# Learning an Adaptive Meta Model-Generator for Incrementally Updating Recommender Systems
This is our experimental code for RecSys 2021 paper "Learning an Adaptive Meta Model-Generator for Incrementally Updating Recommender Systems".  

The paper is available [here](https://github.com/danni9594/ASMG/blob/master/paper.pdf).\
The video is available [here](https://dl.acm.org/doi/10.1145/3460231.3474239).\
The slide is available [here](https://github.com/danni9594/ASMG/blob/master/slide.pdf).

# Requirements
tensorflow 1.4.0  
pandas  
numpy  

GPUs with memory >= 10GB

# Data Preprocessing
The raw data can be obtained from:  
[Tmall Data](https://tianchi.aliyun.com/dataset/dataDetail?dataId=42) `data_format1`  
[Sobazaar Data](https://github.com/hainguyen-telenor/Learning-to-rank-from-implicit-feedback) `Data > Sobazaar-hashID.csv.gz`  
[MovieLens Data](https://grouplens.org/datasets/movielens/) `ml-25m`  

To preprocess the above raw data, save them in the `raw_data` folder under the root directory, and do

> cd preproc  
> python tmall_preproc.py  
> python soba_preproc.py  
> python ml_preproc.py

The preprocessed datasets will be saved in the `datasets` folder for later use.  

# Pretraining
To simulate the real-world applications, the first 10 periods of dataset are used to pretrain an initial Embedding&MLP base model, and all the compared model updating methods will restore from the same pretrained model. 

To pretrain a model for Tmall/Sobazaar/MovieLens, do

> cd Tmall/pretrain  
> python train_tmall.py

> cd Sobazaar/pretrain  
> python train_soba.py

> cd MovieLens/pretrain  
> python train_ml.py

The pretrained base model will be saved in `Tmall/pretrain/ckpts`, `Sobazaar/pretrain/ckpts` and `MovieLens/pretrain/ckpts` respectively.  

All the hyper-parameters can be easily configured in `train_config` at the beginning of each entry file (i.e., `train_xxx.py`).  

**Note**: pretraining must be done before conducting any model updating method.

# Baselines and Variants
All the compared model updating methods for a specific dataset are contained in the folder named by that dataset.  

Our proposed method:  
`ASMGgru_multi`  

Baseline methods:  
`IU`  
`BU`  
`SPMF`  
`IncCTR`  
`SML`  
`SMLmf`  

Variants of `ASMGgru_multi`:  
`ASMGgru_zero`  
`ASMGgru_full`  
`ASMGgru_single`  
(we do not create a separate folder for `ASMGgru_uniform`, as it can be easily implemented in `ASMGgru_multi`, see the code for more details)  

To perform any of the `ASMGgru` methods, we need to first conduct a run of `IU` to generate the input model sequence.

For example, to perform a run of `IU` experiment for Tmall, do

> cd Tmall/IU  
> python train_tmall.py

Then we can proceed to perform any of the `ASMGgru` methods

> cd Tmall/ASMGgru_multi  
> python train_tmall.py

Other model updating methods can be conducted on their own without any pre-requisite.

Note that for `SMLmf`, since it is based on a different base model (i.e., Matrix Factorization), additional pretraining needs to be performed for this method. 

> cd Tmall/SMLmf/pretrain  
> python train_tmall.py

Then

> cd Tmall/SMLmf/SML  
> python train_tmall.py

All the hyper-parameters can be easily configured in `train_config` at the beginning of each entry file (i.e., `train_xxx.py`).  

The evaluation results can be found from the path with the following format:  

><Dataset_Name>/<Method_Name>/ckpts/<Model_Alias>/<Last_Period_Alias>/test_metrics.txt  

where `<Model_Alias>` is configured in `train_config` of the entry file, containing some essential hyper-parameter settings, and `<Last_Period_Alias>` by default is `date20141030` for Tmall and `period30` for MovieLens and Sobazaar.  

Here are some examples of the possible paths that the evaluation results may reside:
 
`Tmall/ASMGgru_multi/ckpts/ASMGgru_multi_linear_train11-23_test24-30_4emb_4mlp_1epoch_3_0.01/date20141030/test_metrics.txt`
   
`MovieLens/IU/ckpts/IU_train11-23_test24-30_1epoch_0.001/period30/test_metrics.txt`


# Citation

If you find this repo useful in your research, please cite the following:

```
@inproceedings{peng2021learning,
  title={Learning an Adaptive Meta Model-Generator for Incrementally Updating Recommender Systems},
  author={Peng, Danni and Pan, Sinno Jialin and Zhang, Jie and Zeng, Anxiang},
  booktitle={Fifteenth ACM Conference on Recommender Systems},
  pages={411--421},
  year={2021}
}
```
