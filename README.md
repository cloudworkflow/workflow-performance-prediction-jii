# workflow-performance-prediction-jii
This source repository is dedicated for the following published journal paper: 
#### Jixiang Yu, Ming Gao*, Yuchan Li, Zehui Zhang, WAI HUNG IP, KAI LEUNG Yung, Workflow performance prediction based on graph structure aware deep attention neural network, Journal of Industrial Information Integration, 2022, https://doi.org/10.1016/j.jii.2022.100337. (https://www.sciencedirect.com/science/article/pii/S2452414X22000097)
If you are interested in this research and use this code, please kindly reference our paper or contact the corresponding author.

This dataset is extracted and aggregated based on cluster-trace-v2018 (https://github.com/alibaba/clusterdata)


### Highlights
#### In workflow performance prediction, DAG structure matters;
#### DAG-Transformer effectively embeds the DAG information and outperforms mainstream ML, DL and GCN methods;
#### A new dataset for cloud workflow performance prediction is accompanied as well as the source code.

# Usage:
## 1 To reproduce experiment 1
python run_exp1 --pred_task=3/5/7 #(default=7) --pred_tgt=CPU/MEM #(default=CPU) --pred_mode=PRIOR_1/PRIOR_ALL #(default=PRIOR_ALL) --use_DAG=T/F #(default=T)
## 2 To reproduce experiment 2
python run_exp2 --model_name=CNN/LSTM/VanillaTransformer/DAGTransformer --split=split9_05_05/split8_1_1/split6_2_2 #(default=split6_2_2)
## 3 To reproduce experiment 3
python run_exp2 --model_name=CNN/LSTM/VanillaTransformer/DAGTransformer/GCN --split=split9_05_05/split8_1_1/split6_2_2 #(default=split6_2_2) --GCN_mode=bidirect/unidirect #(default=bidirect)

# Datasets(You need to extract the *.zip files first):
## 1 In directory datasets_exp1/
There are 6 different sub-datasets, whose name indicates pred_task+pred_tgt. In each sub-dataset, for example, in 3CPU/, there are 3 DAG information files(train, val, and test) and 2 performance datasets(train, val, and test) using different pred_mode(i.e., PRIOR_1 and PRIOR_ALL).

## 2 In directory datasets_exp2_3/
There are 3 different splits. In each split, there are 3 DAG information files(train, val, and test) and their corresponding performance data(train, val, and test).

# Requirements:
CUDA==11.0
python==3.8
pytorch==1.7.0
PyG==corresponding version of pytorch-1.7.0 and CUDA-11.0 
