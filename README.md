# workflow-performance-prediction-jii
# Usage:
## 1 To reproduce experiment 1
python run_exp1 --pred_task=3/5/7 #(default=7) --pred_tgt=CPU/MEM #(default=CPU) --pred_mode=PRIOR_1/PRIOR_ALL #(default=PRIOR_ALL) --use_DAG=T/F #(default=T)
## 2 To reproduce experiment 2
python run_exp2 --model_name=CNN/LSTM/VanillaTransformer/DAGTransformer --split=split9_05_05/split8_1_1/split6_2_2 #(default=split6_2_2)
## 3 To reproduce experiment 3
python run_exp2 --model_name=CNN/LSTM/VanillaTransformer/DAGTransformer/GCN --split=split9_05_05/split8_1_1/split6_2_2 #(default=split6_2_2) --GCN_mode=bidirect/unidirect #(default=bidirect)

# Datasets:
## 1 In directory datasets_exp1/
There are 6 different sub-datasets, whose name indicates pred_task+pred_tgt. In each sub-dataset, for example, in 3CPU/, there are 3 DAG information files(train, val, and test) and 2 performance datasets(train, val, and test) using different pred_mode(i.e., PRIOR_1 and PRIOR_ALL).

## 2 In directory datasets_exp2_3/
There are 3 different splits. In each split, there are 3 DAG information files(train, val, and test) and their corresponding performance data(train, val, and test).

# Requirements:
CUDA==11.0
python==3.8
pytorch==1.7.0
PyG==corresponding version of pytorch-1.7.0 and CUDA-11.0 
