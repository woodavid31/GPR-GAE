# Self-supervised Adversarial Purification for Graph Neural Networks

## Installation
The packages used for running the code (do pip install . to install the modules)
```python
python 3.9.7
pytorch 1.10.2
cudatoolkit 11.3.1
torchvision 0.11.3
pyg 2.0.3
sacred 0.8.2
tqdm 4.62.3
scipy 1.7.3
torchtyping 0.1.4
numba 0.54.1
filelock 3.4.2
numpy 1.20.3
scikit-learn 1.0.2
tqdm 4.62.3
ogb 1.3.2
torchtyping 0.1.4
cvxpy 1.2.1
```

## experiment.py
Running experiment.py 1.Self-supervisedly trains GPR-GAE, 2.Supervisedly trains the surrogate GNN classifier and 3.Evaluate robustness under {adaptive, non-adaptive} attack settings.

```python
python experiment.py --dataset cora --attack PRBCD --surrogate GCN
python experiment.py --dataset cora --attack PRBCD --surrogate GCN --adaptive
```

## experiment_batch.py
A batched version of experiment.py for large scale graphs. We train GPR-GAE in mini batches, sampling a portion of the edges every training epoch. batch_ratio = 0.01 for OGB-arXiv. 

```python
python experiment_batch.py --dataset ogbn-arxiv --attach PRBCD --mask 0.5 --batch_ratio 0.01 --surrogate GCN
```


