## Deep Prediction of Investor Interest: a Supervised Clustering Approach

This repo contains the accompagnying code for the paper https://arxiv.org/abs/1909.05289. Contribution is two-fold:

1. We first show the usefulness of our approach on simulated data exhibiting the characteristics awaited from our model, and compare it to a LGBM benchmark.
2. We use our approach on an open-source dataset (https://zenodo.org/record/2573031#.XaSQ9_fRY5l) to prove its usefulness on real-world data.

Deep learning code is written in **TensorFlow 2.0**.

### Using this repo
#### 1. Simulation

Run *sample_data.py* to sample a dataset - the seed is set so as to get the same results as presented in the paper. *exnet_run.py*, *embedmlp_run.py* and *lgbm_run.py* respectively run the ExNet, the EmbedMLP and the LightGBM algorithms on the previously simulated data.

#### 2. IBEX
Run *create_data.py* to create the dataset described in the article. Run *exnet_run.py* and *lgbm_run.py* to respectively run the ExNet and LGBM algorithms on the previously created data. The weights file corresponding to the best ExNet found is included, and can be investiguated using the *exnet_analysis.py* file.

### Requirements
- python==3.7.3
- pandas==0.25.2
- tensorflow==2.0.0
- tensorflow_addons==0.6.0
- numpy==1.17.2
- lightgbm==2.1.2
- ta (https://github.com/bukosabino/ta)
- umap-learn==0.3.10
