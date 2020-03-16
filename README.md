# Macroscope
Reproduction and Replication Study

### Following is a step by step guide

1. Run **DataClean.ipynb** for initial data cleaning and pre-processing of data. Also this adds some basic network features.
2. Then run **GenerateNetworkGraphs.ipynb** to generate the network graph. This graph has an initial setting to run for references in WOS for 2 hop.
3. Now run **Node2Vec.ipynb** to generate the embeddings. Change the file name appropriately in the main function for input and output file.
4. Run **AddNetworkFetaures.ipynb** to add the generated embeddings from the graph to our dataset. Here also give proper file names.
5. Finally run **Model.ipynb** to train and test or various models.

#### Model.ipynb:
This file has a class: model that has the following public functions:
1. select_best_features_chi2: This will generate the best features. Of which the top 10 will be used to train our models.
2. modelling: This function is used for training and testing. It works with the default K-folds partitioning.
3. modelling_custom_kfolds: This function is also used for training and testing. It works with the custome K-folds partitioning.
4. tuning_hyperparameters: This function is used for hyperparameter tuning.
