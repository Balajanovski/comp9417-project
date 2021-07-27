# comp9417-project
Project created for comp9417. Kaggle competition can be found at https://www.kaggle.com/c/quora-insincere-questions-classification 

## Pre-processing and word vectors
**IMPORTANT:** before running any model, this step must be completed. Required files/subfolders can be found in the Kaggle competition downloads and all paths are relative to the root folder. 
- Ensure that the training data at path `data/train.csv` exists. The training and test data for the models will be sampled from this file. 
- Ensure that the file at path `embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin` exists. Simply extract the `embeddings.zip` file provided in the competition downloads.
- Execute `python -m src.preprocess` to generate the processed data (takes around 30min on CPU)

## Models
All models should be run from the root directory. when the parameter `[data_file]` is specified, enter the name of a file in `processed_data/`. The data for the model will be sourced from here. 

Also, for some models, consider piping the output to a text file, as it can be large. 

Example model execution: `python -m src.naive_bayes bernoulli 5 punct_removed.csv`

### Naive Bayes

`python -m src.naive_bayes [distribution_type] [ngram_max] [data_file]`

where `distribution_type` is either `bernoulli` or `multinomial`, `ngram_max` is the maximum n-gram length to consider. Trains a naive bayes classifier on each of the composite ngram lengths `[1,1]`, `[1,2]`...`[1,ngram_max]`. Outputs metrics and a plot. 

#### OR

`python -m src.naive_bayes [distribution_type] [ngram_min] [ngram_max] [data_file]`

Trains a single naive bayes model in the composite ngram length range of `[ngram_min,ngram_max]`. Outputs metrics. 

#### OR

`python -m src.naive_bayes_gridsearch [distribution_type] [data_file]`

Performs a grid search over Laplace smoothing parameters on naive bayes models of the specified distribution type. To configure the exact settings, please view the code. Outputs parameter search results and the metrics of the best performing model. 

### Support Vector Machine

`python -m src.svm [distribution_type] [kernel] [c0] [c1] [data_file]`

where `distribution_type` is either `bernoulli` or `multinomial` for bag-of-words or `word2vec` for word vectors. `kernel` is the choice of kernel (can be `linear`, `poly`, `rbf`, `sigmoid`). `c0` and `c1` are the class weights for the negative and positive class, respectively. 

Trains a single SVM model and outputs metrics. 

#### OR

`python -m src.svm [distribution_type] [kernel] [data_file]`

where `distribution_type` is either `bernoulli` or `multinomial` for bag-of-words or `word2vec` for word vectors. `kernel` is the choice of kernel (can be `linear`, `poly`, `rbf`, `sigmoid`). 

Performs a grid search over class weights on SVM models of the specified parameters. To configure the exact settings, please view the code. Outputs parameter search results and the metrics of the best performing model. 

### Tree Learning - Random Forest

`python -m src.random_forest [max_tree_depth] [num_trees] [data_distribution] [data_file]`

where `max_tree_depth` is the maximum depth of the tree and `num_trees` is the number of models in the ensemble. `data_distribution` is either `bernoulli` for bag-of-words or `word2vec` for word vectors. 