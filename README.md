# comp9417-project
Project created for comp9417. Kaggle competition can be found at https://www.kaggle.com/c/quora-insincere-questions-classification 

## Pre-processing and word vectors
**IMPORTANT:** before running any model, this step must be completed. Required files/subfolders can be found in the Kaggle competition downloads and all paths are relative to the root folder. 
- Ensure that the training data at path `data/train.csv` exists. The training and test data for the models will be sampled from this file. 
- Ensure that the file at path `embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin` exists. Simply extract the `embeddings.zip` file provided in the competition downloads.
- Execute `python -m src.preprocess` to generate the processed data (takes around 30min on CPU)

## Training models directly using source code
All model training should be run from the root directory. When the parameter `[data_file]` is specified, enter the name of a file in `processed_data/`, eg `punct_stopwords_removed_lemmatized.csv`. The data for the model will be sourced from here. `[distribution_type]` refers to one of the following:
- `bernoulli`, for the Bernoulli ngram distribution
- `multinomial`, for the multinomial ngram distribution
- `word2vec`, for the Word2Vec distribution weighted with tf-idf index

The exception are the naive bayes models, which only support `bernoulli` and `multinomial`. 

Also, for some models, consider piping the output to a text file, as it can be large. This section is meant to serve as a guide for running the files only. Some configurations are not mentioned. To view/configure the exact settings for any model, please open the source code or view the relevant section in the report. This is especially true for sections which pertain to hyperparameter tuning. 

Furthermore, in addition to all other output, training a single model (not parameter search) will save a corresponding pickled version of the model in `model_pickles/`. All models shown in tables in the report, as well as the best model for each section, will already have a pickled version in this folder. **For running these pickled models, please see the later section.**

Example command for training a model: `python -m src.svm linear 1.0 1.0 punct_stopwords_removed_lemmatized.csv`

### Naive Bayes

`python -m src.naive_bayes [distribution_type] [ngram_max] [data_file]`

where `ngram_max` is the maximum n-gram length to consider. Trains a naive bayes classifier on each of the composite ngram lengths `[1,1]`, `[1,2]`...`[1,ngram_max]`. Outputs metrics and a plot. 

#### OR

`python -m src.naive_bayes [distribution_type] [ngram_min] [ngram_max] [data_file]`

Trains a single naive bayes model in the composite ngram length range of `[ngram_min,ngram_max]`. Outputs metrics. 

#### OR

`python -m src.naive_bayes_gridsearch [distribution_type] [data_file]`

Performs a grid search over Laplace smoothing parameters on naive bayes models of the specified distribution type.  Outputs parameter search results and the metrics of the best performing model. 

### Support Vector Machine

`python -m src.svm [distribution_type] [kernel] [c0] [c1] [data_file]`

where `kernel` is the choice of kernel (can be `linear`, `poly`, `rbf`, `sigmoid`). `c0` and `c1` are the class weights for the negative and positive class, respectively. 

Trains a single SVM model and outputs metrics. 

#### OR

`python -m src.svm_parameter_tuning [distribution_type] [kernel] [data_file]`

where `kernel` is the choice of kernel (can be `linear`, `poly`, `rbf`, `sigmoid`). 

Performs a grid search over class weights on SVM models of the specified parameters. Outputs parameter search results and the metrics of the best performing model. 

### Tree Learning - Random Forest

`python -m src.random_forest [max_tree_depth] [num_trees] [distribution_type] [data_file]`

where `max_tree_depth` is the maximum depth of each tree and `num_trees` is the number of decision tree classifiers in the ensemble. 

Trains a single Random Forest model using the given parameters. Outputs metrics. 

#### OR 

`python -m src.random_forest_gridsearch [num_trees] [distribution_type] [data_file]`

where `num_trees` is the number of decision tree classifiers in the ensemble. 

Performs a grid search over the `max_tree_depth` hyperparameter. Outputs parameter search results and the metrics of the best performing model. 

## Running pre-trained models
Pretrained models are stored as pickle files in the `model_pickles/` directory. The command for running a pretrained model is as follows:

`python -m src.pretrained_model [distribution_type] [pickle_file_name] [data_file]`

For pickle files that specify the `word2vec` distribution, please use the `punct_stopwords_removed.csv` data file. For remaining models, please use the `punct_stopwords_removed_lemmatized.csv` data file. Reasoning for these data distributions is described in the report. 

Example pre-trained model execution:

`python -m src.pretrained_model multinomial naive_bayes_multinomial_1_1_0.57.sav punct_stopwords_removed_lemmatized.csv `