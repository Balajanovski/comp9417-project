# comp9417-project
Project created for comp9417

## Pre-processing
Ensure that the file `data/train.csv` exists (too big to upload)
Run the file `src/preprocess.py` (takes around 30min on CPU)

## Models
All models should be run from the root directory

### Naive Bayes

`python -m src.naive_bayes [distribution_type] [ngram_min] [ngram_max] [data_file]`

where `distribution_type` is either `bernouli` or `multinomial`, `ngram_min` and `ngram_max` are the smallest and largest n-grams to consider, respectively

eg `python -m src.naive_bayes bernouli 1 3 punct_removed.csv`

#### OR

`python -m src.naive_bayes [distribution_type] [ngram_max]`

to print out a plot for all ngram numbers up to `ngram_max`

### Support Vector Machine

`python -m src.svm [data_distribution] [kernel] [data_file]`

where `data_distribution` is either `bernouli` or `multinomial` for bag-of-words or `word2vec` for word vectors. `kernel` is the choice of kernel (can be `linear`, `poly`, `rbf`, `sigmoid`)

Note: non-linear kernels may take up to 20min on CPU to train and predict

### Tree Learning - Random Forest

`python -m src.random_forest [max_tree_depth] [num_trees] [data_distribution] [data_file]`

where `max_tree_depth` is the maximum depth of the tree and `num_trees` is the number of models in the ensemble. `data_distribution` is either `bernouli` for bag-of-words or `word2vec` for word vectors. 