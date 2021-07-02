# comp9417-project
Project created for comp9417

## Pre-processing
Ensure that the file `data/train.csv` exists (too big to upload)
Run the file `src/preprocess.py` (takes around 30min on CPU)

## Models
All models should be run from the root directory

### Naive Bayes

`python -m src.naive_bayes [distribution_type] [ngram_min] [ngram_max]`

where `distribution_type` is either `bernouli` or `multinomial`, `ngram_min` and `ngram_max` are the smallest and largest n-grams to consider, respectively

eg `python -m src.naive_bayes bernouli 1 3`

#### OR

`python -m src.naive_bayes [distribution_type] [ngram_max]`

to print out a plot for all ngram numbers up to `ngram_max`

### Support Vector Machine

`python -m src.svm [distribution_type] [kernel]`

where `distribution_type` is either `bernouli` or `multinomial`, `kernel` is the choice of kernel (can be `linear`, `poly`, `rbf`, `sigmoid`)

Note: non-linear kernels may take up to 20min on CPU to train and predict