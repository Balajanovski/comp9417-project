# comp9417-project
Project created for comp9417

## Pre-processing
Ensure that the file `data/train.csv` exists (too big to upload)
Run the file `src/preprocess.py` (takes around 30min on CPU)

## Models
### Naive Bayes

From the root directory

`PYTHONPATH=. python -m src.naive_bayes [distribution_type] [ngram_min] [ngram_max]`

where `distribution_type` is either `bernouli` or `multinomial`, `ngram_min` and `ngram_max` are the smallest and largest n-grams to consider, respectively

eg `python -m src.naive_bayes bernouli 1 3`