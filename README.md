# Anaphora Resolution
- feature.py extract features according to Salience factors in Lappin & Leass:
  * Sentence recency
  * Subject emphasis
  * Existential emphasis
  * Direct object emphasis
  * Indirect and oblique argument emphasis
  * Non-adverbial emphasis
  * Head noun emphasis
- CatBoostClassifier.ipynb train and test with CatBoost Classifier.
- GradientBoostingClassifier.ipynb train and test with Gradient Boosting Classifier.
- RandomForestClassifier.ipynb train and test with Random Forest Classifier and Extra Trees Classifier

### Dataset
GAP is a gender-balanced dataset containing 8,908 coreference-labeled pairs of (ambiguous pronoun, antecedent name), sampled from Wikipedia and released by Google AI Language for the evaluation of coreference resolution in practical applications.

### Performance
| Model | F1 score |
| --- | ----------- |
| CatBoost Classifier | 74.7% |
| Gradient Boosting Classifier | 74.1% |
| Best model in [Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns](https://arxiv.org/abs/1810.05201) | 66.9% |
