# Results

## Baseline classifiers

| Classifier | Settings | 20news train | 20news test | religion test |
|----|-----|------|-------|------|
| NaiveBayes | `MultinomialNB()` | 95.8 | 85.2 | 50.4 |
| SGDClassifier | `SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=123)` | 99.9 | 93.4 | 54.0 |
| SVM RBF | `SVC(C=1000000, kernel='rbf')` | 100.0 | 93.9 | 56.8

(Note: just ran once for each, using seed 123 for shuffle; C for SVM RBF was guessed from 0, 1000000, no grid search yet)
