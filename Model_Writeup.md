# Financial Fraud Detection — Model Write-Up

## Overview

For this project, I built two separate machine learning pipelines to detect fraudulent financial transactions using a dataset of over 6.3 million records. Each model takes a different approach to the same problem, allowing for a meaningful comparison of methods. Below I walk through exactly what I did in each file, from data preparation through final evaluation.

---

## Model 1: PCA + K-Fold Cross Validation + Logistic Regression (`Model_PCA_KFold.ipynb`)

### Feature Engineering

The first thing I did was filter the dataset down to only `TRANSFER` and `CASH_OUT` transaction types, since fraud exclusively occurs in those two categories. This cut the dataset nearly in half and removed guaranteed non-fraud records, making the class imbalance slightly less extreme.

From there, I engineered several new features designed to capture suspicious behavior:

- **`errorBalanceOrig`** — calculated as `newbalanceOrig + amount - oldbalanceOrg`. In a legitimate transaction, this should equal zero. A non-zero value means the sender's balance did not decrease by the expected amount, which is a strong fraud signal.
- **`errorBalanceDest`** — same concept for the receiver: `oldbalanceDest + amount - newbalanceDest`. If money moved but the destination balance did not increase, something is off.
- **`origDrained`** — a binary flag set to 1 when the sender's balance starts positive and ends at exactly zero. This "account drain" pattern is a common fraud behavior.
- **`destUnchanged`** — a binary flag set to 1 when the receiver's balance is the same before and after the transaction, despite funds being sent.
- **`log_amount`** — a log-transformed version of the transaction amount to reduce the extreme right skew (transactions range from near zero to $92 million).
- **`type_encoded`** — a binary encoding of transaction type: 1 for TRANSFER, 0 for CASH_OUT.

### PCA (Principal Component Analysis)

Before applying PCA, I scaled all features using `StandardScaler`, since PCA is sensitive to feature magnitudes and features like `oldbalanceOrg` and `amount` operate on very different scales.

I then fit a full PCA on a 100,000-row stratified sample of the data to examine how many components were needed. I plotted both the individual explained variance per component and the cumulative explained variance, with a red dashed line marking the 95% threshold. This told me exactly how many components to retain without throwing away meaningful information.

I also generated a **component loadings heatmap** — a grid showing how much each original feature contributes to each principal component. This helped me understand what each PC is actually capturing, even though the components themselves are abstract linear combinations of the original features.

### K-Fold Cross Validation

I used **Stratified K-Fold** with 5 splits instead of regular K-Fold. This is important because the fraud class makes up only 0.13% of the data — without stratification, some folds could end up with very few or no fraud cases at all, which would make the evaluation unreliable.

The entire pipeline — `StandardScaler → PCA → LogisticRegression` — was fit inside each fold on the training portion only. This prevents data leakage: the scaler and PCA never see the validation data during fitting, so the evaluation is honest.

For each fold I recorded three metrics: F1 score, PR-AUC (Precision-Recall AUC), and ROC-AUC. I then printed the per-fold results and computed the mean and standard deviation across all five folds. A bar chart visualized how consistent the model was across folds.

### Training

The classifier I chose was **Logistic Regression**, applied to the PCA-transformed features. I set `class_weight='balanced'` so that the model penalizes misclassifying fraud more heavily than legitimate transactions — without this, the model would learn to just predict "not fraud" on everything and still achieve 99.87% accuracy. I set `max_iter=1000` to ensure convergence and used the `lbfgs` solver, which works well for medium-sized datasets.

### Hyperparameter Tuning

The primary tuning decision was **how many PCA components to retain**. Rather than arbitrarily picking a number, I derived this directly from the cumulative explained variance plot, selecting the minimum number of components needed to explain 95% of the total variance. This balances dimensionality reduction against information loss.

The second tuning step was **classification threshold selection**. Logistic Regression outputs a probability, and by default the threshold to classify a transaction as fraud is 0.5. With only 0.13% fraud in the data, this default is far too high — the model will almost never predict fraud. Instead, I computed the full Precision-Recall curve and found the threshold that maximized the F1 score. I marked this point on the curve plot and re-evaluated the model at that threshold.

### Evaluation

I evaluated the final model on a held-out test fold with:
- **Classification report** at both the default and optimal thresholds, showing precision, recall, and F1 for each class
- **Confusion matrix** showing true positives, false negatives, false positives, and true negatives
- **Precision-Recall curve** with the optimal threshold marked, and the PR-AUC score as the headline metric

I chose PR-AUC as the primary metric rather than ROC-AUC or accuracy because it is more informative when the positive class (fraud) is rare. A model that flags everything as fraud would score well on ROC-AUC but would have terrible precision, and PR-AUC captures that.

---

## Model 2: Random Forest (`Model_RandomForest.ipynb`)

### Feature Engineering

I used the same feature engineering as Model 1 — the same filter, the same balance error calculations, the same binary flags, and the same log-transform on amount — so that any performance difference between the two models reflects the modeling approach rather than differences in input data.

### Train/Test Split Strategy

Instead of K-Fold cross validation, I used a **time-based split** on the `step` column. The dataset spans 743 time steps (roughly hours), and I used the first 80% of steps as the training set and the remaining 20% as the test set.

This is a deliberate choice: in a real fraud detection system, the model is always predicting on future transactions it has never seen. A random split would allow the model to train on data from later time periods and test on earlier ones, which is unrealistic and inflates performance estimates. The time-based split enforces a clean temporal boundary.

### Training

I trained a `RandomForestClassifier` with the following configuration:

- **`n_estimators=100`** — 100 decision trees in the ensemble
- **`max_depth=20`** — trees are allowed to grow deep to capture complex patterns, but bounded to reduce overfitting
- **`min_samples_leaf=10`** — each leaf must contain at least 10 samples, preventing the model from memorizing individual transactions
- **`class_weight='balanced_subsample'`** — each tree's bootstrap sample is independently reweighted so that fraud cases count more. This is different from `'balanced'`, which uses the full dataset ratios. The `balanced_subsample` approach is generally better for Random Forests because it introduces diversity between trees in how they handle the imbalance
- **`n_jobs=-1`** — uses all available CPU cores to parallelize training

### Hyperparameter Tuning

The main tuning decisions were the tree depth (`max_depth=20`) and minimum leaf size (`min_samples_leaf=10`), both of which control how much the model can overfit to training data. Beyond the model hyperparameters, I also tuned the **classification threshold** using the same PR-curve F1-maximization method as in Model 1 — finding the probability cutoff that best balances catching fraud (recall) against avoiding false alarms (precision).

### Feature Importances

A major advantage of Random Forest over the PCA model is that it provides **Gini-based feature importance scores** — a measure of how much each original feature contributed to the model's decisions across all trees. I plotted these as a horizontal bar chart, with bars above the median importance highlighted in red. This makes it easy to see which features the model found most predictive of fraud.

This interpretability is a significant practical advantage: I can tell a business stakeholder that "the destination balance discrepancy is the strongest fraud signal" in plain terms, whereas the PCA model can only say "principal component 2 was most important," which is not directly meaningful.

### Evaluation

I evaluated the model with the same set of metrics as Model 1 for a direct comparison:

- **Classification report** at both the default and optimal thresholds
- **Confusion matrix** showing the breakdown of correct and incorrect predictions
- **Precision-Recall curve** with the optimal F1 threshold marked
- **Probability distribution plots** — histograms of the predicted fraud probability for legitimate and fraudulent transactions side by side, with a zoomed-in view near the threshold. A clean separation between the two distributions indicates a confident model.

---

## Comparing the Two Approaches

| | PCA + Logistic Regression | Random Forest |
|---|---|---|
| Split strategy | Stratified K-Fold (5 splits) | Time-based (temporal integrity) |
| Dimensionality | Reduced via PCA | All original features |
| Captures non-linear patterns | No | Yes |
| Interpretability | Low (abstract PCs) | High (feature importances) |
| Imbalance handling | `class_weight='balanced'` | `class_weight='balanced_subsample'` |
| Key metric | PR-AUC | PR-AUC |

The PCA model is faster and its probabilities are better calibrated, making it useful as a baseline and for understanding how much of the signal is captured by linear combinations of the features alone. The Random Forest model is more powerful, can learn interactions between features (e.g., fraud when *both* the balance error is high *and* the account was drained), and produces interpretable feature rankings — which makes it the more practical choice for a real deployment.
