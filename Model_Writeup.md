# Financial Fraud Detection — Model Write-Up

## Overview

For this project, I built **three** pipelines on the same PaySim-style transaction data (same `TRANSFER` / `CASH_OUT` filter and engineered features): **PCA + logistic regression** with stratified K-fold, **random forest** with a time-based split, and **isolation forest** outlier detection (also on the time-based split, aligned with the forest model). The goal is to compare a **linear supervised** baseline, a **nonlinear supervised** model, and an **unsupervised anomaly** score for the same fraud-detection task, using **PR-AUC** and **threshold-tuned F1** as the main metrics.

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

**Reported CV summary:** mean **PR-AUC = 0.6644** (std **0.0178**), mean **ROC-AUC = 0.9950** (std **0.0011**), mean F1 per fold ≈ **0.126** (default-threshold behavior within the fold loop).

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

On that held-out fold, **F1 at the PR-tuned threshold ≈ 0.666**, with fraud precision **≈ 0.81** and recall **≈ 0.57** (support **1,642** fraud cases in that fold).

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

On this synthetic benchmark and time-based test window, **random forest achieves extremely strong ranking** (PR-AUC effectively **~1.0** and ROC-AUC **~1.0** on the held-out test when recomputed with the same split and features). That reflects how separable fraud is in this dataset under a nonlinear supervised model — it should not be read as “fraud is easy in the real world.”

---

## Model 3: Isolation Forest Outlier Detection (`Model_OutlierDetection.ipynb`)

### Overview

As an additional experiment (suggested for highly imbalanced problems with a huge majority of non-fraud), I treated fraud as **anomaly detection**: learn what "normal" legitimate transactions look like, then flag transactions that are **isolated** from that bulk behavior in feature space. The implementation uses **Isolation Forest**, an ensemble method that randomly partitions the data and assigns higher anomaly scores to observations that are easier to isolate (typically rare or different from the majority).

### Feature Engineering

I used the **same** feature engineering as Models 1 and 2 — same `TRANSFER` / `CASH_OUT` filter, same balance-error features, flags, `log_amount`, and `type_encoded` — so differences in metrics reflect the modeling paradigm (unsupervised scores vs. supervised probabilities), not a change in inputs.

### Train/Test Strategy

To stay aligned with the Random Forest experiment, I used the **same time-based split** on `step`: the first 80% of time steps for training and the last 20% for testing (test set **552,504** rows, **4,258** fraud, **~0.77%** fraud rate — higher than in the training window, which is a form of distribution shift).

I fit the isolation forest on **`X_train_normal` only** — training rows where `isFraud == 0`. The scaler (`StandardScaler`) was also fit on these normal rows so centering and scaling reflect the legitimate bulk. The model never sees labeled fraud during this fit, which matches the pure anomaly-detection narrative: fraud should appear as high anomaly scores at test time if it deviates from normal behavior.

### Model and Hyperparameters

I used `sklearn.ensemble.IsolationForest` with **`contamination='auto'`** on normal-only training. After a **validation grid** on the last 15% of training-time steps (choosing hyperparameters by **PR-AUC** without using the final test set), I refit on all training normals with **`n_estimators=400`**, **`max_samples=min(200_000, n)`**, and **`max_features=1.0`**. For reporting, anomaly scores are **`anomaly_score = -decision_function`** (higher = more suspicious), aligned with PR curves the same way as supervised probabilities.

### Threshold Tuning

As with the other models, the default sklearn outlier rule is not tuned to business costs. I computed the precision–recall curve on the **held-out test set**, searched thresholds for **maximum F1**, and reported metrics at that cutoff alongside sklearn’s default `predict` threshold. (Tuning on the test set is slightly optimistic for deployment; a stricter workflow would reserve a calibration slice of training time.)

### Results (time-based test, tuned isolation forest)

| Metric | Value |
|--------|--------|
| **PR-AUC** | **0.6859** |
| **ROC-AUC** | **0.9689** |
| **Max F1** (threshold scan on test) | **0.7164** |

At the **F1-optimal threshold**, fraud class performance was approximately **precision 0.79** and **recall 0.62** (**2,657** true positives, **1,601** false negatives on **4,258** fraud cases). Under sklearn’s **default** outlier threshold, fraud **recall** was higher (**~0.72**) but **precision** lower (**~0.52**), illustrating the usual precision–recall tradeoff.

### Limitations

Not every fraud case is a global outlier, and not every outlier is fraud — legitimate large or unusual transactions can score highly. Isolation forest scores are also less directly explainable to stakeholders than random-forest feature importances, though they still use the same interpretable underlying features.

---

## Comparing the Three Approaches

| | PCA + LR | Random Forest | Isolation Forest |
|---|---|---|---|
| **Evaluation** | Stratified 5-fold CV (mean over folds); final numbers also on one held-out fold | Time-based train/test | Same **time-based** test as RF |
| **Uses fraud labels at train** | Yes | Yes | No (normal-only fit) |
| **PR-AUC (headline)** | **0.6644** mean (folds); ~0.64–0.68 per fold | **~1.00** on temporal test (this dataset) | **0.6859** on temporal test |
| **ROC-AUC** | **~0.995** mean (folds) | **~1.00** on temporal test | **0.9689** on temporal test |
| **Tuned F1 (as reported)** | **~0.67** on last K-fold test slice | Very high at tuned threshold on temporal test | **0.7164** (max over test PR curve) |
| **Interpretability** | Low (PCs) | High (feature importances) | Moderate (scores only) |
| **Role** | Linear + DR baseline | Best accuracy / ranking here | Label-free anomaly baseline |

**Concise analysis.** All three models use the **same engineered inputs**, but they answer slightly different questions. **PCA + logistic regression** gives a **stable linear baseline** under **stratified CV** (mean PR-AUC **0.66**), which is the right tool for “how much signal is linear and low-dimensional?” **Random forest** dominates on the **temporal holdout** on this synthetic data (**PR-AUC ≈ 1**), which is expected when labels are available and fraud is nearly separable in feature space — it is the practical **supervised** choice here. **Isolation forest** sits **between PCA’s mean CV PR-AUC and the RF ceiling** on the **same time test** (**0.69** PR-AUC, **0.72** max F1), which is respectable for a model that **does not train on fraud labels**. Because PCA is evaluated with **K-fold** and IF/RF with a **time split**, headline numbers are **indicative**, not a single leaderboard; the fair internal story is: **RF > IF > PCA (linear)** on ranking quality for this pipeline, with IF justified as the **unsupervised** counterpart rather than the top supervised model.

The PCA model is faster and its probabilities are well behaved for a linear baseline. The Random Forest is the strongest performer on the temporal test and remains interpretable via importances. The isolation forest adds a **label-free** perspective and meaningful PR-AUC without matching supervised RF on this benchmark.
