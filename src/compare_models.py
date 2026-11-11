"""
Compare the performance of different models using a corrected paired t-test.
"""
from itertools import combinations
import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/adult_train/runs.csv")
runs = runs[(runs["data"] == "data/adult_train.csv") & (runs["metric"] == "roc_auc")]

folds = [f"fold_{i}" for i in range(1, 6)]

for (_, model_a), (_, model_b) in combinations(runs.iterrows(), 2):
    result = corrected_paired_test(
        model_a[folds].to_numpy(),
        model_b[folds].to_numpy(),
        test_train_ratio=1 / 4,  # 5-fold CV: validation/training ratio
    )

    print(
        f"{model_a['model']} vs {model_b['model']}: "
        f"mean ROC-AUC difference = {result['mean_diff']:.3f}, "
        f"fold wins = {result['wins']}, "
        f"p-value = {result['p_value']:.4f}"
    )