import argparse
import os
import pathlib
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

DATA_PATH = pathlib.Path("data/processed")
DATABASES_PATH = pathlib.Path("outputs/databases")

OUTPUT_PATH = pathlib.Path("outputs/plots/featurized_simulation")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = general_utils.open_pickle(DATABASES_PATH / "df4_features_merge_without_nans.pickle")

df = df[df["n_group_members"] > 1]

# create new features
df["proportion_guardians_interacted_daily"] = df["n_interacting_guardians_daily"] / df["n_group_members"]
df["Preprimary_class"] = (df["P1"] + df["P2"] > 0).astype(int)
df["Primary_class"] = (df["C1"] + df["C2"] + df["C3"] > 0).astype(int)
df["Primary_Activity"] = (
    df["Primary I (6-7 years)"] + df["Both(Primary I (6-7 years) and Primary II (7-8 years))"] > 0
).astype(int)
df["Preprimary_Activity"] = (
    df["Both(Pre-Primary I (3-4 years) and Pre-Primary II (5-6 years))"]
    + df["Pre-Primary I (3-4 years)"]
    + df["Pre-Primary II (5-6 years)"]
    > 0
).astype(int)
df["Activity_matching_class"] = (
    ((df["Primary_Activity"] == 1) & (df["Primary_class"] == 1))
    | ((df["Preprimary_Activity"] == 1) & (df["Preprimary_class"] == 1))
).astype(int)

drop = [
    "guardian_id",
    "group_id",
    "n_males",
    "n_females",
    "day",
    "DA",
    "hi",
    "Activity",
    "OtherResponseType",
    "text",
    "ModMessageTypeSticker",
    "0-2",
]
df = df.drop(columns=drop)

# df["n_group_members_sq"] =df["n_group_members"]**2

y = df["guardian_interacted"]
drop = [
    "guardian_interacted",
    "mr",
    "n_distinct_moderators_daily",
    "week_cumulative_n_guardian_messages",
    "month_cumulative_n_guardian_messages",
    "Easy",
    "ModMessageTypeChat",
    "weekly_guardian_interaction_indicator",
    "monthly_guardian_interaction_indicator",
    "n_individual_guardian_interactions_daily",
    "monthly_cumulative_guardian_n_days_interacted",
]
drop2 = [
    "P1",
    "P2",
    "C1",
    "C2",
    "C3",
    "Primary I (6-7 years)",
    "Both(Primary I (6-7 years) and Primary II (7-8 years))",
    "Both(Pre-Primary I (3-4 years) and Pre-Primary II (5-6 years))",
    "Pre-Primary I (3-4 years)",
    "Pre-Primary II (5-6 years)",
]

X = df.drop(columns=drop)
X = X.drop(columns=drop2)

model = LogisticRegression(solver="liblinear", random_state=0)
model.fit(X, y)

log_reg = sm.Logit(y, X).fit_regularized(method="l1")
print(log_reg.summary())
breakpoint()

import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
predicted_y = model.predict(X_test)
print(metrics.classification_report(y_test, predicted_y))
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)


def feat_imp(df, model):

    d = dict(zip(df.columns, model.feature_importances_))
    ss = sorted(d, key=d.get, reverse=True)
    n_features = len(ss)
    top_names = ss[0:]

    plt.figure(figsize=(25, 15))
    plt.title("Feature importances")
    plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
    # plt.xlim(-1, n_features)
    plt.xticks(range(n_features), top_names, rotation="vertical")
    plt.savefig("xgb_feature_importance.png")
    plt.tight_layout()

    plt.close()


feat_imp(X_train, model)
breakpoint()

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
x = X.iloc[:, :]
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

breakpoint()
