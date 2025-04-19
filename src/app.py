
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
from pickle import dump
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Datasets sin feature selection

BASE_PATH = "../data/processed"
TRAIN_PATHS = [
    "X_train_con_outliers.xlsx",
    "X_train_sin_outliers.xlsx",
]
TRAIN_DATASETS = []
for path in TRAIN_PATHS:
    TRAIN_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

TEST_PATHS = [
    "X_test_con_outliers.xlsx",
    "X_test_sin_outliers.xlsx",
]
TEST_DATASETS = []
for path in TEST_PATHS:
    TEST_DATASETS.append(
        pd.read_excel(f"{BASE_PATH}/{path}")
    )

y_train = pd.read_excel(f"{BASE_PATH}/y_train.xlsx")
y_test = pd.read_excel(f"{BASE_PATH}/y_test.xlsx")

results = []
models=[]

for index, dataset in enumerate(TRAIN_DATASETS):
    model = XGBClassifier(random_state = 42)
    model.fit(dataset, y_train)
    models.append(model)
    
    y_pred_train = model.predict(dataset)
    y_pred_test = model.predict(TEST_DATASETS[index])

    results.append(
        {
            "train": accuracy_score(y_train, y_pred_train),
            "test": accuracy_score(y_test, y_pred_test)
        }
    )

# Con feature selection

train_data = pd.read_csv("../data/processed/clean_train_con_outliers.csv")
test_data = pd.read_csv("../data/processed/clean_test_con_outliers.csv")

X_train = train_data.drop(["Outcome"], axis = 1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis = 1)
y_test = test_data["Outcome"]

model = XGBClassifier(random_state = 42)
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

## Hiper-parametrización

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'reg_lambda': [0, 1, 10],
    'reg_alpha': [0, 0.1, 1]
}

grid = GridSearchCV(model, param_grid, scoring = "accuracy", cv = 5, njobs=1)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grid.fit(TRAIN_DATASETS[1], y_train)

print(f"Mejores hiperparámetros: {grid.best_params_}")


model = XGBClassifier(
    max_depth=3,             
    min_child_weight=3,      
    subsample=1,
    colsample_bytree=1,
    n_estimators=50,         
    learning_rate=0.1,      
    reg_alpha=1,
    reg_lambda=0,
    random_state=42,
    eval_metric='logloss',
)

model.fit(TRAIN_DATASETS[1], y_train)

y_pred_train = model.predict(TRAIN_DATASETS[1])
y_pred_test = model.predict(TEST_DATASETS[1])

print(f"Train: {accuracy_score(y_train, y_pred_train)}")
print(f"Test: {accuracy_score(y_test, y_pred_test)}")

## Guardando el modelo
dump(model, open("../models/boosting_classifier_sin_outliers_42.sav", "wb"))