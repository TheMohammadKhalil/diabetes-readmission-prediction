import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def make_preprocessor(dense=False):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense,
                                 dtype=np.float32)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], sparse_threshold=0 if dense else 1)


def model_specs(seed=42, quick=False):
    specs = {
        "majority_baseline": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=1000, random_state=seed),
            {"model__C": [0.1, 1.0]},
        ),
        "logistic_balanced": (
            LogisticRegression(solver="liblinear", max_iter=1000,
                               class_weight="balanced", random_state=seed),
            {"model__C": [0.1, 1.0]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(min_samples_leaf=40, class_weight="balanced",
                                   random_state=seed),
            {"model__max_depth": [4, 8]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
                                   class_weight="balanced_subsample", n_jobs=1,
                                   random_state=seed),
            {"model__max_depth": [10, 18]},
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08,
                                           l2_regularization=1.0,
                                           early_stopping=False, random_state=seed),
            {"model__max_leaf_nodes": [15, 31]},
        ),
        "neural_network": (
            MLPClassifier(hidden_layer_sizes=(32,), alpha=0.01, batch_size=512,
                          learning_rate_init=0.001, max_iter=100, tol=0.001,
                          n_iter_no_change=10, early_stopping=False, random_state=seed),
            {},
        ),
    }
    result = {}
    for name, (estimator, grid) in specs.items():
        if quick:
            grid = {key: values[:1] for key, values in grid.items()}
            if name == "random_forest":
                estimator.set_params(n_estimators=30)
            elif name == "gradient_boosting":
                estimator.set_params(max_iter=40)
            elif name == "neural_network":
                estimator.set_params(max_iter=40)
        pipeline = Pipeline([
            ("preprocess", make_preprocessor(dense=name == "gradient_boosting")),
            ("model", estimator),
        ])
        result[name] = (pipeline, grid)
    return result
