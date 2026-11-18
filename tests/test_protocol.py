import numpy as np
import pandas as pd
import pytest

from readmission.data import (
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, clean_cohort, diagnosis_group,
    load_data, make_features, make_splits,
)
from readmission.evaluation import choose_threshold, compute_metrics
from readmission.models import make_preprocessor


@pytest.fixture
def raw_records():
    records = pd.DataFrame({col: [1] * 30 for col in NUMERIC_COLUMNS})
    for col in CATEGORICAL_COLUMNS:
        records[col] = "No"
    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        records[col] = 1
    records["encounter_id"] = np.arange(30)
    records["patient_nbr"] = np.repeat(np.arange(15), 2)
    records["readmitted"] = np.tile(["NO", "<30", ">30"], 10)
    records["A1Cresult"] = "None"
    records["max_glu_serum"] = "None"
    records["diag_1"] = "250.13"
    return records


def test_none_lab_category_and_missing_marker_are_distinct(raw_records, tmp_path):
    raw_records.loc[0, "race"] = "?"
    path = tmp_path / "source.csv"
    raw_records.to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded.A1Cresult.eq("None").all()
    assert pd.isna(loaded.loc[0, "race"])


def test_invalid_target_is_rejected(raw_records, tmp_path):
    raw_records.loc[0, "readmitted"] = "UNKNOWN"
    path = tmp_path / "source.csv"
    raw_records.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unexpected"):
        load_data(path)


def test_death_hospice_exclusions(raw_records):
    raw_records.loc[:5, "discharge_disposition_id"] = [11, 13, 14, 19, 20, 21]
    assert len(clean_cohort(raw_records)) == 24


@pytest.mark.parametrize("code,expected", [
    ("250.13", "Diabetes"), ("459.9", "Circulatory"), ("785.1", "Circulatory"),
    ("786", "Respiratory"), ("788.2", "Genitourinary"), ("239.9", "Neoplasms"),
    ("E879", "Other"), ("V45", "Other"), (np.nan, "Missing"),
])
def test_diagnosis_boundaries(code, expected):
    assert diagnosis_group(code) == expected


def test_features_exclude_identifiers_and_target(raw_records):
    X = make_features(raw_records)
    assert not {"encounter_id", "patient_nbr", "readmitted", "weight", "examide", "citoglipton"} & set(X)
    assert X.admission_type_id.dtype == object
    assert X.total_prior_visits.eq(3).all()
    modified = raw_records.copy()
    modified["readmitted"] = "<30"
    pd.testing.assert_frame_equal(X, make_features(modified))


def test_repeated_patients_never_cross_split_or_fold():
    groups = pd.Series(np.repeat(np.arange(500), 3))
    y = pd.Series(np.random.default_rng(7).binomial(1, 0.15, len(groups)))
    X = pd.DataFrame({"x": np.arange(len(groups))})
    train, test, cv = make_splits(X, y, groups)
    assert not set(groups.iloc[train]) & set(groups.iloc[test])
    assert 0.15 < len(test) / len(X) < 0.25
    visited = np.zeros(len(train), dtype=int)
    for fit, val in cv:
        assert not set(groups.iloc[train].iloc[fit]) & set(groups.iloc[train].iloc[val])
        visited[val] += 1
    assert np.all(visited == 1)


def test_preprocessing_does_not_learn_validation_values():
    train = pd.DataFrame({"count": [0., 2., np.nan], "category": ["a", "b", np.nan]})
    test = pd.DataFrame({"count": [1000.], "category": ["test-only"]})
    processor = make_preprocessor()
    processor.fit(train)
    processor.transform(test)
    scaler = processor.named_transformers_["numeric"].named_steps["scale"]
    encoder = processor.named_transformers_["categorical"].named_steps["encode"]
    assert scaler.mean_[0] == 1
    assert "test-only" not in encoder.categories_[0]


def test_threshold_improves_f1_without_test_labels():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.02, 0.1, 0.25, 0.4])
    threshold = choose_threshold(y, p)
    assert threshold == 0.25
    assert compute_metrics(y, p, threshold)["f1"] == 1
    assert compute_metrics(y, p, 0.5)["recall"] == 0


def test_majority_baseline_can_have_high_accuracy_and_zero_recall():
    y = np.array([0] * 89 + [1] * 11)
    metrics = compute_metrics(y, np.full(100, 0.11))
    assert metrics["accuracy"] == 0.89
    assert metrics["recall"] == 0
    assert metrics["roc_auc"] == 0.5
    assert metrics["average_precision"] == 0.11
