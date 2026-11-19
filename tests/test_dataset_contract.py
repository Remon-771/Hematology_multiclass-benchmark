import pandas as pd
import pytest

from cbcbench.config import DATA_PATH, DATA_SHA256, N_RAW_COLUMNS, N_RECORDS, TARGET_COL
from cbcbench.data import identifier_columns, high_missing_columns, load_raw, sha256

CLASS_COUNTS = {"Microcytic Hypochromic Anaemia": 2994, "Normal / Non-specific": 2541, "Mild-Moderate Leukocytosis": 2501,
                "Thrombocytopaenia": 2468, "Severe Leukocytosis Flag": 2114}
FEATURES = ['WBC', 'Neutrophil count', 'Lymphocyte count', 'Monocyte count', 'Eosinophil count', 'Basophil count',
            'Neutrophil %', 'Lymphocyte %', 'Monocyte %', 'Eosinophil %', 'Basophil %', 'RBC', 'Haemoglobin', 'HCT', 'MCV',
            'MCH', 'MCHC', 'RDW', 'Platelet', 'MPV', 'PCT', 'PDW', 'ESR']


def test_checksum():
    assert sha256(DATA_PATH) == DATA_SHA256


def test_source_headers_preserved():
    raw = pd.read_csv(DATA_PATH)
    assert raw.shape == (N_RECORDS, N_RAW_COLUMNS)
    assert "platelat" in raw.columns and "Lymphoeyte %" in raw.columns  # original headers kept in the file


def test_standardised_schema():
    df = load_raw()
    assert df.drop(columns=[TARGET_COL]).columns.tolist() == FEATURES
    assert df[TARGET_COL].value_counts().to_dict() == CLASS_COUNTS
    assert not df.duplicated().any()
    assert identifier_columns(df) == [] and high_missing_columns(df) == []


def test_missingness_total():
    df = load_raw().drop(columns=[TARGET_COL])
    assert int(df.isna().sum().sum()) == 18247
