# =============================================================
# OCT Early Warning System
# src/data_manager.py
#
# Purpose: Load and standardise input data from either:
#   A) Separate .npy files (dissertation format)
#   B) A single CSV file (future/combined format)
#
# Outputs a clean standardised DataFrame used by all
# subsequent modules.
# =============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union

from config.defaults import (
    COL_SCAN_ID,
    COL_PATIENT_ID,
    COL_VISIT_ID,
    COL_VISIT_DATE,
    COL_TRUE_LABEL,
    COL_PREDICTED_LABEL,
    EMBEDDING_COL_PREFIX,
    PROB_COL_PREFIX,
    DEFAULT_CLASSES,
    NORMAL_CLASS,
)


# -------------------------------------------------------------
# CLASS LABEL MAPPER
# Converts integer class indices to string class names.
# Handles the dissertation format where:
#   CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
#   NORMAL_CLASS_IDX = 3
# -------------------------------------------------------------

class ClassLabelMapper:
    """
    Maps between integer class indices and string class names.

    The dissertation stores labels as integers.
    The EWS system works with string names throughout.
    This class handles the conversion cleanly.

    Args:
        class_names: ordered list of class names matching index order
                     e.g. ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.idx_to_name = {i: name for i, name in enumerate(class_names)}
        self.name_to_idx = {name: i for i, name in enumerate(class_names)}

    def to_names(self, indices: np.ndarray) -> np.ndarray:
        """Convert integer array to string name array."""
        return np.array([self.idx_to_name[i] for i in indices])

    def to_indices(self, names: np.ndarray) -> np.ndarray:
        """Convert string name array to integer array."""
        return np.array([self.name_to_idx[n] for n in names])

    def normal_idx(self) -> int:
        """Return the integer index of the NORMAL class."""
        return self.name_to_idx[NORMAL_CLASS]

    def prob_col_names(self) -> List[str]:
        """Return standardised probability column names."""
        return [f"{PROB_COL_PREFIX}{name}" for name in self.class_names]

    def emb_col_names(self, n_dims: int) -> List[str]:
        """Return standardised embedding column names."""
        return [f"{EMBEDDING_COL_PREFIX}{i}" for i in range(n_dims)]


# -------------------------------------------------------------
# NPY LOADER
# Loads dissertation-format .npy files and combines them
# into one standardised DataFrame.
# -------------------------------------------------------------

class NpyLoader:
    """
    Loads separate .npy files from the dissertation format
    and combines them into a standardised DataFrame.

    Dissertation file structure:
        train_embeddings.npy  shape: (n_samples, 2048)
        train_labels.npy      shape: (n_samples,)  integers
        val_embeddings.npy
        val_labels.npy
        val_predictions.npy   shape: (n_samples,)  integers
        val_probabilities.npy shape: (n_samples, 4)
        test_embeddings.npy
        test_labels.npy
        test_predictions.npy
        test_probabilities.npy

    Args:
        embeddings_root: path to the folder containing .npy files
        class_names: ordered list matching the integer label encoding
                     dissertation uses ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    """

    def __init__(
        self,
        embeddings_root: Union[str, Path],
        class_names: List[str] = None,
    ):
        self.root = Path(embeddings_root)
        self.class_names = class_names or ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        self.mapper = ClassLabelMapper(self.class_names)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Embeddings root not found: {self.root}"
            )

    def _load_split(
        self,
        split: str,
        has_predictions: bool = True,
        has_probabilities: bool = True,
    ) -> pd.DataFrame:
        """
        Load one data split (train, val, or test) into a DataFrame.

        Args:
            split: one of 'train', 'val', 'test'
            has_predictions: whether prediction file exists for this split
            has_probabilities: whether probability file exists for this split

        Returns:
            Standardised DataFrame for this split
        """
        print(f"  Loading {split} split...")

        # Load embeddings and labels (always present)
        embeddings = np.load(self.root / f"{split}_embeddings.npy")
        labels_int = np.load(self.root / f"{split}_labels.npy")

        n_samples, n_dims = embeddings.shape
        print(f"    Embeddings shape: {embeddings.shape}")
        print(f"    Labels: {n_samples} samples, "
              f"classes {np.unique(labels_int)}")

        # Convert integer labels to string names
        true_labels = self.mapper.to_names(labels_int)

        # Build base DataFrame
        df = pd.DataFrame()
        df[COL_SCAN_ID] = [f"{split}_{i:06d}" for i in range(n_samples)]
        df[COL_PATIENT_ID] = None
        df[COL_VISIT_ID] = None
        df[COL_VISIT_DATE] = None
        df[COL_TRUE_LABEL] = true_labels
        df["split"] = split

        # Load predictions if available
        pred_path = self.root / f"{split}_predictions.npy"
        if has_predictions and pred_path.exists():
            predictions_int = np.load(pred_path)
            df[COL_PREDICTED_LABEL] = self.mapper.to_names(predictions_int)
        else:
            df[COL_PREDICTED_LABEL] = None

        # Load probabilities if available
        prob_path = self.root / f"{split}_probabilities.npy"
        if has_probabilities and prob_path.exists():
            probabilities = np.load(prob_path)
            prob_cols = self.mapper.prob_col_names()
            for j, col in enumerate(prob_cols):
                df[col] = probabilities[:, j]
        else:
            for col in self.mapper.prob_col_names():
                df[col] = None

        # Add embedding columns
        emb_cols = self.mapper.emb_col_names(n_dims)
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)
        df = pd.concat([df.reset_index(drop=True),
                        emb_df.reset_index(drop=True)], axis=1)

        print(f"    Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df

    def load_train(self) -> pd.DataFrame:
        """Load training split. No predictions or probabilities."""
        return self._load_split(
            "train",
            has_predictions=False,
            has_probabilities=False
        )

    def load_val(self) -> pd.DataFrame:
        """Load validation split. Has predictions and probabilities."""
        return self._load_split("val")

    def load_test(self) -> pd.DataFrame:
        """Load test split. Has predictions and probabilities."""
        return self._load_split("test")

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all three splits.

        Returns:
            dict with keys 'train', 'val', 'test'
        """
        print("\n" + "=" * 60)
        print("  Loading all splits from .npy files")
        print("=" * 60)

        splits = {
            "train": self.load_train(),
            "val": self.load_val(),
            "test": self.load_test(),
        }

        total = sum(len(df) for df in splits.values())
        print(f"\n  Total rows loaded: {total:,}")
        print("=" * 60)
        return splits


# -------------------------------------------------------------
# CSV LOADER
# Loads a single combined CSV file.
# Used for future datasets or exported dissertation data.
# -------------------------------------------------------------

class CsvLoader:
    """
    Loads a single CSV file where embeddings, predictions,
    and probabilities are already combined.

    Expected columns:
        scan_id, patient_id, visit_id, visit_date,
        true_label, predicted_label,
        prob_CNV, prob_DME, prob_DRUSEN, prob_NORMAL,
        emb_0, emb_1, ..., emb_2047

    Args:
        filepath: path to the CSV file
        class_names: ordered list of class names
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        class_names: List[str] = None,
    ):
        self.filepath = Path(filepath)
        self.class_names = class_names or DEFAULT_CLASSES
        self.mapper = ClassLabelMapper(self.class_names)

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.filepath}"
            )

    def load(self) -> pd.DataFrame:
        """Load and validate the CSV file."""
        print(f"\n  Loading CSV: {self.filepath.name}")
        df = pd.read_csv(self.filepath)
        print(f"  Shape: {df.shape}")
        return df


# -------------------------------------------------------------
# DATA VALIDATOR
# Checks that a DataFrame has the required structure
# before passing it to scoring modules.
# -------------------------------------------------------------

class DataValidator:
    """
    Validates a standardised DataFrame before scoring.

    Checks:
        - Required columns are present
        - Embedding columns exist and are numeric
        - Probability columns exist and sum to ~1.0
        - No unexpected missing values in key columns
        - Class labels are in the expected set
    """

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        self.mapper = ClassLabelMapper(self.class_names)

    def validate(self, df: pd.DataFrame, split: str = "") -> bool:
        """
        Run all validation checks on a DataFrame.

        Returns True if all checks pass.
        Prints a warning for each issue found.
        """
        print(f"\n  Validating{' ' + split if split else ''} data...")
        issues = []

        # Check required columns
        required = [COL_SCAN_ID, COL_TRUE_LABEL, "split"]
        for col in required:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")

        # Check embedding columns exist
        emb_cols = [c for c in df.columns
                    if c.startswith(EMBEDDING_COL_PREFIX)]
        if len(emb_cols) == 0:
            issues.append("No embedding columns found "
                          f"(expected prefix '{EMBEDDING_COL_PREFIX}')")
        else:
            print(f"    Embedding columns: {len(emb_cols)} "
                  f"(dims 0 to {len(emb_cols)-1})")

        # Check probability columns exist
        prob_cols = [c for c in df.columns
                     if c.startswith(PROB_COL_PREFIX)]
        if len(prob_cols) == 0:
            print("    Warning: No probability columns found. "
                  "Uncertainty features will be unavailable.")
        else:
            print(f"    Probability columns: {prob_cols}")
            # Check probabilities sum to ~1.0 where not null
            prob_data = df[prob_cols].dropna()
            if len(prob_data) > 0:
                row_sums = prob_data.sum(axis=1)
                bad = (row_sums < 0.98) | (row_sums > 1.02)
                if bad.any():
                    issues.append(
                        f"{bad.sum()} rows have probabilities "
                        f"that do not sum to 1.0"
                    )

        # Check class labels
        if COL_TRUE_LABEL in df.columns:
            unique_labels = set(df[COL_TRUE_LABEL].dropna().unique())
            expected = set(self.class_names)
            unexpected = unique_labels - expected
            if unexpected:
                issues.append(
                    f"Unexpected class labels found: {unexpected}"
                )

        # Check missing values in scan_id
        if COL_SCAN_ID in df.columns:
            nulls = df[COL_SCAN_ID].isnull().sum()
            if nulls > 0:
                issues.append(f"{nulls} rows missing scan_id")

        # Report
        if issues:
            print(f"    Validation issues ({len(issues)}):")
            for issue in issues:
                print(f"      WARNING: {issue}")
            return False
        else:
            print(f"    All validation checks passed.")
            print(f"    Rows: {len(df):,}")
            return True


# -------------------------------------------------------------
# HELPER FUNCTIONS
# Convenience functions used by other modules.
# -------------------------------------------------------------

def detect_embedding_cols(df: pd.DataFrame) -> List[str]:
    """Return list of embedding column names in order."""
    cols = [c for c in df.columns if c.startswith(EMBEDDING_COL_PREFIX)]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def detect_prob_cols(df: pd.DataFrame) -> List[str]:
    """Return list of probability column names."""
    return [c for c in df.columns if c.startswith(PROB_COL_PREFIX)]


def get_embeddings_array(df: pd.DataFrame) -> np.ndarray:
    """Extract embedding columns as a numpy array."""
    emb_cols = detect_embedding_cols(df)
    return df[emb_cols].values


def get_probabilities_array(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Extract probability columns as a numpy array. None if not present."""
    prob_cols = detect_prob_cols(df)
    if not prob_cols:
        return None
    return df[prob_cols].values


def filter_normal_predicted(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where predicted label is NORMAL."""
    if COL_PREDICTED_LABEL not in df.columns:
        raise ValueError(
            f"Column '{COL_PREDICTED_LABEL}' not found. "
            "Cannot filter normal predictions."
        )
    mask = df[COL_PREDICTED_LABEL] == NORMAL_CLASS
    result = df[mask].copy()
    print(f"  NORMAL-predicted cases: {len(result):,} "
          f"of {len(df):,} total")
    return result