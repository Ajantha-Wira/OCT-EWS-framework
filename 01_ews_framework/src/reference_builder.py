# =============================================================
# OCT Early Warning System
# src/reference_builder.py
#
# Purpose: Build the normal reference model in embedding space.
# This is the geometric foundation of Layer A.
#
# Takes confirmed NORMAL training embeddings and computes:
#   - Normal mean vector (mu_normal)
#   - Regularised covariance matrix (sigma_normal)
#   - Inverted covariance matrix (sigma_normal_inv)
#
# Optionally refines using only the core-normal subset
# to produce a tighter, more robust reference.
# =============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field

from config.defaults import (
    COL_TRUE_LABEL,
    NORMAL_CLASS,
    DEFAULT_REGULARISATION,
    COV_METHOD_STANDARD,
    COV_METHOD_SHRINKAGE,
    COV_METHOD_ROBUST,
    EMBEDDING_COL_PREFIX,
)
from src.data_manager import get_embeddings_array, detect_embedding_cols


# -------------------------------------------------------------
# REFERENCE MODEL DATACLASS
# Stores the fitted normal reference geometry.
# Passed to the anomaly scorer for Mahalanobis computation.
# -------------------------------------------------------------

@dataclass
class NormalReferenceModel:
    """
    Stores the fitted normal reference geometry.

    Attributes:
        mu:          normal mean vector, shape (n_dims,)
        sigma:       regularised covariance matrix, shape (n_dims, n_dims)
        sigma_inv:   inverted covariance matrix, shape (n_dims, n_dims)
        n_samples:   number of NORMAL samples used to fit the model
        n_dims:      embedding dimensionality
        method:      covariance estimation method used
        regularisation: lambda value added to diagonal
        core_normal_refined: whether core-normal refinement was applied
        core_normal_n: number of samples in core-normal subset if refined
    """
    mu: np.ndarray
    sigma: np.ndarray
    sigma_inv: np.ndarray
    n_samples: int
    n_dims: int
    method: str
    regularisation: float
    core_normal_refined: bool = False
    core_normal_n: Optional[int] = None

    def summary(self) -> None:
        """Print a summary of the fitted reference model."""
        print("\n" + "=" * 60)
        print("  Normal Reference Model Summary")
        print("=" * 60)
        print(f"  Samples used       : {self.n_samples:,}")
        print(f"  Embedding dims     : {self.n_dims:,}")
        print(f"  Covariance method  : {self.method}")
        print(f"  Regularisation     : {self.regularisation:.2e}")
        print(f"  Core-normal refine : {self.core_normal_refined}")
        if self.core_normal_refined:
            print(f"  Core-normal samples: {self.core_normal_n:,}")
        print(f"  Mean vector shape  : {self.mu.shape}")
        print(f"  Sigma shape        : {self.sigma.shape}")
        print(f"  Sigma_inv shape    : {self.sigma_inv.shape}")
        print("=" * 60)


# -------------------------------------------------------------
# REFERENCE BUILDER
# Fits the normal reference model from training embeddings.
# -------------------------------------------------------------

class ReferenceBuilder:
    """
    Builds the normal reference model from confirmed NORMAL
    training embeddings.

    The reference model defines what normal looks like in
    embedding space. Everything in Layer A is measured
    relative to this model.

    Args:
        regularisation: lambda added to covariance diagonal
                        for numerical stability in high
                        dimensional space. Default 1e-6.
        method: covariance estimation method.
                Options: standard, shrinkage, robust
        verbose: whether to print progress messages
    """

    def __init__(
        self,
        regularisation: float = DEFAULT_REGULARISATION,
        method: str = COV_METHOD_STANDARD,
        verbose: bool = True,
    ):
        self.regularisation = regularisation
        self.method = method
        self.verbose = verbose
        self.model: Optional[NormalReferenceModel] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _extract_normal_embeddings(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Filter DataFrame to confirmed NORMAL cases using
        true_label, then extract embedding array.

        Uses true_label not predicted_label because we want
        the ground-truth normal distribution, not the
        classifier's normal predictions.
        """
        if COL_TRUE_LABEL not in df.columns:
            raise ValueError(
                f"Column '{COL_TRUE_LABEL}' not found. "
                "Reference builder requires ground truth labels."
            )

        normal_mask = df[COL_TRUE_LABEL] == NORMAL_CLASS
        normal_df = df[normal_mask]

        if len(normal_df) == 0:
            raise ValueError(
                "No NORMAL samples found in the provided DataFrame. "
                f"Check that true_label contains '{NORMAL_CLASS}'."
            )

        self._log(f"\n  NORMAL samples found: {len(normal_df):,} "
                  f"of {len(df):,} total")

        return get_embeddings_array(normal_df)

    def _compute_covariance(
        self, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute covariance matrix using the selected method.

        standard:  standard empirical covariance via numpy
        shrinkage: Ledoit-Wolf shrinkage estimator via sklearn
        robust:    minimum covariance determinant via sklearn
        """
        if self.method == COV_METHOD_STANDARD:
            return np.cov(embeddings.T)

        elif self.method == COV_METHOD_SHRINKAGE:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(embeddings)
            return lw.covariance_

        elif self.method == COV_METHOD_ROBUST:
            from sklearn.covariance import MinCovDet
            self._log("  Note: Robust covariance (MinCovDet) may be "
                      "slow on large datasets.")
            mcd = MinCovDet(random_state=42)
            mcd.fit(embeddings)
            return mcd.covariance_

        else:
            raise ValueError(
                f"Unknown covariance method: {self.method}. "
                f"Choose from: standard, shrinkage, robust"
            )

    def _regularise_and_invert(
        self, sigma: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add regularisation to the diagonal and invert.

        Regularisation prevents singular matrix errors in
        high-dimensional embedding space where the number
        of dimensions (2048) may approach or exceed the
        effective rank of the covariance matrix.

        lambda = 1e-6 is the same value used in the
        dissertation implementation.
        """
        n_dims = sigma.shape[0]
        sigma_reg = sigma + np.eye(n_dims) * self.regularisation
        self._log(f"  Regularisation applied: lambda = "
                  f"{self.regularisation:.2e}")

        self._log("  Inverting covariance matrix...")
        sigma_inv = np.linalg.inv(sigma_reg)
        self._log("  Inversion complete.")

        return sigma_reg, sigma_inv

    def fit(self, train_df: pd.DataFrame) -> NormalReferenceModel:
        """
        Fit the normal reference model from training data.

        Args:
            train_df: standardised training DataFrame from
                      NpyLoader. Must contain true_label and
                      embedding columns.

        Returns:
            NormalReferenceModel with mu, sigma, sigma_inv.
        """
        self._log("\n" + "=" * 60)
        self._log("  Building Normal Reference Model")
        self._log("=" * 60)
        self._log(f"  Method: {self.method}")

        # Extract confirmed NORMAL embeddings
        normal_embeddings = self._extract_normal_embeddings(train_df)
        n_samples, n_dims = normal_embeddings.shape
        self._log(f"  Embedding shape: {normal_embeddings.shape}")

        # Compute mean vector
        self._log("\n  Computing normal mean vector...")
        mu = normal_embeddings.mean(axis=0)
        self._log(f"  Mean shape: {mu.shape}")

        # Compute covariance matrix
        self._log(f"\n  Computing covariance ({self.method})...")
        sigma_raw = self._compute_covariance(normal_embeddings)
        self._log(f"  Covariance shape: {sigma_raw.shape}")

        # Regularise and invert
        sigma_reg, sigma_inv = self._regularise_and_invert(sigma_raw)

        # Store model
        self.model = NormalReferenceModel(
            mu=mu,
            sigma=sigma_reg,
            sigma_inv=sigma_inv,
            n_samples=n_samples,
            n_dims=n_dims,
            method=self.method,
            regularisation=self.regularisation,
            core_normal_refined=False,
            core_normal_n=None,
        )

        if self.verbose:
            self.model.summary()

        return self.model

    def fit_with_core_refinement(
        self,
        train_df: pd.DataFrame,
        core_percentile: float = 50.0,
    ) -> NormalReferenceModel:
        """
        Fit the reference model using iterative core-normal
        refinement.

        This two-stage process:
        1. Fits an initial model on all NORMAL embeddings
        2. Computes preliminary Mahalanobis scores
        3. Selects only the core-normal subset (cases below
           the core_percentile of Mahalanobis scores)
        4. Re-fits the model on this tighter subset

        The result is a reference model that represents the
        dense central region of the normal distribution,
        less influenced by borderline or atypical normals
        in the training set.

        Args:
            train_df: training DataFrame
            core_percentile: percentile cutoff for core-normal
                             subset. Default 50.0 (median).

        Returns:
            Refined NormalReferenceModel.
        """
        self._log("\n" + "=" * 60)
        self._log("  Building Normal Reference Model")
        self._log("  with Core-Normal Refinement")
        self._log("=" * 60)

        # Stage 1: initial fit on all NORMAL embeddings
        self._log("\n  Stage 1: Initial fit on all NORMAL embeddings")
        normal_embeddings = self._extract_normal_embeddings(train_df)
        n_all = len(normal_embeddings)

        mu_init = normal_embeddings.mean(axis=0)
        sigma_raw = self._compute_covariance(normal_embeddings)
        sigma_reg, sigma_inv = self._regularise_and_invert(sigma_raw)

        # Stage 2: compute preliminary Mahalanobis scores
        self._log("\n  Stage 2: Computing preliminary Mahalanobis scores")
        diffs = normal_embeddings - mu_init
        scores = np.array([
            np.sqrt(d @ sigma_inv @ d) for d in diffs
        ])
        self._log(f"  Score range: {scores.min():.2f} to "
                  f"{scores.max():.2f}")

        # Stage 3: select core-normal subset
        threshold = np.percentile(scores, core_percentile)
        core_mask = scores <= threshold
        core_embeddings = normal_embeddings[core_mask]
        n_core = len(core_embeddings)

        self._log(f"\n  Stage 3: Core-normal subset")
        self._log(f"  Percentile cutoff : {core_percentile:.1f}")
        self._log(f"  Score threshold   : {threshold:.4f}")
        self._log(f"  Core samples      : {n_core:,} of {n_all:,} "
                  f"({n_core/n_all*100:.1f}%)")

        # Stage 4: re-fit on core-normal subset
        self._log("\n  Stage 4: Re-fitting on core-normal subset")
        mu_core = core_embeddings.mean(axis=0)
        sigma_core_raw = self._compute_covariance(core_embeddings)
        sigma_core_reg, sigma_core_inv = self._regularise_and_invert(
            sigma_core_raw
        )

        self.model = NormalReferenceModel(
            mu=mu_core,
            sigma=sigma_core_reg,
            sigma_inv=sigma_core_inv,
            n_samples=n_all,
            n_dims=normal_embeddings.shape[1],
            method=self.method,
            regularisation=self.regularisation,
            core_normal_refined=True,
            core_normal_n=n_core,
        )

        if self.verbose:
            self.model.summary()

        return self.model

    def save(self, save_dir: str, notes: str = "") -> str:
        """
        Save the fitted reference model to disk.

        Saves mu, sigma, and sigma_inv as separate .npy files
        and a metadata JSON file for full traceability.

        Args:
            save_dir: directory to save the model files
            notes: optional free-text notes about this model

        Returns:
            reference_model_id string
        """
        if self.model is None:
            raise RuntimeError(
                "No model fitted yet. Call fit() first."
            )

        import json
        from datetime import datetime

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate reference model ID
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        pca_tag = "pcaon" if False else "pcaoff"
        core_tag = (
            "coreon" if self.model.core_normal_refined
            else "coreoff"
        )
        ref_id = (
            f"ref_{timestamp}_{self.method}_"
            f"d{self.model.n_dims}_{core_tag}_{pca_tag}"
        )

        # Save numpy arrays
        np.save(save_path / "normal_mu.npy", self.model.mu)
        np.save(save_path / "normal_sigma.npy", self.model.sigma)
        np.save(save_path / "normal_sigma_inv.npy",
                self.model.sigma_inv)

        # Save metadata JSON
        metadata = {
            "reference_model_id": ref_id,
            "created_at": datetime.now().isoformat(),
            "source_split": "train",
            "normal_case_count": self.model.n_samples,
            "embedding_dim": self.model.n_dims,
            "covariance_method": self.method,
            "regularisation": self.regularisation,
            "core_normal_enabled": self.model.core_normal_refined,
            "core_normal_percentile": (
                None if not self.model.core_normal_refined
                else "50.0"
            ),
            "core_normal_n": self.model.core_normal_n,
            "pca_enabled": False,
            "pca_components": None,
            "notes": notes,
        }

        with open(save_path / "reference_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        self._log(f"\n  Reference model saved to: {save_path}")
        self._log(f"  Reference model ID: {ref_id}")
        self._log(f"    normal_mu.npy        {self.model.mu.shape}")
        self._log(f"    normal_sigma.npy     {self.model.sigma.shape}")
        self._log(f"    normal_sigma_inv.npy "
                f"{self.model.sigma_inv.shape}")
        self._log(f"    reference_metadata.json")

        return ref_id

    @staticmethod
    def load(save_dir: str) -> Tuple[NormalReferenceModel, dict]:
        """
        Load a previously saved reference model from disk.

        Args:
            save_dir: directory containing the saved files

        Returns:
            Tuple of (NormalReferenceModel, metadata dict)
        """
        import json

        save_path = Path(save_dir)

        mu = np.load(save_path / "normal_mu.npy")
        sigma = np.load(save_path / "normal_sigma.npy")
        sigma_inv = np.load(save_path / "normal_sigma_inv.npy")

        # Load metadata
        metadata_path = save_path / "reference_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {
                "reference_model_id": "ref_unknown",
                "covariance_method": "unknown",
                "notes": "No metadata file found.",
            }

        model = NormalReferenceModel(
            mu=mu,
            sigma=sigma,
            sigma_inv=sigma_inv,
            n_samples=metadata.get("normal_case_count", 0),
            n_dims=mu.shape[0],
            method=metadata.get("covariance_method", "unknown"),
            regularisation=metadata.get("regularisation", 0.0),
            core_normal_refined=metadata.get(
                "core_normal_enabled", False
            ),
            core_normal_n=metadata.get("core_normal_n"),
        )

        print(f"\n  Reference model loaded from: {save_path}")
        print(f"  Reference model ID: "
            f"{metadata.get('reference_model_id', 'unknown')}")
        print(f"  Created at: {metadata.get('created_at', 'unknown')}")
        print(f"  Mean shape : {mu.shape}")
        print(f"  Sigma shape: {sigma.shape}")

        return model, metadata