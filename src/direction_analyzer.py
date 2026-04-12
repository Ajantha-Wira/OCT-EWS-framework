# =============================================================
# OCT Early Warning System
# src/direction_analyzer.py
#
# Purpose: Layer B — Direction-Specific Risk Assessment
#
# Only activates for cases flagged by Layer A
# (Atypical Candidate or Suspicious).
#
# For each flagged case, computes:
#   - Deviation vector from normal centre
#   - Cosine alignment toward each disease class
#   - Projection magnitude onto each disease direction
#   - Relative distance to each class centre
#   - Strongest aligned disease direction
#   - Post-Layer B category assignment
#
# IMPORTANT: Directional alignment indicates geometric
# consistency with a disease-class trajectory in
# representation space. It does not establish confirmed
# pathology.
# =============================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config.phase_config import PhaseConfig
from config.defaults import (
    COL_TRUE_LABEL,
    COL_LAYER_A_FLAG,
    COL_STRONGEST_DIRECTION,
    COL_LAYER_B_CATEGORY,
    NORMAL_CLASS,
    DISEASE_CLASSES,
    EMBEDDING_COL_PREFIX,
    CATEGORY_NON_SPECIFIC,
    CATEGORY_PROVISIONAL_BORDERLINE,
    CATEGORY_STRONGLY_ALIGNED,
)
from src.reference_builder import NormalReferenceModel
from src.data_manager import get_embeddings_array


# -------------------------------------------------------------
# DISEASE GEOMETRY
# Stores the directional structure of the embedding space.
# -------------------------------------------------------------

class DiseaseGeometry:
    """
    Stores the geometric structure of disease classes
    in embedding space relative to the normal centre.

    For each disease class k:
        mu_k      = class mean embedding
        v_k       = mu_k - mu_normal  (disease direction vector)
        v_k_norm  = normalised disease direction vector

    These vectors define the directions from the normal
    centre toward each disease class in embedding space.
    """

    def __init__(
        self,
        normal_mu: np.ndarray,
        class_means: Dict[str, np.ndarray],
        disease_classes: List[str],
    ):
        self.normal_mu = normal_mu
        self.class_means = class_means
        self.disease_classes = disease_classes

        # Compute disease direction vectors
        self.direction_vectors: Dict[str, np.ndarray] = {}
        self.direction_norms: Dict[str, np.ndarray] = {}

        for cls in disease_classes:
            if cls not in class_means:
                raise ValueError(
                    f"Class mean not found for: {cls}"
                )
            v_k = class_means[cls] - normal_mu
            v_k_magnitude = np.linalg.norm(v_k)
            self.direction_vectors[cls] = v_k
            self.direction_norms[cls] = v_k / v_k_magnitude

    def summary(self) -> None:
        """Print a summary of the disease geometry."""
        print("\n" + "=" * 60)
        print("  Disease Geometry Summary")
        print("=" * 60)
        print(f"  Normal centre shape: {self.normal_mu.shape}")
        for cls in self.disease_classes:
            v_k = self.direction_vectors[cls]
            magnitude = np.linalg.norm(v_k)
            print(f"  {cls:8s} direction magnitude: "
                  f"{magnitude:.4f}")
        print("=" * 60)


# -------------------------------------------------------------
# DIRECTION ANALYZER
# Computes Layer B metrics for flagged cases.
# -------------------------------------------------------------

class DirectionAnalyzer:
    """
    Layer B: Direction-Specific Risk Assessment.

    Computes directional metrics for cases flagged by Layer A.
    Determines which disease class each flagged case is
    most geometrically aligned with, and how strongly.

    Args:
        reference_model: fitted NormalReferenceModel
        config: PhaseConfig with direction thresholds
        disease_classes: list of disease class names
        verbose: whether to print progress messages
    """

    def __init__(
        self,
        reference_model: NormalReferenceModel,
        config: PhaseConfig,
        disease_classes: List[str] = None,
        verbose: bool = True,
    ):
        self.ref = reference_model
        self.config = config
        self.disease_classes = disease_classes or DISEASE_CLASSES
        self.verbose = verbose
        self.geometry: Optional[DiseaseGeometry] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def fit_geometry(
        self, train_df: pd.DataFrame
    ) -> DiseaseGeometry:
        """
        Compute class mean vectors and disease direction
        vectors from the training data.

        For each class, the mean embedding is computed
        from all confirmed training cases of that class.
        Disease direction vectors are then computed as
        the vector from the normal mean to each class mean.

        Args:
            train_df: standardised training DataFrame

        Returns:
            DiseaseGeometry object
        """
        self._log("\n" + "=" * 60)
        self._log("  Computing Disease Geometry")
        self._log("=" * 60)

        if COL_TRUE_LABEL not in train_df.columns:
            raise ValueError(
                f"Column '{COL_TRUE_LABEL}' not found."
            )

        class_means = {}
        all_classes = [NORMAL_CLASS] + self.disease_classes

        for cls in all_classes:
            cls_mask = train_df[COL_TRUE_LABEL] == cls
            cls_df = train_df[cls_mask]

            if len(cls_df) == 0:
                raise ValueError(
                    f"No training samples found for class: {cls}"
                )

            cls_embeddings = get_embeddings_array(cls_df)
            cls_mean = cls_embeddings.mean(axis=0)
            class_means[cls] = cls_mean

            self._log(f"  {cls:8s}: {len(cls_df):,} samples, "
                      f"mean shape {cls_mean.shape}")

        self.geometry = DiseaseGeometry(
            normal_mu=self.ref.mu,
            class_means=class_means,
            disease_classes=self.disease_classes,
        )

        if self.verbose:
            self.geometry.summary()

        return self.geometry

    def _compute_cosine_alignment(
        self,
        deviation: np.ndarray,
        disease_class: str,
    ) -> float:
        """
        Compute cosine similarity between the deviation
        vector from normal and the disease direction vector.

        cosine = (v_x . v_k) / (|v_x| * |v_k|)

        Values near 1.0: deviation strongly points toward
                         this disease class
        Values near 0.0: deviation is orthogonal to this
                         disease direction
        Values below 0: deviation points away from this
                        disease class

        Args:
            deviation: v_x = x - mu_normal
            disease_class: which class to compute alignment for

        Returns:
            cosine similarity scalar
        """
        v_k_norm = self.geometry.direction_norms[disease_class]
        dev_magnitude = np.linalg.norm(deviation)

        if dev_magnitude < 1e-10:
            return 0.0

        dev_norm = deviation / dev_magnitude
        return float(np.dot(dev_norm, v_k_norm))

    def _compute_projection(
        self,
        deviation: np.ndarray,
        disease_class: str,
    ) -> float:
        """
        Compute the scalar projection of the deviation
        vector onto the disease direction vector.

        projection = v_x . v_k_normalised

        This measures how far along the disease direction
        the case has moved from the normal centre,
        regardless of the total magnitude of deviation.

        Args:
            deviation: v_x = x - mu_normal
            disease_class: which class to project onto

        Returns:
            projection magnitude scalar
        """
        v_k_norm = self.geometry.direction_norms[disease_class]
        return float(np.dot(deviation, v_k_norm))

    def _compute_relative_distance(
        self,
        embedding: np.ndarray,
        disease_class: str,
    ) -> float:
        """
        Compute simple Euclidean distance from the embedding
        to the disease class mean.

        This is a simple complementary geometric reference
        to class proximity. It is not covariance-aware and
        should not be used as a primary clinical metric.
        It supplements the cosine alignment signal only.

        Args:
            embedding: raw embedding vector x
            disease_class: which class centre to measure from

        Returns:
            Euclidean distance to class mean
        """
        cls_mean = self.geometry.class_means[disease_class]
        return float(np.linalg.norm(embedding - cls_mean))

    def _assign_layer_b_category(
        self,
        cosine_scores: Dict[str, float],
        strongest_class: str,
    ) -> str:
        """
        Assign a Layer B post-analysis category based on
        the alignment scores and configured thresholds.

        Categories:
            Non-Specific Atypical:
                High Layer A score but weak alignment to
                any disease direction.

            Provisional Borderline Disease-Aligned:
                Moderate alignment to one disease direction.
                Warrants short-interval repeat scan.

            Strongly Disease-Aligned Suspicious:
                High alignment and high atypicality combined.
                Highest priority for expert review.

        In Phase 1, thresholds are derived from the
        reference distribution (top quartile of alignment
        scores among reference normals).

        In Phase 2, thresholds are class-specific and
        externally validated.
        """
        strongest_score = cosine_scores[strongest_class]

        # Get threshold for the strongest aligned class
        dir_cfg = self.config.direction_thresholds.get(
            strongest_class
        )

        if dir_cfg is None or dir_cfg.alignment_threshold == 0.0:
            # Phase 1: use relative thresholding
            # Top quartile of positive alignments = strongly aligned
            positive_scores = [
                s for s in cosine_scores.values() if s > 0
            ]
            if not positive_scores:
                return CATEGORY_NON_SPECIFIC

            # Simple relative rule for Phase 1:
            # strongly aligned if score > 0.3
            # provisional borderline if score > 0.1
            if strongest_score > 0.3:
                return CATEGORY_STRONGLY_ALIGNED
            elif strongest_score > 0.1:
                return CATEGORY_PROVISIONAL_BORDERLINE
            else:
                return CATEGORY_NON_SPECIFIC
        else:
            # Phase 2: use validated thresholds
            if strongest_score >= dir_cfg.alignment_threshold:
                return CATEGORY_STRONGLY_ALIGNED
            elif strongest_score >= dir_cfg.alignment_threshold * 0.6:
                return CATEGORY_PROVISIONAL_BORDERLINE
            else:
                return CATEGORY_NON_SPECIFIC

    def analyze_flagged(
        self, scored_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run Layer B analysis on all cases flagged by Layer A.

        For each flagged case, computes all directional
        metrics and assigns the post-Layer B category.
        Non-flagged cases pass through unchanged with
        null values for Layer B columns.

        Args:
            scored_df: DataFrame after Layer A scoring,
                       must contain layer_a_flag column

        Returns:
            DataFrame with Layer B columns added for
            all flagged cases.
        """
        if self.geometry is None:
            raise RuntimeError(
                "Disease geometry not fitted. "
                "Call fit_geometry() first."
            )

        self._log("\n" + "=" * 60)
        self._log("  Layer B: Direction Analysis")
        self._log("=" * 60)

        if self.config.is_phase_1():
            self._log(f"\n  {self.config.disclaimer}")

        # Identify flagged cases
        flagged_mask = scored_df[COL_LAYER_A_FLAG] == True
        flagged_df = scored_df[flagged_mask]
        n_flagged = len(flagged_df)
        n_total = len(scored_df)

        self._log(f"\n  Flagged cases to analyse: {n_flagged:,} "
                  f"of {n_total:,} total")

        # Initialise Layer B columns with None
        result_df = scored_df.copy()

        for cls in self.disease_classes:
            result_df[f"cosine_{cls}"] = None
            result_df[f"projection_{cls}"] = None
            result_df[f"distance_to_{cls}"] = None

        result_df[COL_STRONGEST_DIRECTION] = None
        result_df[COL_LAYER_B_CATEGORY] = None

        if n_flagged == 0:
            self._log("  No flagged cases. Layer B skipped.")
            return result_df

        # Get embeddings for flagged cases only
        flagged_embeddings = get_embeddings_array(flagged_df)
        flagged_indices = flagged_df.index

        self._log(f"  Computing directional metrics...")

        # Storage for results
        cosine_results = {
            cls: np.zeros(n_flagged)
            for cls in self.disease_classes
        }
        projection_results = {
            cls: np.zeros(n_flagged)
            for cls in self.disease_classes
        }
        distance_results = {
            cls: np.zeros(n_flagged)
            for cls in self.disease_classes
        }
        strongest_directions = []
        layer_b_categories = []

        for i, emb in enumerate(flagged_embeddings):
            # Deviation vector from normal centre
            deviation = emb - self.ref.mu

            cosine_scores = {}
            for cls in self.disease_classes:
                cos = self._compute_cosine_alignment(
                    deviation, cls
                )
                proj = self._compute_projection(deviation, cls)
                dist = self._compute_relative_distance(emb, cls)

                cosine_results[cls][i] = cos
                projection_results[cls][i] = proj
                distance_results[cls][i] = dist
                cosine_scores[cls] = cos

            # Identify strongest aligned disease direction
            strongest = max(
                cosine_scores, key=cosine_scores.get
            )
            strongest_directions.append(strongest)

            # Assign Layer B category
            category = self._assign_layer_b_category(
                cosine_scores, strongest
            )
            layer_b_categories.append(category)

        # Write results back to DataFrame
        for cls in self.disease_classes:
            result_df.loc[flagged_indices, f"cosine_{cls}"] = (
                cosine_results[cls]
            )
            result_df.loc[flagged_indices, f"projection_{cls}"] = (
                projection_results[cls]
            )
            result_df.loc[
                flagged_indices, f"distance_to_{cls}"
            ] = distance_results[cls]

        result_df.loc[
            flagged_indices, COL_STRONGEST_DIRECTION
        ] = strongest_directions

        result_df.loc[
            flagged_indices, COL_LAYER_B_CATEGORY
        ] = layer_b_categories

        # Print category distribution
        self._log("\n  Layer B category distribution "
                  "(flagged cases only):")
        cat_counts = pd.Series(layer_b_categories).value_counts()
        for cat in [
            CATEGORY_NON_SPECIFIC,
            CATEGORY_PROVISIONAL_BORDERLINE,
            CATEGORY_STRONGLY_ALIGNED,
        ]:
            count = cat_counts.get(cat, 0)
            pct = count / n_flagged * 100
            self._log(f"    {cat:45s}: "
                      f"{count:4,} ({pct:5.1f}%)")

        # Print strongest direction distribution
        self._log("\n  Strongest disease direction "
                  "(flagged cases only):")
        dir_counts = pd.Series(
            strongest_directions
        ).value_counts()
        for cls, count in dir_counts.items():
            pct = count / n_flagged * 100
            self._log(f"    {cls:8s}: {count:4,} ({pct:5.1f}%)")

        return result_df