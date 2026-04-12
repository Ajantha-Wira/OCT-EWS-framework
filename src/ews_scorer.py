# =============================================================
# OCT Early Warning System
# src/ews_scorer.py
#
# Purpose: Composite EWS Scoring and Layer C Prioritisation
#
# Combines Layer A and Layer B signals into a single
# interpretable composite score, then applies Layer C
# operational budget to assign final actions.
#
# The composite score has three components:
#   1. Atypicality score  (from Layer A Mahalanobis distance)
#   2. Alignment score    (from Layer B cosine alignment)
#   3. Uncertainty score  (from classifier probabilities)
#
# Each component is returned separately so the score
# is fully transparent and traceable.
#
# Layer C then ranks all flagged cases by composite score
# and assigns:
#   Immediate Review    top immediate_review_rate fraction
#   Deferred Review     next deferred_review_rate fraction
#   Cases are NEVER silently dropped.
# =============================================================

import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from config.phase_config import PhaseConfig
from config.defaults import (
    COL_MAHALANOBIS,
    COL_LAYER_A_FLAG,
    COL_LAYER_A_BAND,
    COL_LAYER_B_CATEGORY,
    COL_STRONGEST_DIRECTION,
    COL_COMPOSITE_SCORE,
    COL_LAYER_C_ACTION,
    COL_PRIORITY_SCORE,
    NORMAL_CLASS,
    DISEASE_CLASSES,
    BAND_CORE_NORMAL,
    BAND_EXTENDED_NORMAL,
    BAND_ATYPICAL_CANDIDATE,
    BAND_SUSPICIOUS,
    CATEGORY_NON_SPECIFIC,
    CATEGORY_PROVISIONAL_BORDERLINE,
    CATEGORY_STRONGLY_ALIGNED,
    ACTION_SAFE_TO_DISMISS,
    ACTION_ROUTINE_MONITORING,
    ACTION_IMMEDIATE_REVIEW,
    ACTION_DEFERRED_REVIEW,
    ACTION_LOG_ONLY,
    PROB_COL_PREFIX,
)


# -------------------------------------------------------------
# UNCERTAINTY FEATURES
# Derived from classifier probability outputs.
# Used as a supplementary signal in the composite score.
# -------------------------------------------------------------

def compute_uncertainty_features(
    df: pd.DataFrame,
    disease_classes: List[str] = None,
) -> pd.DataFrame:
    """
    Compute uncertainty features from classifier probabilities.

    These are supplementary signals used for interpretation
    and triage. They are NOT the primary detection mechanism.
    Primary detection is driven by Mahalanobis distance.

    Features computed:
        entropy       Shannon entropy of class distribution
                      High entropy = high uncertainty
        margin        Gap between top-1 and top-2 probabilities
                      Low margin = classifier is uncertain
        disease_mass  Sum of non-NORMAL class probabilities
                      High mass = residual disease evidence
        top_prob      Maximum class probability
        second_prob   Second highest class probability

    Args:
        df: DataFrame with probability columns
        disease_classes: list of disease class names

    Returns:
        df with uncertainty feature columns added
    """
    disease_classes = disease_classes or DISEASE_CLASSES

    # Check probability columns exist
    prob_cols = [
        c for c in df.columns if c.startswith(PROB_COL_PREFIX)
    ]

    if not prob_cols:
        print("  Warning: No probability columns found. "
              "Uncertainty features will be zero.")
        result = df.copy()
        result["entropy"] = 0.0
        result["margin"] = 0.0
        result["disease_mass"] = 0.0
        result["top_prob"] = 0.0
        result["second_prob"] = 0.0
        return result

    probs = df[prob_cols].values.astype(float)

    # Entropy: -sum(p * log(p))
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    # Normalise to [0, 1] by dividing by log(n_classes)
    max_entropy = np.log(probs.shape[1])
    entropy_norm = entropy / max_entropy

    # Sort probabilities descending for margin calculation
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    top_prob = sorted_probs[:, 0]
    second_prob = sorted_probs[:, 1]
    margin = top_prob - second_prob

    # Disease mass: sum of non-NORMAL probabilities
    normal_col = f"{PROB_COL_PREFIX}{NORMAL_CLASS}"
    if normal_col in prob_cols:
        normal_probs = df[normal_col].values
        disease_mass = 1.0 - normal_probs
    else:
        # If NORMAL column not found sum disease class columns
        disease_prob_cols = [
            f"{PROB_COL_PREFIX}{cls}"
            for cls in disease_classes
            if f"{PROB_COL_PREFIX}{cls}" in prob_cols
        ]
        if disease_prob_cols:
            disease_mass = df[disease_prob_cols].sum(axis=1).values
        else:
            disease_mass = np.zeros(len(df))

    result = df.copy()
    result["entropy"] = entropy_norm
    result["margin"] = margin
    result["disease_mass"] = disease_mass
    result["top_prob"] = top_prob
    result["second_prob"] = second_prob

    return result


# -------------------------------------------------------------
# EWS SCORER
# Builds composite score and applies Layer C budget.
# -------------------------------------------------------------

class EWSScorer:
    """
    Composite EWS Scoring and Layer C Prioritisation.

    Combines Layer A atypicality, Layer B alignment, and
    classifier uncertainty into a transparent composite score.
    Then applies the operational budget to assign final
    review actions.

    The composite score is provisional in Phase 1 and
    workflow-ranking only. It does not alter the underlying
    detection status of any case.

    Args:
        config: PhaseConfig with budget and threshold settings
        disease_classes: list of disease class names
        weights: optional dict controlling component weights
                 keys: atypicality, alignment, uncertainty
                 default: equal weighting
        verbose: whether to print progress messages
    """

    def __init__(
        self,
        config: PhaseConfig,
        disease_classes: List[str] = None,
        weights: Optional[Dict[str, float]] = None,
        reference_model_id: str = "ref_unknown",
        formula_version: str = "v1",
        verbose: bool = True,
    ):
        self.config = config
        self.disease_classes = disease_classes or DISEASE_CLASSES
        self.verbose = verbose
        self.reference_model_id = reference_model_id
        self.formula_version = formula_version

        # Default equal weights for three components
        self.weights = weights or {
            "atypicality": 1.0,
            "alignment": 1.0,
            "uncertainty": 0.5,
        }

        # Build scoring metadata
        self.scoring_metadata = {
            "formula_version": self.formula_version,
            "reference_model_id": self.reference_model_id,
            "phase": self.config.phase,
            "atypicality_weight": self.weights["atypicality"],
            "alignment_weight": self.weights["alignment"],
            "uncertainty_weight": self.weights["uncertainty"],
            "urgency_enabled": True,
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _normalise_scores(
        self, scores: np.ndarray
    ) -> np.ndarray:
        """
        Normalise scores to [0, 1] range using min-max scaling.
        Handles edge case where all scores are identical.
        """
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s < 1e-10:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def _compute_atypicality_component(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Normalised Mahalanobis distance component.
        Only meaningful for flagged cases.
        """
        if COL_MAHALANOBIS not in df.columns:
            return np.zeros(len(df))
        scores = df[COL_MAHALANOBIS].fillna(0).values
        return self._normalise_scores(scores)

    def _compute_alignment_component(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Strongest cosine alignment component.
        Zero for non-flagged cases.
        """
        alignment = np.zeros(len(df))
        flagged_mask = df[COL_LAYER_A_FLAG] == True

        for cls in self.disease_classes:
            col = f"cosine_{cls}"
            if col in df.columns:
                cos_vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
                # Take maximum alignment across all classes
                alignment = np.maximum(alignment, cos_vals)

        # Zero out non-flagged cases
        alignment[~flagged_mask.values] = 0.0
        return self._normalise_scores(alignment)

    def _compute_uncertainty_component(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Uncertainty component combining entropy and
        disease mass. Supplementary signal only.
        """
        if "entropy" not in df.columns:
            return np.zeros(len(df))

        entropy = df["entropy"].fillna(0).values
        disease_mass = df["disease_mass"].fillna(0).values

        # Combine entropy and disease mass equally
        uncertainty = (entropy + disease_mass) / 2.0
        return self._normalise_scores(uncertainty)

    def _category_urgency_weight(
        self, category: Optional[str]
    ) -> float:
        """
        Return urgency multiplier based on Layer B category.
        Non-flagged cases return 0.
        """
        if category == CATEGORY_STRONGLY_ALIGNED:
            return 1.0
        elif category == CATEGORY_PROVISIONAL_BORDERLINE:
            return 0.6
        elif category == CATEGORY_NON_SPECIFIC:
            return 0.3
        else:
            return 0.0

    def compute_composite_score(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute the composite EWS score for all cases.

        Score formula:
            composite = (w_a * atypicality_norm +
                         w_b * alignment_norm +
                         w_u * uncertainty_norm) * urgency_weight

        Where urgency_weight is derived from Layer B category.
        Non-flagged cases always receive a composite score
        of 0.0 regardless of other signals.

        All component scores are returned separately alongside
        the composite so the result is fully interpretable.

        Args:
            df: DataFrame after Layer A and Layer B scoring

        Returns:
            df with composite score and component columns added
        """
        self._log("\n" + "=" * 60)
        self._log("  Computing Composite EWS Score")
        self._log("=" * 60)

        if self.config.is_phase_1():
            self._log(f"\n  {self.config.disclaimer}")

        result_df = df.copy()

        # Compute three components
        atypicality = self._compute_atypicality_component(result_df)
        alignment = self._compute_alignment_component(result_df)
        uncertainty = self._compute_uncertainty_component(result_df)

        # Urgency weights from Layer B category
        urgency = np.array([
            self._category_urgency_weight(cat)
            for cat in result_df.get(
                COL_LAYER_B_CATEGORY,
                pd.Series([None] * len(result_df))
            )
        ])

        # Weighted composite
        w = self.weights
        raw_composite = (
            w["atypicality"] * atypicality +
            w["alignment"] * alignment +
            w["uncertainty"] * uncertainty
        ) * urgency

        # Non-flagged cases always get 0
        flagged_mask = result_df[COL_LAYER_A_FLAG] == True
        raw_composite[~flagged_mask.values] = 0.0

        # Store components
        result_df["score_atypicality"] = atypicality
        result_df["score_alignment"] = alignment
        result_df["score_uncertainty"] = uncertainty
        result_df["score_urgency_weight"] = urgency
        result_df[COL_COMPOSITE_SCORE] = raw_composite

        # Stats for flagged cases only
        flagged_scores = raw_composite[flagged_mask.values]
        if len(flagged_scores) > 0:
            self._log(f"\n  Composite score (flagged cases):")
            self._log(f"    Min    : {flagged_scores.min():.4f}")
            self._log(f"    Max    : {flagged_scores.max():.4f}")
            self._log(f"    Mean   : {flagged_scores.mean():.4f}")
            self._log(f"    Median : "
                      f"{np.median(flagged_scores):.4f}")

        return result_df

    def apply_layer_c(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply Layer C operational budget to assign
        final review actions.

        Layer C is a workflow-ranking mechanism only.
        It does not change the detection status of any case.
        Cases above the detection threshold but outside
        the immediate review budget go to deferred queue.
        No detected case is silently dropped.

        Action assignment:
            Non-flagged cases:
                Core Normal     -> Safe to Dismiss
                Extended Normal -> Routine Monitoring

            Flagged cases ranked by composite score:
                Top immediate_review_rate    -> Immediate Review
                Next deferred_review_rate    -> Deferred Review
                Remaining flagged            -> Log Only

        Args:
            df: DataFrame after compute_composite_score()

        Returns:
            df with layer_c_action and priority_score columns.
        """
        self._log("\n" + "=" * 60)
        self._log("  Layer C: Operational Budget")
        self._log("=" * 60)

        budget = self.config.budget
        n_total = len(df)

        self._log(f"\n  Total cases         : {n_total:,}")
        self._log(f"  Immediate review    : "
                  f"{budget.immediate_review_rate*100:.0f}%")
        self._log(f"  Deferred review     : "
                  f"{budget.deferred_review_rate*100:.0f}%")
        self._log(f"  Review cycle        : "
                  f"{budget.review_cycle_days} days")

        result_df = df.copy()
        result_df[COL_LAYER_C_ACTION] = None
        result_df[COL_PRIORITY_SCORE] = 0.0

        # Non-flagged cases
        non_flagged_mask = result_df[COL_LAYER_A_FLAG] == False
        band = result_df[COL_LAYER_A_BAND]

        result_df.loc[
            non_flagged_mask & (band == BAND_CORE_NORMAL),
            COL_LAYER_C_ACTION
        ] = ACTION_SAFE_TO_DISMISS

        result_df.loc[
            non_flagged_mask & (band == BAND_EXTENDED_NORMAL),
            COL_LAYER_C_ACTION
        ] = ACTION_ROUTINE_MONITORING

        # Flagged cases — rank by composite score
        flagged_mask = result_df[COL_LAYER_A_FLAG] == True
        flagged_df = result_df[flagged_mask]
        n_flagged = len(flagged_df)

        if n_flagged == 0:
            self._log("  No flagged cases. Layer C skipped.")
            return result_df

        # Priority score = composite score
        # (could add class urgency weighting here in Phase 2)
        priority_scores = flagged_df[COL_COMPOSITE_SCORE].values
        result_df.loc[
            flagged_mask, COL_PRIORITY_SCORE
        ] = priority_scores

        # Rank flagged cases by priority score descending
        ranked_indices = flagged_df.index[
            np.argsort(priority_scores)[::-1]
        ]

        # Apply budget
        n_immediate = max(
            1, int(np.ceil(
                n_total * budget.immediate_review_rate
            ))
        )
        n_deferred = max(
            1, int(np.ceil(
                n_total * budget.deferred_review_rate
            ))
        )

        immediate_indices = ranked_indices[:n_immediate]
        deferred_indices = ranked_indices[
            n_immediate:n_immediate + n_deferred
        ]
        remaining_indices = ranked_indices[
            n_immediate + n_deferred:
        ]

        result_df.loc[
            immediate_indices, COL_LAYER_C_ACTION
        ] = ACTION_IMMEDIATE_REVIEW

        result_df.loc[
            deferred_indices, COL_LAYER_C_ACTION
        ] = ACTION_DEFERRED_REVIEW

        if budget.overflow_action == "queue":
            result_df.loc[
                remaining_indices, COL_LAYER_C_ACTION
            ] = ACTION_DEFERRED_REVIEW
        elif budget.overflow_action == "escalate":
            result_df.loc[
                remaining_indices, COL_LAYER_C_ACTION
            ] = ACTION_IMMEDIATE_REVIEW
        else:
            result_df.loc[
                remaining_indices, COL_LAYER_C_ACTION
            ] = ACTION_LOG_ONLY

        # Print action distribution
        self._log("\n  Layer C action distribution:")
        action_counts = result_df[COL_LAYER_C_ACTION].value_counts()
        for action in [
            ACTION_SAFE_TO_DISMISS,
            ACTION_ROUTINE_MONITORING,
            ACTION_IMMEDIATE_REVIEW,
            ACTION_DEFERRED_REVIEW,
            ACTION_LOG_ONLY,
        ]:
            count = action_counts.get(action, 0)
            pct = count / n_total * 100
            self._log(f"    {action:30s}: "
                      f"{count:5,} ({pct:5.1f}%)")

        self._log(f"\n  NOTE: The Layer C priority score is an "
                  f"operational workflow-ranking measure.")
        self._log(f"  It does not alter the detection status "
                  f"of any case.")

        return result_df

    def run_full_pipeline(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run composite scoring and Layer C in one call.

        Convenience method that chains compute_composite_score
        and apply_layer_c together.

        Args:
            df: DataFrame after Layer A and Layer B scoring

        Returns:
            Fully scored DataFrame with all EWS columns.
        """
        df = compute_uncertainty_features(
            df, self.disease_classes
        )
        df = self.compute_composite_score(df)
        df = self.apply_layer_c(df)
        
        # Attach scoring metadata as columns
        df["formula_version"] = self.formula_version
        df["reference_model_id"] = self.reference_model_id
        df["scoring_phase"] = self.config.phase        
        return df