# =============================================================
# OCT Early Warning System
# src/anomaly_scorer.py
#
# Purpose: Layer A — Atypicality Detection
#
# Takes the fitted normal reference model and computes
# Mahalanobis distances for each case in the screening
# population. Assigns each case to one of four bands:
#
#   Core Normal        below core_normal_pct
#   Extended Normal    core_normal_pct to extended_normal_pct
#   Atypical Candidate extended_normal_pct to atypical_candidate_pct
#   Suspicious         above atypical_candidate_pct
#
# Band boundaries are derived from the distribution of
# Mahalanobis scores across the reference normal population
# in Phase 1, or from fixed validated values in Phase 2.
# =============================================================

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from config.phase_config import PhaseConfig, AtypicalityConfig
from config.defaults import (
    COL_MAHALANOBIS,
    COL_MAHALANOBIS_SQ,
    COL_LAYER_A_BAND,
    COL_LAYER_A_FLAG,
    BAND_CORE_NORMAL,
    BAND_EXTENDED_NORMAL,
    BAND_ATYPICAL_CANDIDATE,
    BAND_SUSPICIOUS,
    NORMAL_CLASS,
    COL_TRUE_LABEL,
    DEFAULT_REGULARISATION,
)
from src.reference_builder import NormalReferenceModel
from src.data_manager import get_embeddings_array


# -------------------------------------------------------------
# MAHALANOBIS SCORER
# Computes distances and assigns Layer A bands.
# -------------------------------------------------------------

class AnomalyScorer:
    """
    Layer A: Atypicality Detection.

    Computes Mahalanobis distance from the normal reference
    distribution for each case, then assigns a risk band
    based on the score distribution.

    The Mahalanobis distance measures how far a case sits
    from the centre of the normal embedding distribution,
    accounting for the covariance structure of that
    distribution. Cases with high scores are atypical
    relative to normal, regardless of what the classifier
    predicted.

    Args:
        reference_model: fitted NormalReferenceModel
        config: PhaseConfig controlling band boundaries
                and thresholding method
        verbose: whether to print progress messages
    """

    def __init__(
        self,
        reference_model: NormalReferenceModel,
        config: PhaseConfig,
        verbose: bool = True,
    ):
        self.ref = reference_model
        self.config = config
        self.verbose = verbose
        self.band_thresholds: Optional[Dict[str, float]] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def compute_mahalanobis(
        self, embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Mahalanobis distance for each embedding.

        Formula:
            D_M(x) = sqrt( (x - mu)^T * sigma_inv * (x - mu) )

        This is computed in vectorised form for efficiency
        across large embedding arrays.

        Args:
            embeddings: array of shape (n_samples, n_dims)

        Returns:
            distances: array of shape (n_samples,)
            distances_sq: squared distances, shape (n_samples,)
        """
        # Subtract normal mean from each embedding
        diffs = embeddings - self.ref.mu

        # Vectorised Mahalanobis:
        # For each row d: d @ sigma_inv @ d.T
        # Equivalent to (diffs @ sigma_inv) * diffs summed per row
        left = diffs @ self.ref.sigma_inv
        distances_sq = (left * diffs).sum(axis=1)

        # Clip negative values caused by floating point errors
        distances_sq = np.clip(distances_sq, 0, None)
        distances = np.sqrt(distances_sq)

        return distances, distances_sq

    def fit_bands(
        self, reference_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Derive band boundaries from the distribution of
        Mahalanobis scores across the normal reference
        population.

        In Phase 1, boundaries are derived from the score
        distribution using the selected method (percentile,
        robust, or chisquare).

        In Phase 2, boundaries are fixed validated values
        supplied externally via the config.

        Args:
            reference_scores: Mahalanobis scores computed
                on the NORMAL training population

        Returns:
            dict with threshold values for each band boundary
        """
        cfg = self.config.atypicality

        self._log("\n  Fitting Layer A band thresholds...")
        self._log(f"  Method: {cfg.method}")
        self._log(f"  Reference scores: {len(reference_scores):,} "
                  f"samples")
        self._log(f"  Score range: {reference_scores.min():.4f} "
                  f"to {reference_scores.max():.4f}")

        if self.config.is_phase_2():
            # Phase 2: use fixed validated boundaries
            # These must be set externally in the config
            self._log("  Phase 2: using fixed validated thresholds")
            thresholds = {
                "core_normal": cfg.core_normal_pct,
                "extended_normal": cfg.extended_normal_pct,
                "atypical_candidate": cfg.atypical_candidate_pct,
            }
        else:
            # Phase 1: derive from score distribution
            if cfg.method == "percentile":
                thresholds = self._fit_percentile(
                    reference_scores, cfg
                )
            elif cfg.method == "robust":
                thresholds = self._fit_robust(
                    reference_scores, cfg
                )
            elif cfg.method == "chisquare":
                thresholds = self._fit_chisquare(cfg)
            else:
                raise ValueError(
                    f"Unknown method: {cfg.method}. "
                    "Choose from: percentile, robust, chisquare"
                )

        self._log(f"\n  Band thresholds derived:")
        self._log(f"    Core Normal      : score <= "
                  f"{thresholds['core_normal']:.4f}")
        self._log(f"    Extended Normal  : score <= "
                  f"{thresholds['extended_normal']:.4f}")
        self._log(f"    Atypical Cand.   : score <= "
                  f"{thresholds['atypical_candidate']:.4f}")
        self._log(f"    Suspicious       : score >  "
                  f"{thresholds['atypical_candidate']:.4f}")

        self.band_thresholds = thresholds
        return thresholds

    def _fit_percentile(
        self,
        scores: np.ndarray,
        cfg: AtypicalityConfig,
    ) -> Dict[str, float]:
        """Derive thresholds from empirical percentiles."""
        return {
            "core_normal": float(
                np.percentile(scores, cfg.core_normal_pct)
            ),
            "extended_normal": float(
                np.percentile(scores, cfg.extended_normal_pct)
            ),
            "atypical_candidate": float(
                np.percentile(scores, cfg.atypical_candidate_pct)
            ),
        }

    def _fit_robust(
        self,
        scores: np.ndarray,
        cfg: AtypicalityConfig,
    ) -> Dict[str, float]:
        """
        Derive thresholds using median and MAD.
        More resistant to outliers in the reference population.

        Converts percentile targets to robust equivalents
        using the normal distribution relationship between
        percentiles and standard deviations.
        """
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        # Scale MAD to approximate standard deviation
        mad_std = mad * 1.4826

        # Convert percentile targets to z-score multipliers
        from scipy import stats
        z_core = stats.norm.ppf(cfg.core_normal_pct / 100)
        z_extended = stats.norm.ppf(cfg.extended_normal_pct / 100)
        z_atypical = stats.norm.ppf(
            cfg.atypical_candidate_pct / 100
        )

        return {
            "core_normal": float(median + z_core * mad_std),
            "extended_normal": float(
                median + z_extended * mad_std
            ),
            "atypical_candidate": float(
                median + z_atypical * mad_std
            ),
        }

    def _fit_chisquare(
        self, cfg: AtypicalityConfig
    ) -> Dict[str, float]:
        """
        Derive thresholds from chi-square distribution.

        Under the assumption that embeddings follow a
        multivariate normal distribution, squared Mahalanobis
        distances follow a chi-square distribution with
        degrees of freedom equal to the embedding dimension.

        Note: this assumption rarely holds exactly in deep
        learning embeddings, but provides a theoretically
        grounded reference point.
        """
        from scipy import stats
        df = self.ref.n_dims

        # Chi-square quantiles for squared distances
        # then take sqrt for distance scale
        core_sq = stats.chi2.ppf(
            cfg.core_normal_pct / 100, df=df
        )
        extended_sq = stats.chi2.ppf(
            cfg.extended_normal_pct / 100, df=df
        )
        atypical_sq = stats.chi2.ppf(
            cfg.atypical_candidate_pct / 100, df=df
        )

        return {
            "core_normal": float(np.sqrt(core_sq)),
            "extended_normal": float(np.sqrt(extended_sq)),
            "atypical_candidate": float(np.sqrt(atypical_sq)),
        }

    def assign_bands(
        self, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign Layer A band and flag to each case.

        Band assignment:
            Core Normal        score <= core_normal threshold
            Extended Normal    score <= extended_normal threshold
            Atypical Candidate score <= atypical_candidate threshold
            Suspicious         score > atypical_candidate threshold

        Flag assignment:
            False = Core Normal or Extended Normal
                    (does not proceed to Layer B)
            True  = Atypical Candidate or Suspicious
                    (proceeds to Layer B direction analysis)

        Args:
            scores: Mahalanobis distances, shape (n_samples,)

        Returns:
            bands: string array of band labels
            flags: boolean array, True = passes to Layer B
        """
        if self.band_thresholds is None:
            raise RuntimeError(
                "Band thresholds not fitted. "
                "Call fit_bands() before assign_bands()."
            )

        t = self.band_thresholds
        bands = np.where(
            scores <= t["core_normal"],
            BAND_CORE_NORMAL,
            np.where(
                scores <= t["extended_normal"],
                BAND_EXTENDED_NORMAL,
                np.where(
                    scores <= t["atypical_candidate"],
                    BAND_ATYPICAL_CANDIDATE,
                    BAND_SUSPICIOUS,
                )
            )
        )

        flags = (
            (bands == BAND_ATYPICAL_CANDIDATE) |
            (bands == BAND_SUSPICIOUS)
        )

        return bands, flags

    def score_dataframe(
        self,
        df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Score a full DataFrame and add Layer A columns.

        This is the main method called by other modules.
        It computes Mahalanobis distances, fits band
        thresholds (if not already fitted), assigns bands
        and flags, and returns the DataFrame with four
        new columns added:
            mahalanobis_distance
            mahalanobis_sq
            layer_a_band
            layer_a_flag

        Args:
            df: standardised DataFrame to score
            reference_df: optional separate reference DataFrame
                          used to fit band thresholds. If None,
                          uses the training normal distribution
                          from the reference model directly.

        Returns:
            df with Layer A columns added.
        """
        self._log("\n" + "=" * 60)
        self._log("  Layer A: Atypicality Scoring")
        self._log("=" * 60)

        if self.config.is_phase_1():
            self._log(f"\n  {self.config.disclaimer}")

        # Extract embeddings
        embeddings = get_embeddings_array(df)
        self._log(f"\n  Scoring {len(df):,} cases...")

        # Compute Mahalanobis distances
        distances, distances_sq = self.compute_mahalanobis(
            embeddings
        )
        self._log(f"  Score range: {distances.min():.4f} "
                  f"to {distances.max():.4f}")
        self._log(f"  Score mean : {distances.mean():.4f}")
        self._log(f"  Score median: {np.median(distances):.4f}")

        # Fit band thresholds if not already done
        if self.band_thresholds is None:
            if reference_df is not None:
                ref_embeddings = get_embeddings_array(
                    reference_df[
                        reference_df[COL_TRUE_LABEL] == NORMAL_CLASS
                    ]
                )
                ref_scores, _ = self.compute_mahalanobis(
                    ref_embeddings
                )
            else:
                # Use scores on NORMAL cases in current df
                # as reference if available
                if COL_TRUE_LABEL in df.columns:
                    normal_mask = df[COL_TRUE_LABEL] == NORMAL_CLASS
                    ref_scores = distances[normal_mask.values]
                    self._log(f"\n  Using {len(ref_scores):,} "
                              f"NORMAL cases from input "
                              f"for band fitting")
                else:
                    # Fall back to all scores
                    ref_scores = distances
                    self._log("\n  Warning: No true labels found. "
                              "Using all scores for band fitting.")

            self.fit_bands(ref_scores)

        # Assign bands and flags
        bands, flags = self.assign_bands(distances)

        # Add columns to DataFrame
        result_df = df.copy()
        result_df[COL_MAHALANOBIS] = distances
        result_df[COL_MAHALANOBIS_SQ] = distances_sq
        result_df[COL_LAYER_A_BAND] = bands
        result_df[COL_LAYER_A_FLAG] = flags

        # Print band distribution
        self._log("\n  Layer A band distribution:")
        band_counts = pd.Series(bands).value_counts()
        total = len(bands)
        for band in [BAND_CORE_NORMAL, BAND_EXTENDED_NORMAL,
                     BAND_ATYPICAL_CANDIDATE, BAND_SUSPICIOUS]:
            count = band_counts.get(band, 0)
            pct = count / total * 100
            flag = " --> Layer B" if band in [
                BAND_ATYPICAL_CANDIDATE, BAND_SUSPICIOUS
            ] else ""
            self._log(f"    {band:30s}: {count:5,} "
                      f"({pct:5.1f}%){flag}")

        flagged_total = flags.sum()
        self._log(f"\n  Total flagged for Layer B: "
                  f"{flagged_total:,} ({flagged_total/total*100:.1f}%)")

        return result_df

    def get_band_summary(
        self, scored_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return a summary table of band counts and percentages.

        Args:
            scored_df: DataFrame after score_dataframe()

        Returns:
            Summary DataFrame with band, count, percentage columns
        """
        counts = scored_df[COL_LAYER_A_BAND].value_counts()
        total = len(scored_df)

        summary = pd.DataFrame({
            "band": counts.index,
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(2),
            "passes_to_layer_b": [
                b in [BAND_ATYPICAL_CANDIDATE, BAND_SUSPICIOUS]
                for b in counts.index
            ]
        })

        return summary.reset_index(drop=True)