# =============================================================
# OCT Early Warning System
# src/report_generator.py
#
# Purpose: Generate all system outputs in a structured,
# traceable, and self-describing format.
#
# Produces:
#   1. Scan-level CSV with all EWS columns and metadata
#   2. Patient-level monitoring summary CSV
#   3. Phase 1 shortlist of borderline cases
#   4. Visualisations:
#       - Mahalanobis score histogram with band overlays
#       - Layer A band distribution bar chart
#       - Cosine alignment distributions by disease class
#       - Layer C action distribution
#
# Every output carries:
#   - reference_model_id
#   - formula_version
#   - phase
#   - disclaimer (Phase 1 only)
#   - generation timestamp
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

from config.phase_config import PhaseConfig
from config.defaults import (
    COL_MAHALANOBIS,
    COL_MAHALANOBIS_SQ,
    COL_LAYER_A_BAND,
    COL_LAYER_A_FLAG,
    COL_LAYER_B_CATEGORY,
    COL_STRONGEST_DIRECTION,
    COL_COMPOSITE_SCORE,
    COL_LAYER_C_ACTION,
    COL_PRIORITY_SCORE,
    COL_TRUE_LABEL,
    COL_PREDICTED_LABEL,
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
    NORMAL_CLASS,
    DISEASE_CLASSES,
)


# -------------------------------------------------------------
# COLOUR SCHEME
# Consistent colours used across all visualisations.
# -------------------------------------------------------------

BAND_COLOURS = {
    BAND_CORE_NORMAL: "#2E8B57",
    BAND_EXTENDED_NORMAL: "#90EE90",
    BAND_ATYPICAL_CANDIDATE: "#FFA500",
    BAND_SUSPICIOUS: "#DC143C",
}

ACTION_COLOURS = {
    ACTION_SAFE_TO_DISMISS: "#2E8B57",
    ACTION_ROUTINE_MONITORING: "#90EE90",
    ACTION_DEFERRED_REVIEW: "#FFA500",
    ACTION_IMMEDIATE_REVIEW: "#DC143C",
}

DIRECTION_COLOURS = {
    "CNV": "#DC143C",
    "DME": "#FF8C00",
    "DRUSEN": "#4169E1",
}


# -------------------------------------------------------------
# REPORT GENERATOR
# -------------------------------------------------------------

class ReportGenerator:
    """
    Generates all EWS system outputs in a structured,
    traceable, and self-describing format.

    Every output carries reference_model_id, formula_version,
    phase, disclaimer, and generation timestamp so reports
    remain interpretable when separated from the code.

    Args:
        config: PhaseConfig for this run
        output_dir: directory to save all outputs
        reference_model_id: ID of the reference model used
        formula_version: composite score formula version
        verbose: whether to print progress messages
    """

    def __init__(
        self,
        config: PhaseConfig,
        output_dir: str,
        reference_model_id: str = "ref_unknown",
        formula_version: str = "v1",
        verbose: bool = True,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_model_id = reference_model_id
        self.formula_version = formula_version
        self.verbose = verbose
        self.generated_at = datetime.now().isoformat()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _metadata_block(self) -> Dict:
        """Return standard metadata dict attached to all outputs."""
        return {
            "generated_at": self.generated_at,
            "reference_model_id": self.reference_model_id,
            "formula_version": self.formula_version,
            "phase": self.config.phase,
            "disclaimer": self.config.get_disclaimer(),
        }

    def _save_csv_with_metadata(
        self,
        df: pd.DataFrame,
        filename: str,
        description: str,
    ) -> Path:
        """
        Save a DataFrame as CSV with a metadata header block
        at the top so the file is self-describing.

        The metadata block appears as commented lines
        starting with # at the top of the file.

        Args:
            df: DataFrame to save
            filename: output filename
            description: brief description of this report

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        meta = self._metadata_block()

        # Write metadata header
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# OCT Early Warning System — {description}\n")
            f.write(f"# Generated at  : {meta['generated_at']}\n")
            f.write(f"# Reference ID  : "
                    f"{meta['reference_model_id']}\n")
            f.write(f"# Formula ver   : "
                    f"{meta['formula_version']}\n")
            f.write(f"# Phase         : {meta['phase']}\n")
            if meta["disclaimer"]:
                f.write(f"# DISCLAIMER    : "
                        f"{meta['disclaimer']}\n")
            f.write("#\n")

        # Append DataFrame
        df.to_csv(filepath, mode="a", index=False)
        self._log(f"  Saved: {filepath.name} "
                  f"({len(df):,} rows)")
        return filepath

    # ----------------------------------------------------------
    # REPORT 1: SCAN-LEVEL OUTPUT
    # ----------------------------------------------------------

    def save_scan_report(
        self,
        scored_df: pd.DataFrame,
        filename: str = "scan_level_report.csv",
    ) -> Path:
        """
        Save the complete scan-level EWS assessment as CSV.

        Includes all EWS columns plus metadata. This is the
        primary output of the system for each scoring run.

        Args:
            scored_df: fully scored DataFrame
            filename: output filename

        Returns:
            Path to saved file
        """
        self._log("\n" + "=" * 60)
        self._log("  Report 1: Scan-Level Output")
        self._log("=" * 60)

        # Select key columns for the report
        report_cols = [
            "scan_id", COL_TRUE_LABEL, COL_PREDICTED_LABEL,
            COL_MAHALANOBIS, COL_LAYER_A_BAND,
            COL_LAYER_A_FLAG, COL_LAYER_B_CATEGORY,
            COL_STRONGEST_DIRECTION, COL_COMPOSITE_SCORE,
            COL_LAYER_C_ACTION, COL_PRIORITY_SCORE,
        ]

        # Add optional columns if present
        optional = [
            "score_atypicality", "score_alignment",
            "score_uncertainty", "score_urgency_weight",
            "entropy", "disease_mass", "margin",
            "formula_version", "reference_model_id",
            "scoring_phase",
        ]
        for col in optional:
            if col in scored_df.columns:
                report_cols.append(col)

        # Add cosine columns if present
        for cls in DISEASE_CLASSES:
            col = f"cosine_{cls}"
            if col in scored_df.columns:
                report_cols.append(col)

        available = [c for c in report_cols
                     if c in scored_df.columns]
        report_df = scored_df[available].copy()

        return self._save_csv_with_metadata(
            report_df,
            filename,
            "Scan-Level EWS Assessment"
        )

    # ----------------------------------------------------------
    # REPORT 2: PATIENT MONITORING SUMMARY
    # ----------------------------------------------------------

    def save_patient_summary(
        self,
        patient_summaries: pd.DataFrame,
        filename: str = "patient_monitoring_summary.csv",
    ) -> Path:
        """
        Save the patient-level monitoring summary as CSV.

        Args:
            patient_summaries: DataFrame from
                               PatientMonitor.get_all_summaries()
            filename: output filename

        Returns:
            Path to saved file
        """
        self._log("\n" + "=" * 60)
        self._log("  Report 2: Patient Monitoring Summary")
        self._log("=" * 60)

        return self._save_csv_with_metadata(
            patient_summaries,
            filename,
            "Patient Monitoring Summary"
        )

    # ----------------------------------------------------------
    # REPORT 3: PHASE 1 SHORTLIST
    # ----------------------------------------------------------

    def save_shortlist(
        self,
        scored_df: pd.DataFrame,
        filename: str = "phase1_shortlist.csv",
        top_n: Optional[int] = None,
    ) -> Path:
        """
        Save the Phase 1 shortlist of borderline cases
        recommended for repeat scan follow-up.

        The shortlist contains all cases that:
        1. Are predicted NORMAL by the classifier
        2. Are flagged by Layer A (Atypical Candidate
           or Suspicious)

        Cases are ranked by composite EWS score descending
        so the highest priority cases appear first.

        Args:
            scored_df: fully scored DataFrame
            filename: output filename
            top_n: if specified, return only the top N cases

        Returns:
            Path to saved file
        """
        self._log("\n" + "=" * 60)
        self._log("  Report 3: Phase 1 Shortlist")
        self._log("=" * 60)

        # Filter: NORMAL-predicted AND flagged by Layer A
        normal_mask = (
            scored_df[COL_PREDICTED_LABEL] == NORMAL_CLASS
        )
        flagged_mask = scored_df[COL_LAYER_A_FLAG] == True
        shortlist = scored_df[
            normal_mask & flagged_mask
        ].copy()

        # Sort by composite score descending
        if COL_COMPOSITE_SCORE in shortlist.columns:
            shortlist = shortlist.sort_values(
                COL_COMPOSITE_SCORE, ascending=False
            )

        # Apply top_n limit if specified
        if top_n is not None:
            shortlist = shortlist.head(top_n)

        self._log(f"  Shortlist size: {len(shortlist):,} cases")
        self._log(f"  (NORMAL-predicted AND Layer A flagged)")

        # Select shortlist columns
        shortlist_cols = [
            "scan_id", "patient_id", COL_TRUE_LABEL,
            COL_PREDICTED_LABEL, COL_MAHALANOBIS,
            COL_LAYER_A_BAND, COL_LAYER_B_CATEGORY,
            COL_STRONGEST_DIRECTION, COL_COMPOSITE_SCORE,
            COL_LAYER_C_ACTION,
        ]

        for cls in DISEASE_CLASSES:
            col = f"cosine_{cls}"
            if col in shortlist.columns:
                shortlist_cols.append(col)

        available = [c for c in shortlist_cols
                     if c in shortlist.columns]
        shortlist_report = shortlist[available].copy()

        return self._save_csv_with_metadata(
            shortlist_report,
            filename,
            "Phase 1 Repeat-Scan Shortlist "
            "(NORMAL-predicted flagged cases)"
        )
    # ----------------------------------------------------------
    # REPORT 4: EWS-SPECIFIC SHORTLIST
    # ----------------------------------------------------------

    def save_ews_shortlist(
        self,
        scored_df: pd.DataFrame,
        filename: str = "ews_early_warning_shortlist.csv",
        top_n: Optional[int] = None,
    ) -> Path:
        """
        Save the EWS-specific early warning shortlist.

        This is the true research output of the EWS system.
        It is distinct from the global Layer C ranking which
        is dominated by confirmed obvious disease cases.

        The EWS shortlist answers the research question:
        Among cases that appear normal at the classifier
        output level, which ones show the most concerning
        embedding-space geometry?

        Filters to NORMAL-predicted cases only, then ranks
        within that subset by composite score. This surfaces
        the early warning signal that is the actual scientific
        contribution of the EWS framework.

        Args:
            scored_df: fully scored DataFrame
            filename: output filename
            top_n: if specified, return only the top N cases

        Returns:
            Path to saved file
        """
        self._log("\n" + "=" * 60)
        self._log("  Report 4: EWS Early Warning Shortlist")
        self._log("  (NORMAL-predicted cases ranked within subset)")
        self._log("=" * 60)

        # Filter to NORMAL-predicted cases only
        normal_predicted = scored_df[
            scored_df[COL_PREDICTED_LABEL] == NORMAL_CLASS
        ].copy()

        n_normal_predicted = len(normal_predicted)
        self._log(f"  NORMAL-predicted cases: "
                  f"{n_normal_predicted:,}")

        # Split into flagged and non-flagged
        flagged = normal_predicted[
            normal_predicted[COL_LAYER_A_FLAG] == True
        ].copy()
        not_flagged = normal_predicted[
            normal_predicted[COL_LAYER_A_FLAG] == False
        ].copy()

        self._log(f"  Flagged by Layer A    : {len(flagged):,}")
        self._log(f"  Not flagged           : {len(not_flagged):,}")

        if len(flagged) == 0:
            self._log("  No flagged NORMAL cases. "
                      "EWS shortlist is empty.")
            empty = pd.DataFrame()
            return self._save_csv_with_metadata(
                empty, filename,
                "EWS Early Warning Shortlist (empty)"
            )

        # Rank flagged cases within NORMAL-predicted subset
        # by composite score descending
        if COL_COMPOSITE_SCORE in flagged.columns:
            flagged = flagged.sort_values(
                COL_COMPOSITE_SCORE, ascending=False
            )

        if top_n is not None:
            flagged = flagged.head(top_n)

        # Add EWS rank column
        flagged = flagged.copy()
        flagged.insert(0, "ews_rank", range(1, len(flagged) + 1))

        # Add clinical interpretation column
        def interpret(row):
            band = row.get(COL_LAYER_A_BAND, "")
            cat = row.get(COL_LAYER_B_CATEGORY, "")
            direction = row.get(COL_STRONGEST_DIRECTION, "")

            if cat == CATEGORY_STRONGLY_ALIGNED:
                return (
                    f"High priority: atypical normal with "
                    f"strong {direction} direction alignment. "
                    f"Recommend short-interval repeat scan."
                )
            elif cat == CATEGORY_PROVISIONAL_BORDERLINE:
                return (
                    f"Moderate priority: borderline normal "
                    f"with emerging {direction} direction signal. "
                    f"Recommend follow-up monitoring."
                )
            else:
                return (
                    f"Lower priority: atypical normal with "
                    f"non-specific deviation. "
                    f"Recommend routine follow-up."
                )

        flagged["clinical_interpretation"] = flagged.apply(
            interpret, axis=1
        )

        # Select columns for EWS shortlist report
        shortlist_cols = [
            "ews_rank", "scan_id", "patient_id",
            COL_TRUE_LABEL, COL_PREDICTED_LABEL,
            COL_MAHALANOBIS, COL_LAYER_A_BAND,
            COL_LAYER_B_CATEGORY, COL_STRONGEST_DIRECTION,
            COL_COMPOSITE_SCORE, COL_LAYER_C_ACTION,
            "clinical_interpretation",
        ]

        for cls in DISEASE_CLASSES:
            col = f"cosine_{cls}"
            if col in flagged.columns:
                shortlist_cols.append(col)

        available = [c for c in shortlist_cols
                     if c in flagged.columns]
        shortlist_report = flagged[available].copy()

        self._log(f"\n  EWS shortlist: {len(shortlist_report)} "
                  f"cases ranked within NORMAL-predicted subset")

        # Print top 10
        if self.verbose and len(shortlist_report) > 0:
            self._log("\n  Top cases in EWS shortlist:")
            display_cols = [
                "ews_rank", "scan_id",
                COL_MAHALANOBIS, COL_LAYER_A_BAND,
                COL_STRONGEST_DIRECTION,
                COL_LAYER_B_CATEGORY,
            ]
            avail = [c for c in display_cols
                     if c in shortlist_report.columns]
            self._log(
                shortlist_report[avail].head(10).to_string()
            )

        return self._save_csv_with_metadata(
            shortlist_report,
            filename,
            "EWS Early Warning Shortlist "
            "(NORMAL-predicted cases ranked within subset)"
        )

    # ----------------------------------------------------------
    # VISUALISATION 5: NORMAL-ONLY COSINE ALIGNMENTS
    # ----------------------------------------------------------

    def plot_normal_cosine_alignments(
        self,
        scored_df: pd.DataFrame,
        filename: str = "normal_cosine_alignments.png",
    ) -> Path:
        """
        Plot cosine alignment distributions specifically for
        flagged true-NORMAL cases only.

        This is the most clinically important visualisation
        for the EWS research contribution. It shows whether
        the DRUSEN dominance observed in tabular outputs is
        a consistent geometric signal across the actual
        early warning population.

        Unlike plot_cosine_alignments which covers all flagged
        cases (dominated by confirmed disease), this plot
        focuses exclusively on the 32 true-NORMAL cases
        that were flagged by Layer A. These are the cases
        the EWS is designed to surface.

        Args:
            scored_df: fully scored DataFrame with true labels
            filename: output filename

        Returns:
            Path to saved figure
        """
        self._log("  Plot 5: Cosine Alignments — "
                  "Flagged True-NORMAL Cases Only")

        if COL_TRUE_LABEL not in scored_df.columns:
            self._log("  No true labels available. Skipping.")
            return None

        # Filter to flagged true-NORMAL cases only
        flagged_normal = scored_df[
            (scored_df[COL_TRUE_LABEL] == NORMAL_CLASS)
            & (scored_df[COL_LAYER_A_FLAG] == True)
        ]

        n_cases = len(flagged_normal)
        self._log(f"  Flagged true-NORMAL cases: {n_cases}")

        if n_cases == 0:
            self._log("  No flagged NORMAL cases. Skipping.")
            return None

        cosine_cols = [
            f"cosine_{cls}" for cls in DISEASE_CLASSES
            if f"cosine_{cls}" in scored_df.columns
        ]

        if not cosine_cols:
            self._log("  No cosine columns found. Skipping.")
            return None

        n_classes = len(cosine_cols)
        fig, axes = plt.subplots(
            2, 2,
            figsize=(14, 12)
        )
        axes = axes.flatten()

        # Per-class histograms
        for ax, col in zip(axes[:n_classes], cosine_cols):
            cls = col.replace("cosine_", "")
            color = DIRECTION_COLOURS.get(cls, "grey")

            values = pd.to_numeric(
                flagged_normal[col], errors="coerce"
            ).dropna().values

            ax.hist(
                values, bins=15, color=color,
                alpha=0.8, edgecolor="white"
            )
            ax.axvline(
                0.3, color="black", linestyle="--",
                linewidth=1.5,
                label="Phase 1 threshold (0.3)"
            )

            mean_val = values.mean() if len(values) > 0 else 0
            ax.axvline(
                mean_val, color="darkred",
                linestyle="-", linewidth=1.5,
                label=f"Mean ({mean_val:.2f})"
            )

            ax.set_xlabel(f"Cosine to {cls}")
            ax.set_ylabel("Count")
            ax.set_title(f"{cls} Direction")
            ax.legend(fontsize=16)
            ax.set_xlim(-1, 1)

        # Strongest direction bar chart
        ax_last = axes[-1]
        if COL_STRONGEST_DIRECTION in flagged_normal.columns:
            direction_counts = (
                flagged_normal[COL_STRONGEST_DIRECTION]
                .value_counts()
            )
            colours = [
                DIRECTION_COLOURS.get(d, "grey")
                for d in direction_counts.index
            ]
            bars = ax_last.bar(
                direction_counts.index,
                direction_counts.values,
                color=colours,
                edgecolor="white"
            )
            for bar, val in zip(bars, direction_counts.values):
                ax_last.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(val),
                    ha="center", va="bottom", fontsize=14
                )
            ax_last.set_ylabel("Count")
            ax_last.set_title("Strongest Direction\n(flagged NORMAL)")
            ax_last.set_ylim(
                0, direction_counts.max() * 1.3
            )

        fig.suptitle(
            f"Disease-Direction Cosine Alignments\n"
            f"Flagged True-NORMAL Cases Only "
            f"(n={n_cases}) | "
            f"Ref: {self.reference_model_id}",
            fontsize=11, y=1.02
        )



        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=720, bbox_inches="tight")
        plt.close()
        self._log(f"  Saved: {filename}")
        return filepath


    # ----------------------------------------------------------
    # VISUALISATIONS
    # ----------------------------------------------------------

    def plot_mahalanobis_histogram(
        self,
        scored_df: pd.DataFrame,
        band_thresholds: Optional[Dict] = None,
        filename: str = "mahalanobis_histogram.png",
    ) -> Path:
        """
        Plot histogram of Mahalanobis scores with Layer A
        band boundaries overlaid.

        Shows separate distributions for true NORMAL cases
        and disease cases, with band threshold lines.

        Args:
            scored_df: fully scored DataFrame
            band_thresholds: dict with threshold values
                             from AnomalyScorer.band_thresholds
            filename: output filename

        Returns:
            Path to saved figure
        """
        self._log("\n  Plot 1: Mahalanobis Score Histogram")

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle(
            f"Mahalanobis Score Distribution\n"
            f"Ref: {self.reference_model_id} | "
            f"Phase: {self.config.phase}",
            fontsize=22, y=1.02
        )

        scores = scored_df[COL_MAHALANOBIS].values
        has_true_labels = COL_TRUE_LABEL in scored_df.columns

        # Left plot: full score distribution
        ax1 = axes[0]
        if has_true_labels:
            for label in scored_df[COL_TRUE_LABEL].unique():
                mask = scored_df[COL_TRUE_LABEL] == label
                color = (
                    BAND_COLOURS[BAND_CORE_NORMAL]
                    if label == NORMAL_CLASS
                    else DIRECTION_COLOURS.get(label, "grey")
                )
                ax1.hist(
                    scores[mask], bins=50, alpha=0.5,
                    label=label, color=color, density=True
                )
            ax1.legend(fontsize=16)
        else:
            ax1.hist(scores, bins=50, color="#4169E1",
                     alpha=0.7, density=True)

        # Add band threshold lines
        if band_thresholds:
            colours = ["#2E8B57", "#90EE90", "#FFA500"]
            labels = [
                "Core Normal boundary",
                "Extended Normal boundary",
                "Atypical Candidate boundary",
            ]
            keys = [
                "core_normal", "extended_normal",
                "atypical_candidate"
            ]
            for key, col, lbl in zip(keys, colours, labels):
                if key in band_thresholds:
                    ax1.axvline(
                        band_thresholds[key],
                        color=col, linestyle="--",
                        linewidth=1.5, label=lbl
                    )
            ax1.legend(fontsize=16)

        ax1.set_xlabel("Mahalanobis Distance")
        ax1.set_ylabel("Density")
        ax1.set_title("Score Distribution by True Label")
        ax1.set_xlim(left=0)

        # Right plot: NORMAL cases only zoomed in
        ax2 = axes[1]
        if has_true_labels:
            normal_scores = scores[
                scored_df[COL_TRUE_LABEL] == NORMAL_CLASS
            ]
        else:
            normal_scores = scores

        ax2.hist(
            normal_scores, bins=40,
            color=BAND_COLOURS[BAND_CORE_NORMAL],
            alpha=0.7, density=True
        )

        if band_thresholds:
            for key, col, lbl in zip(keys, colours, labels):
                if key in band_thresholds:
                    ax2.axvline(
                        band_thresholds[key],
                        color=col, linestyle="--",
                        linewidth=1.5, label=lbl
                    )
            ax2.legend(fontsize=16)

        ax2.set_xlabel("Mahalanobis Distance")
        ax2.set_ylabel("Density")
        ax2.set_title("NORMAL Cases Only (zoomed)")

        # Add disclaimer if Phase 1

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=720, bbox_inches="tight")
        plt.close()
        self._log(f"  Saved: {filename}")
        return filepath

    def plot_band_distribution(
        self,
        scored_df: pd.DataFrame,
        filename: str = "layer_a_band_distribution.png",
    ) -> Path:
        """
        Plot Layer A band distribution as a horizontal
        stacked bar chart, split by true label.

        Args:
            scored_df: fully scored DataFrame
            filename: output filename

        Returns:
            Path to saved figure
        """
        self._log("  Plot 2: Layer A Band Distribution")

        bands = [
            BAND_CORE_NORMAL, BAND_EXTENDED_NORMAL,
            BAND_ATYPICAL_CANDIDATE, BAND_SUSPICIOUS,
        ]

        has_true_labels = COL_TRUE_LABEL in scored_df.columns

        if has_true_labels:
            labels = sorted(
                scored_df[COL_TRUE_LABEL].unique()
            )
            rows = []
            for label in labels:
                row = {}
                for band in bands:
                    mask = (
                        (scored_df[COL_TRUE_LABEL] == label)
                        & (scored_df[COL_LAYER_A_BAND] == band)
                    )
                    row[band] = int(mask.sum())
                rows.append(row)
            plot_df = pd.DataFrame(rows, index=labels)

        else:
            counts = scored_df[COL_LAYER_A_BAND].value_counts()
            plot_df = pd.DataFrame(
                {band: [counts.get(band, 0)] for band in bands},
                index=["All cases"]
            )

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(plot_df.index))
        width = 0.18
        offsets = np.linspace(
            -width * 1.5, width * 1.5, len(bands)
        )

        for i, band in enumerate(bands):
            values = plot_df[band].values
            bars = ax.bar(
                x + offsets[i], values, width,
                label=band,
                color=BAND_COLOURS[band],
                edgecolor="white", linewidth=0.5
            )
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        str(int(val)),
                        ha="center", va="bottom",
                        fontsize=14
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df.index, fontsize=14)
        ax.set_ylabel("Case Count")
        ax.set_title(
            f"Layer A Band Distribution by True Label\n"
            f"Ref: {self.reference_model_id}"
        )
        ax.legend(
            title="Layer A Band",
            bbox_to_anchor=(1.01, 1),
            loc="upper left", fontsize=9
        )


        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=720, bbox_inches="tight")
        plt.close()
        self._log(f"  Saved: {filename}")
        return filepath

    def plot_cosine_alignments(
        self,
        scored_df: pd.DataFrame,
        filename: str = "cosine_alignment_distributions.png",
    ) -> Path:
        """
        Plot cosine alignment distributions for each
        disease class across all flagged cases.

        Shows how strongly flagged cases point toward
        each disease class direction.

        Args:
            scored_df: fully scored DataFrame
            filename: output filename

        Returns:
            Path to saved figure
        """
        self._log("  Plot 3: Cosine Alignment Distributions")

        flagged = scored_df[
            scored_df[COL_LAYER_A_FLAG] == True
        ]

        cosine_cols = [
            f"cosine_{cls}" for cls in DISEASE_CLASSES
            if f"cosine_{cls}" in scored_df.columns
        ]

        if not cosine_cols:
            self._log("  No cosine columns found. Skipping.")
            return None

        n_classes = len(cosine_cols)
        fig, axes = plt.subplots(
            2, 2, figsize=(14, 12)
        )
        axes = axes.flatten()

        if n_classes == 1:
            axes = [axes]

        for ax, col in zip(axes, cosine_cols):
            cls = col.replace("cosine_", "")
            color = DIRECTION_COLOURS.get(cls, "grey")

            values = pd.to_numeric(
                flagged[col], errors="coerce"
            ).dropna().values

            ax.hist(
                values, bins=30, color=color,
                alpha=0.7, edgecolor="white"
            )
            ax.axvline(
                0.3, color="black", linestyle="--",
                linewidth=1, label="Provisional threshold"
            )
            ax.set_xlabel(f"Cosine Alignment to {cls}")
            ax.set_ylabel("Count")
            ax.set_title(f"{cls} Direction\n"
                         f"(flagged cases, n={len(values):,})")
            ax.legend(fontsize=16)
            ax.set_xlim(-1, 1)

        fig.suptitle(
            f"Disease-Direction Cosine Alignments "
            f"(Layer B Flagged Cases)\n"
            f"Ref: {self.reference_model_id}",
            fontsize=11, y=1.02
        )


        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=720, bbox_inches="tight")
        plt.close()
        self._log(f"  Saved: {filename}")
        return filepath

    def plot_action_distribution(
        self,
        scored_df: pd.DataFrame,
        filename: str = "layer_c_action_distribution.png",
    ) -> Path:
        """
        Plot Layer C action distribution as a pie chart
        and bar chart side by side.

        Args:
            scored_df: fully scored DataFrame
            filename: output filename

        Returns:
            Path to saved figure
        """
        self._log("  Plot 4: Layer C Action Distribution")

        actions = [
            ACTION_SAFE_TO_DISMISS,
            ACTION_ROUTINE_MONITORING,
            ACTION_DEFERRED_REVIEW,
            ACTION_IMMEDIATE_REVIEW,
        ]

        counts = scored_df[COL_LAYER_C_ACTION].value_counts()
        values = [counts.get(a, 0) for a in actions]
        colours = [ACTION_COLOURS[a] for a in actions]
        total = sum(values)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle(
            f"Layer C Action Distribution (n={total:,})\n"
            f"Ref: {self.reference_model_id} | "
            f"Budget: immediate="
            f"{self.config.budget.immediate_review_rate*100:.0f}%"
            f", deferred="
            f"{self.config.budget.deferred_review_rate*100:.0f}%",
            fontsize=10, y=1.02
        )

        # Pie chart
        ax1 = axes[0]
        non_zero = [(v, a, c) for v, a, c
                    in zip(values, actions, colours) if v > 0]
        if non_zero:
            pie_vals, pie_labs, pie_cols = zip(*non_zero)
            wedges, texts, autotexts = ax1.pie(
                pie_vals,
                labels=[
                    a.replace(" ", "\n") for a in pie_labs
                ],
                colors=pie_cols,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 9}
            )
        ax1.set_title("Proportion")

        # Bar chart
        ax2 = axes[1]
        short_labels = [
            a.replace(" Queue", "").replace(" to ", "\nto ")
            for a in actions
        ]
        bars = ax2.bar(
            short_labels, values, color=colours,
            edgecolor="white", linewidth=0.5
        )
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{val:,}\n({val/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=14
            )
        ax2.set_ylabel("Case Count")
        ax2.set_title("Counts")
        ax2.set_ylim(0, max(values) * 1.2)


        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=720, bbox_inches="tight")
        plt.close()
        self._log(f"  Saved: {filename}")
        return filepath

    # ----------------------------------------------------------
    # FULL REPORT RUN
    # ----------------------------------------------------------

    def generate_all(
        self,
        scored_df: pd.DataFrame,
        patient_summaries: Optional[pd.DataFrame] = None,
        band_thresholds: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """
        Generate all reports and visualisations in one call.

        Args:
            scored_df: fully scored DataFrame
            patient_summaries: optional patient summary
                               DataFrame from PatientMonitor
            band_thresholds: optional dict from
                             AnomalyScorer.band_thresholds

        Returns:
            dict mapping report name to file path
        """
        self._log("\n" + "=" * 60)
        self._log("  Generating All EWS Reports")
        self._log("=" * 60)
        self._log(f"  Output directory: {self.output_dir}")
        self._log(f"  Reference model : {self.reference_model_id}")
        self._log(f"  Formula version : {self.formula_version}")
        self._log(f"  Phase           : {self.config.phase}")

        if self.config.is_phase_1():
            self._log(f"\n  {self.config.disclaimer}")

        outputs = {}

        # CSV reports
        outputs["scan_report"] = self.save_scan_report(
            scored_df
        )
        if patient_summaries is not None:
            outputs["patient_summary"] = (
                self.save_patient_summary(patient_summaries)
            )
        outputs["shortlist"] = self.save_shortlist(scored_df)
        outputs["ews_shortlist"] = self.save_ews_shortlist(
            scored_df
        )

        # Visualisations
        self._log("\n  Generating visualisations...")
        outputs["histogram"] = self.plot_mahalanobis_histogram(
            scored_df, band_thresholds
        )
        outputs["band_distribution"] = (
            self.plot_band_distribution(scored_df)
        )
        outputs["cosine_alignments"] = (
            self.plot_cosine_alignments(scored_df)
        )
        
        outputs["action_distribution"] = (
            self.plot_action_distribution(scored_df)
        )
        outputs["normal_cosine"] = (
            self.plot_normal_cosine_alignments(scored_df)
        )

        self._log("\n" + "=" * 60)
        self._log(
            f"  All reports generated: {len(outputs)} files"
        )
        self._log(f"  Location: {self.output_dir}")
        self._log("=" * 60)

        return outputs