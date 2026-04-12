# =============================================================
# OCT Early Warning System
# src/patient_monitor.py
#
# Purpose: Longitudinal Patient Monitoring
#
# Tracks patients across multiple visits. For each new visit,
# compares against prior visits and computes:
#   - Change in Mahalanobis score
#   - Change in disease-direction alignment
#   - Change in uncertainty
#   - Change in composite EWS score
#   - Trend label: improving, stable, worsening, fluctuating
#   - Patient monitoring state
#
# Patient states:
#   Safe                    signed off, no further monitoring
#   Routine Follow-Up       standard interval monitoring
#   Short-Interval Repeat   closer monitoring warranted
#   Escalate to Specialist  urgent clinical attention needed
#
# A patient remains in monitoring until either:
#   1. Signed off as safe (consistent core-normal scores)
#   2. Escalated to specialist for clinical intervention
# =============================================================

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config.phase_config import PhaseConfig
from config.defaults import (
    COL_PATIENT_ID,
    COL_VISIT_ID,
    COL_VISIT_DATE,
    COL_PREDICTED_LABEL,
    COL_MAHALANOBIS,
    COL_LAYER_A_BAND,
    COL_LAYER_A_FLAG,
    COL_LAYER_B_CATEGORY,
    COL_STRONGEST_DIRECTION,
    COL_COMPOSITE_SCORE,
    COL_LAYER_C_ACTION,
    BAND_CORE_NORMAL,
    BAND_EXTENDED_NORMAL,
    BAND_ATYPICAL_CANDIDATE,
    BAND_SUSPICIOUS,
    STATE_SAFE,
    STATE_ROUTINE_FOLLOW_UP,
    STATE_SHORT_INTERVAL_REPEAT,
    STATE_ESCALATE,
    TREND_IMPROVING,
    TREND_STABLE,
    TREND_WORSENING,
    TREND_FLUCTUATING,
    ACTION_IMMEDIATE_REVIEW,
    ACTION_DEFERRED_REVIEW,
    ACTION_SAFE_TO_DISMISS,
    ACTION_ROUTINE_MONITORING,
    NORMAL_CLASS,
)


# -------------------------------------------------------------
# VISIT RECORD
# Stores the EWS assessment for one patient visit.
# -------------------------------------------------------------

@dataclass
class VisitRecord:
    """
    Stores the complete EWS assessment for one patient visit.

    Attributes:
        patient_id:      unique patient identifier
        visit_id:        unique visit identifier
        visit_date:      date of visit as string YYYY-MM-DD
        predicted_label: classifier prediction for this visit
        mahalanobis:     Mahalanobis distance score
        layer_a_band:    Layer A band assignment
        layer_a_flag:    whether case was flagged for Layer B
        layer_b_category: Layer B post-analysis category
        strongest_direction: strongest disease direction
        composite_score: composite EWS score
        layer_c_action:  Layer C action assigned
        notes:           optional clinical notes
    """
    patient_id: str
    visit_id: str
    visit_date: Optional[str]
    predicted_label: Optional[str]
    mahalanobis: float
    layer_a_band: str
    layer_a_flag: bool
    layer_b_category: Optional[str]
    strongest_direction: Optional[str]
    composite_score: float
    layer_c_action: Optional[str]
    notes: str = ""


# -------------------------------------------------------------
# PATIENT RECORD
# Stores the full visit history for one patient.
# -------------------------------------------------------------

@dataclass
class PatientRecord:
    """
    Stores the complete monitoring history for one patient.

    Attributes:
        patient_id:     unique patient identifier
        visits:         list of VisitRecord in chronological order
        current_state:  current monitoring state
        signed_off:     whether patient has been signed off
        escalated:      whether patient has been escalated
    """
    patient_id: str
    visits: List[VisitRecord] = field(default_factory=list)
    current_state: str = STATE_ROUTINE_FOLLOW_UP
    signed_off: bool = False
    escalated: bool = False

    def n_visits(self) -> int:
        return len(self.visits)

    def latest_visit(self) -> Optional[VisitRecord]:
        if not self.visits:
            return None
        return self.visits[-1]

    def first_visit(self) -> Optional[VisitRecord]:
        if not self.visits:
            return None
        return self.visits[0]

    def mahalanobis_history(self) -> List[float]:
        return [v.mahalanobis for v in self.visits]

    def composite_score_history(self) -> List[float]:
        return [v.composite_score for v in self.visits]


# -------------------------------------------------------------
# PATIENT MONITOR
# Manages longitudinal tracking for all patients.
# -------------------------------------------------------------

class PatientMonitor:
    """
    Longitudinal Patient Monitoring.

    Tracks patients across multiple visits, computes
    visit-to-visit changes, assigns trend labels, and
    maintains patient monitoring states.

    Args:
        config: PhaseConfig controlling monitoring rules
        stable_threshold: maximum change in Mahalanobis
                          score to be considered stable
        improving_threshold: minimum decrease in Mahalanobis
                             to be considered improving
        consecutive_normal_for_signoff: number of consecutive
                             Core Normal visits required
                             before a patient can be signed off
        verbose: whether to print progress messages
    """

    def __init__(
        self,
        config: PhaseConfig,
        stable_threshold: float = 1.0,
        improving_threshold: float = 1.5,
        consecutive_normal_for_signoff: int = 3,
        verbose: bool = True,
    ):
        self.config = config
        self.stable_threshold = stable_threshold
        self.improving_threshold = improving_threshold
        self.consecutive_normal_for_signoff = (
            consecutive_normal_for_signoff
        )
        self.verbose = verbose
        self.patients: Dict[str, PatientRecord] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _row_to_visit_record(
        self, row: pd.Series, patient_id: str
    ) -> VisitRecord:
        """Convert a scored DataFrame row to a VisitRecord."""
        return VisitRecord(
            patient_id=patient_id,
            visit_id=str(row.get(COL_VISIT_ID, "")),
            visit_date=str(row.get(COL_VISIT_DATE, "")),
            predicted_label=row.get(COL_PREDICTED_LABEL),
            mahalanobis=float(row.get(COL_MAHALANOBIS, 0)),
            layer_a_band=str(row.get(COL_LAYER_A_BAND, "")),
            layer_a_flag=bool(row.get(COL_LAYER_A_FLAG, False)),
            layer_b_category=row.get(COL_LAYER_B_CATEGORY),
            strongest_direction=row.get(COL_STRONGEST_DIRECTION),
            composite_score=float(
                row.get(COL_COMPOSITE_SCORE, 0)
            ),
            layer_c_action=row.get(COL_LAYER_C_ACTION),
        )

    def add_visit(
        self,
        patient_id: str,
        visit_record: VisitRecord,
    ) -> PatientRecord:
        """
        Add a new visit record for a patient.

        If the patient does not exist, creates a new
        PatientRecord. If they exist, appends the visit
        to their history and updates their monitoring state.

        Args:
            patient_id: unique patient identifier
            visit_record: VisitRecord for this visit

        Returns:
            Updated PatientRecord
        """
        if patient_id not in self.patients:
            self.patients[patient_id] = PatientRecord(
                patient_id=patient_id
            )

        record = self.patients[patient_id]

        # Do not add visits for signed-off or escalated patients
        # without explicit override
        if record.signed_off or record.escalated:
            self._log(f"  Patient {patient_id} is "
                      f"{'signed off' if record.signed_off else 'escalated'}. "
                      f"Visit recorded but state not updated.")

        record.visits.append(visit_record)
        self._update_patient_state(record)
        return record

    def _compute_trend(
        self, record: PatientRecord
    ) -> str:
        """
        Compute the trend label for the most recent visit
        based on the last two visits.

        Improving:    Mahalanobis decreased by more than
                      improving_threshold
        Stable:       Mahalanobis changed by less than
                      stable_threshold in either direction
        Worsening:    Mahalanobis increased by more than
                      improving_threshold
        Fluctuating:  Multiple visits with no consistent
                      direction across last three visits

        Returns:
            Trend label string
        """
        history = record.mahalanobis_history()

        if len(history) < 2:
            return TREND_STABLE

        latest = history[-1]
        previous = history[-2]
        change = latest - previous

        if len(history) >= 3:
            # Check for fluctuation across last three visits
            prev_change = history[-2] - history[-3]
            if (change > 0 and prev_change < 0) or (
                change < 0 and prev_change > 0
            ):
                if abs(change) > self.stable_threshold:
                    return TREND_FLUCTUATING

        if change < -self.improving_threshold:
            return TREND_IMPROVING
        elif change > self.improving_threshold:
            return TREND_WORSENING
        else:
            return TREND_STABLE

    def _count_consecutive_core_normal(
        self, record: PatientRecord
    ) -> int:
        """
        Count consecutive Core Normal visits from the most
        recent visit backwards.
        """
        count = 0
        for visit in reversed(record.visits):
            if visit.layer_a_band == BAND_CORE_NORMAL:
                count += 1
            else:
                break
        return count

    def _update_patient_state(
        self, record: PatientRecord
    ) -> None:
        """
        Update the patient monitoring state based on
        their visit history.

        State transition rules:
            Escalate:
                Latest visit has Immediate Review action
                OR latest Mahalanobis score is Suspicious
                AND trend is Worsening

            Short-Interval Repeat:
                Latest visit is Atypical Candidate or Suspicious
                OR trend is Worsening

            Safe (sign off):
                N consecutive Core Normal visits where N is
                consecutive_normal_for_signoff

            Routine Follow-Up:
                All other cases
        """
        if not record.visits:
            return

        latest = record.latest_visit()
        trend = self._compute_trend(record)

        # Check for escalation
        if (
            latest.layer_c_action == ACTION_IMMEDIATE_REVIEW
            or (
                latest.layer_a_band == BAND_SUSPICIOUS
                and trend == TREND_WORSENING
            )
        ):
            record.current_state = STATE_ESCALATE
            record.escalated = True
            return

        # Check for sign-off
        consecutive_normal = self._count_consecutive_core_normal(
            record
        )
        if (
            consecutive_normal >= self.consecutive_normal_for_signoff
            and not record.escalated
        ):
            record.current_state = STATE_SAFE
            record.signed_off = True
            return

        # Check for short-interval repeat
        if (
            latest.layer_a_band in [
                BAND_ATYPICAL_CANDIDATE, BAND_SUSPICIOUS
            ]
            or trend == TREND_WORSENING
        ):
            record.current_state = STATE_SHORT_INTERVAL_REPEAT
            return

        # Default: routine follow-up
        record.current_state = STATE_ROUTINE_FOLLOW_UP

    def load_from_dataframe(
        self,
        scored_df: pd.DataFrame,
        patient_id_col: str = COL_PATIENT_ID,
    ) -> None:
        """
        Load patient visit records from a fully scored
        DataFrame.

        If patient_id column is not present or all null,
        generates synthetic patient IDs from scan_id for
        single-visit datasets like the dissertation test set.

        Args:
            scored_df: fully scored DataFrame
            patient_id_col: column name for patient IDs
        """
        self._log("\n" + "=" * 60)
        self._log("  Loading Patient Records")
        self._log("=" * 60)

        df = scored_df.copy()

        # Handle missing patient IDs
        if (
            patient_id_col not in df.columns
            or df[patient_id_col].isnull().all()
        ):
            self._log(
                "  No patient_id column found. "
                "Generating synthetic patient IDs from scan_id."
            )
            df[patient_id_col] = df["scan_id"].apply(
                lambda x: f"patient_{x}"
            )

        # Sort by patient then visit date if available
        sort_cols = [patient_id_col]
        if (
            COL_VISIT_DATE in df.columns
            and not df[COL_VISIT_DATE].isnull().all()
        ):
            sort_cols.append(COL_VISIT_DATE)
            df = df.sort_values(sort_cols)

        for _, row in df.iterrows():
            patient_id = str(row[patient_id_col])
            visit = self._row_to_visit_record(row, patient_id)
            self.add_visit(patient_id, visit)

        self._log(f"\n  Patients loaded: {len(self.patients):,}")
        self._log(f"  Total visits    : "
                  f"{sum(p.n_visits() for p in self.patients.values()):,}")

    def get_patient_summary(
        self, patient_id: str
    ) -> Optional[Dict]:
        """
        Return a summary dictionary for one patient.

        Args:
            patient_id: patient identifier

        Returns:
            dict with patient summary, or None if not found
        """
        if patient_id not in self.patients:
            return None

        record = self.patients[patient_id]
        latest = record.latest_visit()
        first = record.first_visit()
        trend = self._compute_trend(record)

        # Compute changes from first to latest visit
        mah_change = None
        score_change = None
        if record.n_visits() >= 2:
            mah_change = (
                latest.mahalanobis - first.mahalanobis
            )
            score_change = (
                latest.composite_score - first.composite_score
            )

        return {
            "patient_id": patient_id,
            "n_visits": record.n_visits(),
            "current_state": record.current_state,
            "signed_off": record.signed_off,
            "escalated": record.escalated,
            "trend": trend,
            "first_visit_date": first.visit_date if first else None,
            "latest_visit_date": (
                latest.visit_date if latest else None
            ),
            "first_mahalanobis": (
                first.mahalanobis if first else None
            ),
            "latest_mahalanobis": (
                latest.mahalanobis if latest else None
            ),
            "mahalanobis_change": mah_change,
            "first_layer_a_band": (
                first.layer_a_band if first else None
            ),
            "latest_layer_a_band": (
                latest.layer_a_band if latest else None
            ),
            "latest_layer_b_category": (
                latest.layer_b_category if latest else None
            ),
            "strongest_direction": (
                latest.strongest_direction if latest else None
            ),
            "latest_composite_score": (
                latest.composite_score if latest else None
            ),
            "composite_score_change": score_change,
            "latest_action": (
                latest.layer_c_action if latest else None
            ),
        }

    def get_all_summaries(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per patient
        summarising their current monitoring status.

        Returns:
            DataFrame with patient summaries
        """
        summaries = []
        for patient_id in self.patients:
            summary = self.get_patient_summary(patient_id)
            if summary:
                summaries.append(summary)

        if not summaries:
            return pd.DataFrame()

        return pd.DataFrame(summaries)

    def get_state_counts(self) -> Dict[str, int]:
        """
        Return count of patients in each monitoring state.
        """
        counts = {
            STATE_SAFE: 0,
            STATE_ROUTINE_FOLLOW_UP: 0,
            STATE_SHORT_INTERVAL_REPEAT: 0,
            STATE_ESCALATE: 0,
        }
        for record in self.patients.values():
            state = record.current_state
            if state in counts:
                counts[state] += 1
        return counts

    def get_patients_by_state(
        self, state: str
    ) -> List[PatientRecord]:
        """Return list of PatientRecords in a given state."""
        return [
            r for r in self.patients.values()
            if r.current_state == state
        ]

    def print_monitoring_summary(self) -> None:
        """Print a summary of all patient monitoring states."""
        counts = self.get_state_counts()
        total = len(self.patients)

        print("\n" + "=" * 60)
        print("  Patient Monitoring Summary")
        print("=" * 60)
        print(f"  Total patients tracked: {total:,}")
        print()
        for state, count in counts.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {state:35s}: "
                  f"{count:5,} ({pct:5.1f}%)")
        print("=" * 60)

        if self.config.is_phase_1():
            print(f"\n  {self.config.disclaimer}")