# =============================================================
# OCT Early Warning System
# config/phase_config.py
#
# Purpose: Central configuration for Phase 1 and Phase 2 modes.
# All modules import from here. Never hardcode thresholds
# anywhere else in the project.
# =============================================================

from dataclasses import dataclass, field
from typing import Dict


# -------------------------------------------------------------
# ATYPICALITY CONFIG (Layer A)
# Controls how Mahalanobis scores are banded into risk zones.
# -------------------------------------------------------------

@dataclass
class AtypicalityConfig:
    """
    Controls Layer A: how atypical is this case relative
    to the normal embedding distribution?

    method options:
        percentile  - use empirical percentiles of normal scores
        robust      - use median and MAD of normal scores
        chisquare   - use chi-square approximation
    """
    method: str = "percentile"
    core_normal_pct: float = 50.0
    extended_normal_pct: float = 75.0
    atypical_candidate_pct: float = 90.0
    # Phase 2 overrides the above with fixed validated values


# -------------------------------------------------------------
# DIRECTION CONFIG (Layer B)
# Controls per-class direction alignment thresholds.
# -------------------------------------------------------------

@dataclass
class DirectionConfig:
    """
    Controls Layer B thresholds for one disease class.

    alignment_threshold  - minimum cosine similarity to flag
    projection_threshold - minimum projection magnitude to flag
    urgency              - high, moderate, or low
    """
    alignment_threshold: float = 0.0
    projection_threshold: float = 0.0
    urgency: str = "moderate"


# -------------------------------------------------------------
# BUDGET CONFIG (Layer C)
# Controls operational review capacity. This is a workflow
# tool, not a detection rule.
# -------------------------------------------------------------

@dataclass
class BudgetConfig:
    """
    Controls Layer C: how many cases the clinical workflow
    can handle per screening cycle.

    immediate_review_rate - top fraction escalated immediately
    deferred_review_rate  - next fraction placed in queue
    review_cycle_days     - days before deferred cases reviewed
    overflow_action       - queue, escalate, or log-only
    """
    immediate_review_rate: float = 0.05
    deferred_review_rate: float = 0.10
    review_cycle_days: int = 30
    overflow_action: str = "queue"


# -------------------------------------------------------------
# PHASE CONFIG (master config object)
# This is what every module imports and uses.
# -------------------------------------------------------------

@dataclass
class PhaseConfig:
    """
    Master configuration object for the EWS system.

    phase options:
        phase_1_preliminary - research mode, provisional thresholds
        phase_2_validated   - operational mode, fixed thresholds

    Every Phase 1 output must carry the disclaimer string.
    """
    phase: str = "phase_1_preliminary"
    atypicality: AtypicalityConfig = field(
        default_factory=AtypicalityConfig
    )
    direction_thresholds: Dict[str, DirectionConfig] = field(
        default_factory=dict
    )
    budget: BudgetConfig = field(
        default_factory=BudgetConfig
    )
    disclaimer: str = (
        "PRELIMINARY RESEARCH OUTPUT. "
        "Thresholds are not clinically validated. "
        "Not for clinical decision making."
    )

    def is_phase_1(self) -> bool:
        return self.phase == "phase_1_preliminary"

    def is_phase_2(self) -> bool:
        return self.phase == "phase_2_validated"

    def get_disclaimer(self) -> str:
        """Returns disclaimer for Phase 1, empty string for Phase 2."""
        if self.is_phase_1():
            return self.disclaimer
        return ""


# -------------------------------------------------------------
# DEFAULT PHASE 1 CONFIG
# Ready to use out of the box for research mode.
# -------------------------------------------------------------

def get_default_phase1_config() -> PhaseConfig:
    """
    Returns a ready-to-use Phase 1 config with default
    direction thresholds for CNV, DME, and DRUSEN.
    These are provisional research values derived from
    the reference distribution, not clinical cutoffs.
    """
    return PhaseConfig(
        phase="phase_1_preliminary",
        atypicality=AtypicalityConfig(
            method="percentile",
            core_normal_pct=50.0,
            extended_normal_pct=75.0,
            atypical_candidate_pct=90.0
        ),
        direction_thresholds={
            "CNV": DirectionConfig(
                alignment_threshold=0.0,
                projection_threshold=0.0,
                urgency="high"
            ),
            "DME": DirectionConfig(
                alignment_threshold=0.0,
                projection_threshold=0.0,
                urgency="high"
            ),
            "DRUSEN": DirectionConfig(
                alignment_threshold=0.0,
                projection_threshold=0.0,
                urgency="moderate"
            ),
        },
        budget=BudgetConfig(
            immediate_review_rate=0.05,
            deferred_review_rate=0.10,
            review_cycle_days=30,
            overflow_action="queue"
        )
    )


# -------------------------------------------------------------
# DEFAULT PHASE 2 CONFIG TEMPLATE
# Thresholds must be replaced with clinically validated values
# before use. This is a structural placeholder only.
# -------------------------------------------------------------

def get_phase2_config_template() -> PhaseConfig:
    """
    Returns a Phase 2 config template.
    Direction thresholds must be replaced with values
    derived from clinical trial evidence before use.
    """
    return PhaseConfig(
        phase="phase_2_validated",
        atypicality=AtypicalityConfig(
            method="percentile",
            core_normal_pct=50.0,
            extended_normal_pct=75.0,
            atypical_candidate_pct=90.0
        ),
        direction_thresholds={
            "CNV": DirectionConfig(
                alignment_threshold=0.65,
                projection_threshold=0.40,
                urgency="high"
            ),
            "DME": DirectionConfig(
                alignment_threshold=0.60,
                projection_threshold=0.35,
                urgency="high"
            ),
            "DRUSEN": DirectionConfig(
                alignment_threshold=0.55,
                projection_threshold=0.30,
                urgency="moderate"
            ),
        },
        budget=BudgetConfig(
            immediate_review_rate=0.05,
            deferred_review_rate=0.10,
            review_cycle_days=30,
            overflow_action="queue"
        )
    )