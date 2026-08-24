# =============================================================
# OCT Early Warning System
# config/__init__.py
#
# Makes config a Python package.
# Exposes the most commonly used config objects directly.
# =============================================================

from config.phase_config import (
    PhaseConfig,
    AtypicalityConfig,
    DirectionConfig,
    BudgetConfig,
    get_default_phase1_config,
    get_phase2_config_template,
)

from config.defaults import (
    DEFAULT_CLASSES,
    NORMAL_CLASS,
    DISEASE_CLASSES,
    BAND_CORE_NORMAL,
    BAND_EXTENDED_NORMAL,
    BAND_ATYPICAL_CANDIDATE,
    BAND_SUSPICIOUS,
    DEFAULT_REGULARISATION,
)

__all__ = [
    "PhaseConfig",
    "AtypicalityConfig",
    "DirectionConfig",
    "BudgetConfig",
    "get_default_phase1_config",
    "get_phase2_config_template",
    "DEFAULT_CLASSES",
    "NORMAL_CLASS",
    "DISEASE_CLASSES",
    "BAND_CORE_NORMAL",
    "BAND_EXTENDED_NORMAL",
    "BAND_ATYPICAL_CANDIDATE",
    "BAND_SUSPICIOUS",
    "DEFAULT_REGULARISATION",
]