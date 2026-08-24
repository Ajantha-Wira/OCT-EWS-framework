# =============================================================
# OCT Early Warning System
# config/defaults.py
#
# Purpose: Global default values used across all modules.
# Change these here and they update everywhere.
# =============================================================


# -------------------------------------------------------------
# CLASS NAMES
# Default disease classes matching the Kermany OCT2017 dataset.
# Override these if using a different dataset.
# -------------------------------------------------------------

DEFAULT_CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN"]

NORMAL_CLASS = "NORMAL"

DISEASE_CLASSES = ["CNV", "DME", "DRUSEN"]


# -------------------------------------------------------------
# COLUMN NAME CONVENTIONS
# Standard column names expected by all modules.
# -------------------------------------------------------------

# Required input columns
COL_SCAN_ID = "scan_id"
COL_PATIENT_ID = "patient_id"
COL_VISIT_ID = "visit_id"
COL_VISIT_DATE = "visit_date"
COL_TRUE_LABEL = "true_label"
COL_PREDICTED_LABEL = "predicted_label"

# Embedding column prefix
# Embedding columns are detected automatically as emb_0, emb_1, etc.
EMBEDDING_COL_PREFIX = "emb_"

# Probability column prefix
# Probability columns are detected as prob_NORMAL, prob_CNV, etc.
PROB_COL_PREFIX = "prob_"

# Output column names produced by the EWS modules
COL_MAHALANOBIS = "mahalanobis_distance"
COL_MAHALANOBIS_SQ = "mahalanobis_sq"
COL_LAYER_A_BAND = "layer_a_band"
COL_LAYER_A_FLAG = "layer_a_flag"
COL_STRONGEST_DIRECTION = "strongest_disease_direction"
COL_LAYER_B_CATEGORY = "layer_b_category"
COL_COMPOSITE_SCORE = "ews_composite_score"
COL_LAYER_C_ACTION = "layer_c_action"
COL_PRIORITY_SCORE = "layer_c_priority_score"


# -------------------------------------------------------------
# LAYER A BAND LABELS
# These are the four output categories from Layer A.
# -------------------------------------------------------------

BAND_CORE_NORMAL = "Core Normal"
BAND_EXTENDED_NORMAL = "Extended Normal"
BAND_ATYPICAL_CANDIDATE = "Atypical Candidate"
BAND_SUSPICIOUS = "Suspicious"

LAYER_A_BANDS_ORDERED = [
    BAND_CORE_NORMAL,
    BAND_EXTENDED_NORMAL,
    BAND_ATYPICAL_CANDIDATE,
    BAND_SUSPICIOUS,
]


# -------------------------------------------------------------
# LAYER B CATEGORY LABELS
# These are the three output categories from Layer B.
# -------------------------------------------------------------

CATEGORY_NON_SPECIFIC = "Non-Specific Atypical"
CATEGORY_PROVISIONAL_BORDERLINE = "Provisional Borderline Disease-Aligned"
CATEGORY_STRONGLY_ALIGNED = "Strongly Disease-Aligned Suspicious"


# -------------------------------------------------------------
# LAYER C ACTION LABELS
# These are the operational outputs from Layer C.
# -------------------------------------------------------------

ACTION_SAFE_TO_DISMISS = "Safe to Dismiss"
ACTION_ROUTINE_MONITORING = "Routine Monitoring"
ACTION_IMMEDIATE_REVIEW = "Immediate Review"
ACTION_DEFERRED_REVIEW = "Deferred Review Queue"
ACTION_LOG_ONLY = "Logged Only"


# -------------------------------------------------------------
# PATIENT MONITORING STATE LABELS
# -------------------------------------------------------------

STATE_SAFE = "Safe"
STATE_ROUTINE_FOLLOW_UP = "Routine Follow-Up"
STATE_SHORT_INTERVAL_REPEAT = "Short-Interval Repeat Scan"
STATE_ESCALATE = "Escalate to Specialist"


# -------------------------------------------------------------
# TREND LABELS
# -------------------------------------------------------------

TREND_IMPROVING = "Improving"
TREND_STABLE = "Stable"
TREND_WORSENING = "Worsening"
TREND_FLUCTUATING = "Fluctuating"


# -------------------------------------------------------------
# REGULARISATION DEFAULT
# Applied to covariance matrix to ensure numerical stability
# in high-dimensional embedding space.
# -------------------------------------------------------------

DEFAULT_REGULARISATION = 1e-6


# -------------------------------------------------------------
# COVARIANCE METHOD OPTIONS
# -------------------------------------------------------------

COV_METHOD_STANDARD = "standard"
COV_METHOD_SHRINKAGE = "shrinkage"
COV_METHOD_ROBUST = "robust"


# -------------------------------------------------------------
# REPORT SETTINGS
# -------------------------------------------------------------

REPORT_FLOAT_PRECISION = 4
REPORT_DATE_FORMAT = "%Y-%m-%d"