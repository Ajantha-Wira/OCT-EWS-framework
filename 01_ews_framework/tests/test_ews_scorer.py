import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader
from src.reference_builder import ReferenceBuilder
from src.anomaly_scorer import AnomalyScorer
from src.direction_analyzer import DirectionAnalyzer
from src.ews_scorer import EWSScorer, compute_uncertainty_features
from config import get_default_phase1_config
from config.defaults import (
    COL_LAYER_A_FLAG,
    COL_LAYER_B_CATEGORY,
    COL_COMPOSITE_SCORE,
    COL_LAYER_C_ACTION,
    COL_PRIORITY_SCORE,
    ACTION_IMMEDIATE_REVIEW,
    ACTION_DEFERRED_REVIEW,
    ACTION_SAFE_TO_DISMISS,
    ACTION_ROUTINE_MONITORING,
)

# Load data
loader = NpyLoader(
    embeddings_root=r'C:\Users\Ajant\Documents\OCT_EWS\data\raw',
    class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']
)

print("Loading data...")
train_df = loader.load_train()
test_df = loader.load_test()

# Load reference model
ref_model, ref_metadata = ReferenceBuilder.load(
    r'C:\Users\Ajant\Documents\OCT_EWS\data\processed'
)

config = get_default_phase1_config()

# Layer A
scorer = AnomalyScorer(
    reference_model=ref_model,
    config=config,
    verbose=False
)
scored_df = scorer.score_dataframe(
    df=test_df,
    reference_df=train_df
)

# Layer B
analyzer = DirectionAnalyzer(
    reference_model=ref_model,
    config=config,
    disease_classes=['CNV', 'DME', 'DRUSEN'],
    verbose=False
)
analyzer.fit_geometry(train_df)
scored_df = analyzer.analyze_flagged(scored_df)

# Layer C composite scoring
ews = EWSScorer(
    config=config,
    disease_classes=['CNV', 'DME', 'DRUSEN'],
    verbose=True
)

print("\nRunning full EWS pipeline...")
final_df = ews.run_full_pipeline(scored_df)

# Print action distribution
print("\nFinal action distribution:")
print(final_df[COL_LAYER_C_ACTION].value_counts())

# Show top immediate review cases
immediate = final_df[
    final_df[COL_LAYER_C_ACTION] == ACTION_IMMEDIATE_REVIEW
].sort_values(COL_PRIORITY_SCORE, ascending=False)

print(f"\nTop 15 Immediate Review cases:")
print(immediate[[
    'true_label',
    'predicted_label',
    'mahalanobis_distance',
    'layer_a_band',
    'strongest_disease_direction',
    'layer_b_category',
    COL_COMPOSITE_SCORE,
    COL_PRIORITY_SCORE,
]].head(15).to_string())

# Show flagged true-NORMAL cases with actions
print("\nFlagged true-NORMAL cases with final actions:")
normal_flagged = final_df[
    (final_df['true_label'] == 'NORMAL') &
    (final_df[COL_LAYER_A_FLAG] == True)
]
print(normal_flagged[[
    'scan_id',
    'mahalanobis_distance',
    'layer_a_band',
    'strongest_disease_direction',
    'layer_b_category',
    COL_COMPOSITE_SCORE,
    COL_LAYER_C_ACTION,
]].to_string())

# Assertions
assert COL_COMPOSITE_SCORE in final_df.columns
assert COL_LAYER_C_ACTION in final_df.columns
assert COL_PRIORITY_SCORE in final_df.columns
assert final_df[COL_LAYER_C_ACTION].isnull().sum() == 0
assert len(final_df) == 968
print("\nAll assertions passed.")
print("\nFull three-layer EWS pipeline complete.")