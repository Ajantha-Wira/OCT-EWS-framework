import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader
from src.reference_builder import ReferenceBuilder
from src.anomaly_scorer import AnomalyScorer
from src.direction_analyzer import DirectionAnalyzer
from config import get_default_phase1_config
from config.defaults import (
    COL_LAYER_A_FLAG,
    COL_STRONGEST_DIRECTION,
    COL_LAYER_B_CATEGORY,
    CATEGORY_NON_SPECIFIC,
    CATEGORY_PROVISIONAL_BORDERLINE,
    CATEGORY_STRONGLY_ALIGNED,
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

# Get config
config = get_default_phase1_config()

# Layer A scoring
scorer = AnomalyScorer(
    reference_model=ref_model,
    config=config,
    verbose=True
)

print("\nRunning Layer A...")
scored_df = scorer.score_dataframe(
    df=test_df,
    reference_df=train_df
)

# Layer B direction analysis
analyzer = DirectionAnalyzer(
    reference_model=ref_model,
    config=config,
    disease_classes=['CNV', 'DME', 'DRUSEN'],
    verbose=True
)

print("\nFitting disease geometry...")
geometry = analyzer.fit_geometry(train_df)

print("\nRunning Layer B...")
result_df = analyzer.analyze_flagged(scored_df)

# Print sample of flagged cases with Layer B results
flagged = result_df[result_df[COL_LAYER_A_FLAG] == True]

print("\nSample flagged cases with Layer B results:")
print(flagged[[
    'true_label',
    'predicted_label',
    'mahalanobis_distance',
    'layer_a_band',
    COL_STRONGEST_DIRECTION,
    COL_LAYER_B_CATEGORY,
    'cosine_CNV',
    'cosine_DME',
    'cosine_DRUSEN',
]].head(15).to_string())

# NORMAL cases that were flagged
print("\nFlagged true-NORMAL cases:")
normal_flagged = flagged[flagged['true_label'] == 'NORMAL']
print(normal_flagged[[
    'scan_id',
    'mahalanobis_distance',
    'layer_a_band',
    COL_STRONGEST_DIRECTION,
    COL_LAYER_B_CATEGORY,
]].to_string())

# Category breakdown
print(f"\nLayer B category breakdown for all flagged cases:")
print(result_df[result_df[COL_LAYER_A_FLAG] == True][
    COL_LAYER_B_CATEGORY
].value_counts())

# Assertions
assert COL_STRONGEST_DIRECTION in result_df.columns
assert COL_LAYER_B_CATEGORY in result_df.columns
assert len(result_df) == 968
non_flagged = result_df[result_df[COL_LAYER_A_FLAG] == False]
assert non_flagged[COL_LAYER_B_CATEGORY].isnull().all()
print("\nAll assertions passed.")