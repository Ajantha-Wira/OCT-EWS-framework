import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader
from src.reference_builder import ReferenceBuilder
from src.anomaly_scorer import AnomalyScorer
from config import get_default_phase1_config
from config.defaults import (
    COL_MAHALANOBIS, COL_LAYER_A_BAND, COL_LAYER_A_FLAG,
    BAND_ATYPICAL_CANDIDATE, BAND_SUSPICIOUS
)

# Load data
loader = NpyLoader(
    embeddings_root=r'C:\Users\Ajant\Documents\OCT_EWS\data\raw',
    class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']
)

print("Loading data...")
train_df = loader.load_train()
test_df = loader.load_test()

# Load saved reference model
from src.reference_builder import ReferenceBuilder
ref_model, ref_metadata = ReferenceBuilder.load(
    r'C:\Users\Ajant\Documents\OCT_EWS\data\processed'
)

# Get Phase 1 config
config = get_default_phase1_config()

# Build scorer
scorer = AnomalyScorer(
    reference_model=ref_model,
    config=config,
    verbose=True
)

# Score test set using train normals for band fitting
print("\nScoring test set...")
scored_df = scorer.score_dataframe(
    df=test_df,
    reference_df=train_df
)

# Print results
print("\nSample results:")
print(scored_df[[
    'scan_id', 'true_label', 'predicted_label',
    COL_MAHALANOBIS, COL_LAYER_A_BAND, COL_LAYER_A_FLAG
]].head(10).to_string())

# Band summary
print("\nBand summary:")
summary = scorer.get_band_summary(scored_df)
print(summary.to_string())

# Check the NORMAL class specifically
print("\nNORMAL cases band distribution:")
normal_cases = scored_df[scored_df['true_label'] == 'NORMAL']
print(normal_cases[COL_LAYER_A_BAND].value_counts())

# Check flagged cases
flagged = scored_df[scored_df[COL_LAYER_A_FLAG] == True]
print(f"\nTotal flagged for Layer B: {len(flagged)}")
print(f"Of which true label NORMAL: "
      f"{(flagged['true_label'] == 'NORMAL').sum()}")

# Assertions
assert COL_MAHALANOBIS in scored_df.columns
assert COL_LAYER_A_BAND in scored_df.columns
assert COL_LAYER_A_FLAG in scored_df.columns
assert len(scored_df) == 968
print("\nAll assertions passed.")