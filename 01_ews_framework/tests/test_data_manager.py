import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader, DataValidator

loader = NpyLoader(
    embeddings_root=r'C:\Users\Ajant\Documents\OCT_EWS\data\raw',
    class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']
)

# Load all splits
splits = loader.load_all()

# Summary
for name, df in splits.items():
    print(f"\n  {name}: {df.shape}")
    print(f"  Classes: {df['true_label'].value_counts().to_dict()}")

# Validate test split
print()
validator = DataValidator(class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL'])
validator.validate(splits['test'], split='test')

# Test helper functions
from src.data_manager import (
    detect_embedding_cols,
    get_embeddings_array,
    get_probabilities_array,
    filter_normal_predicted
)

test_df = splits['test']
emb_cols = detect_embedding_cols(test_df)
emb_array = get_embeddings_array(test_df)
prob_array = get_probabilities_array(test_df)

print(f"\n  Embedding array shape : {emb_array.shape}")
print(f"  Probability array shape: {prob_array.shape}")

normal_df = filter_normal_predicted(test_df)
print(f"\n  NORMAL-predicted in test: {len(normal_df)}")