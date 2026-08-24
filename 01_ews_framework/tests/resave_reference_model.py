import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader
from src.reference_builder import ReferenceBuilder

loader = NpyLoader(
    embeddings_root=r'C:\Users\Ajant\Documents\OCT_EWS\data\raw',
    class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']
)

print("Loading training data...")
train_df = loader.load_train()

builder = ReferenceBuilder(
    regularisation=1e-6,
    method='standard',
    verbose=True
)

builder.fit(train_df)

ref_id = builder.save(
    r'C:\Users\Ajant\Documents\OCT_EWS\data\processed',
    notes='Initial standard empirical covariance, full 2048D, no PCA, no core-normal refinement.'
)

print(f"\nSaved with ID: {ref_id}")

# Verify reload
model, metadata = ReferenceBuilder.load(
    r'C:\Users\Ajant\Documents\OCT_EWS\data\processed'
)

print(f"\nMetadata fields:")
for key, value in metadata.items():
    print(f"  {key:30s}: {value}")