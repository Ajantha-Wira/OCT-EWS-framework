import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader
from src.reference_builder import ReferenceBuilder

# Load training data
loader = NpyLoader(
    embeddings_root=r'C:\Users\Ajant\Documents\OCT_EWS\data\raw',
    class_names=['CNV', 'DME', 'DRUSEN', 'NORMAL']
)

print("Loading training data...")
train_df = loader.load_train()

# Build standard reference model
print("\nTest 1: Standard reference model")
builder = ReferenceBuilder(
    regularisation=1e-6,
    method='standard',
    verbose=True
)

model = builder.fit(train_df)

# Quick sanity checks
assert model.mu.shape == (2048,), f"Expected (2048,) got {model.mu.shape}"
assert model.sigma.shape == (2048, 2048), "Sigma shape wrong"
assert model.sigma_inv.shape == (2048, 2048), "Sigma_inv shape wrong"
assert model.n_samples == 19275, f"Expected 19275 got {model.n_samples}"
assert not model.core_normal_refined
print("\n  All assertions passed.")

# Test save and load
print("\nTest 2: Save and reload")
builder.save(r'C:\Users\Ajant\Documents\OCT_EWS\data\processed')

loaded_model = ReferenceBuilder.load(
    r'C:\Users\Ajant\Documents\OCT_EWS\data\processed'
)
assert loaded_model.mu.shape == (2048,)
print("  Save and load working correctly.")

# Test core-normal refinement
print("\nTest 3: Core-normal refinement")
builder2 = ReferenceBuilder(
    regularisation=1e-6,
    method='standard',
    verbose=True
)

refined_model = builder2.fit_with_core_refinement(
    train_df,
    core_percentile=50.0
)

assert refined_model.core_normal_refined
assert refined_model.core_normal_n > 0
print("\n  Core-normal refinement working correctly.")
print(f"  Core subset: {refined_model.core_normal_n:,} samples")

print("\n" + "=" * 60)
print("  All reference builder tests passed.")
print("=" * 60)