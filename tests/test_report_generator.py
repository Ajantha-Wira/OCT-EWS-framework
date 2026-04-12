import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader
from src.reference_builder import ReferenceBuilder
from src.anomaly_scorer import AnomalyScorer
from src.direction_analyzer import DirectionAnalyzer
from src.ews_scorer import EWSScorer, compute_uncertainty_features
from src.patient_monitor import PatientMonitor
from src.report_generator import ReportGenerator
from config import get_default_phase1_config

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
ref_id = ref_metadata["reference_model_id"]

config = get_default_phase1_config()

# Run full pipeline
print("\nRunning full pipeline...")
scorer = AnomalyScorer(ref_model, config, verbose=False)
scored_df = scorer.score_dataframe(
    test_df, reference_df=train_df
)

analyzer = DirectionAnalyzer(
    ref_model, config,
    disease_classes=['CNV', 'DME', 'DRUSEN'],
    verbose=False
)
analyzer.fit_geometry(train_df)
scored_df = analyzer.analyze_flagged(scored_df)

ews = EWSScorer(
    config,
    disease_classes=['CNV', 'DME', 'DRUSEN'],
    reference_model_id=ref_id,
    formula_version="v1",
    verbose=False
)
final_df = ews.run_full_pipeline(scored_df)

# Patient monitor
monitor = PatientMonitor(config=config, verbose=False)
monitor.load_from_dataframe(final_df)
patient_summaries = monitor.get_all_summaries()

print(f"Pipeline complete. {len(final_df)} cases scored.")
print(f"Reference model ID: {ref_id}")

# Generate all reports
reporter = ReportGenerator(
    config=config,
    output_dir=r'C:\Users\Ajant\Documents\OCT_EWS\data\outputs',
    reference_model_id=ref_id,
    formula_version="v1",
    verbose=True
)

outputs = reporter.generate_all(
    scored_df=final_df,
    patient_summaries=patient_summaries,
    band_thresholds=scorer.band_thresholds,
)

# Verify outputs exist
print("\nVerifying outputs:")
for name, path in outputs.items():
    if path is not None:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"  {name:25s}: {path.name} "
              f"({'OK' if exists else 'MISSING'}, "
              f"{size:,} bytes)")

# Check scan report has metadata columns
import pandas as pd
scan_report = pd.read_csv(
    outputs["scan_report"], comment="#", encoding="utf-8"
)
print(f"\nScan report shape: {scan_report.shape}")
print(f"Columns: {list(scan_report.columns[:10])}")

# Check shortlist
shortlist = pd.read_csv(
    outputs["shortlist"], comment="#"
)
print(f"\nShortlist: {len(shortlist)} cases")
print(shortlist[[
    'scan_id', 'true_label', 'predicted_label',
    'mahalanobis_distance', 'layer_a_band',
    'layer_b_category', 'strongest_disease_direction'
]].head(10).to_string())

print("\nAll report generation tests passed.")