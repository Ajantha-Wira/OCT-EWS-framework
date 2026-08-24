import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_manager import NpyLoader
from src.reference_builder import ReferenceBuilder
from src.anomaly_scorer import AnomalyScorer
from src.direction_analyzer import DirectionAnalyzer
from src.ews_scorer import EWSScorer, compute_uncertainty_features
from src.patient_monitor import PatientMonitor, VisitRecord
from config import get_default_phase1_config
from config.defaults import (
    STATE_SAFE, STATE_ROUTINE_FOLLOW_UP,
    STATE_SHORT_INTERVAL_REPEAT, STATE_ESCALATE,
    TREND_IMPROVING, TREND_STABLE,
    TREND_WORSENING, TREND_FLUCTUATING,
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

# Run full pipeline silently
scorer = AnomalyScorer(ref_model, config, verbose=False)
scored_df = scorer.score_dataframe(test_df, reference_df=train_df)

analyzer = DirectionAnalyzer(ref_model, config,
    disease_classes=['CNV', 'DME', 'DRUSEN'], verbose=False)
analyzer.fit_geometry(train_df)
scored_df = analyzer.analyze_flagged(scored_df)

ews = EWSScorer(config, disease_classes=['CNV','DME','DRUSEN'],
    verbose=False)
final_df = ews.run_full_pipeline(scored_df)

print(f"Pipeline complete. {len(final_df)} cases scored.")

# Test 1: Load single-visit data into monitor
print("\nTest 1: Load test set into patient monitor")
monitor = PatientMonitor(
    config=config,
    stable_threshold=1.0,
    improving_threshold=1.5,
    consecutive_normal_for_signoff=3,
    verbose=True
)

monitor.load_from_dataframe(final_df)
monitor.print_monitoring_summary()

# Test 2: Get all summaries
print("\nTest 2: Patient summary DataFrame")
summaries = monitor.get_all_summaries()
print(f"Summary shape: {summaries.shape}")
print(summaries[[
    'patient_id', 'current_state', 'trend',
    'latest_mahalanobis', 'latest_layer_a_band',
    'latest_action'
]].head(10).to_string())

# Test 3: State counts
print("\nTest 3: State counts")
counts = monitor.get_state_counts()
for state, count in counts.items():
    print(f"  {state:35s}: {count:,}")

# Test 4: Simulate longitudinal monitoring
# Create a patient with three visits showing worsening trend
print("\nTest 4: Simulated longitudinal patient")
monitor2 = PatientMonitor(config=config, verbose=True)

visit1 = VisitRecord(
    patient_id="sim_patient_001",
    visit_id="v1",
    visit_date="2025-01-15",
    predicted_label="NORMAL",
    mahalanobis=9.2,
    layer_a_band="Extended Normal",
    layer_a_flag=False,
    layer_b_category=None,
    strongest_direction=None,
    composite_score=0.0,
    layer_c_action="Routine Monitoring",
    notes="First visit. Routine screening."
)

visit2 = VisitRecord(
    patient_id="sim_patient_001",
    visit_id="v2",
    visit_date="2025-04-15",
    predicted_label="NORMAL",
    mahalanobis=11.8,
    layer_a_band="Atypical Candidate",
    layer_a_flag=True,
    layer_b_category="Provisional Borderline Disease-Aligned",
    strongest_direction="DRUSEN",
    composite_score=0.45,
    layer_c_action="Deferred Review Queue",
    notes="Score increased. DRUSEN direction emerging."
)

visit3 = VisitRecord(
    patient_id="sim_patient_001",
    visit_id="v3",
    visit_date="2025-07-15",
    predicted_label="NORMAL",
    mahalanobis=14.3,
    layer_a_band="Suspicious",
    layer_a_flag=True,
    layer_b_category="Strongly Disease-Aligned Suspicious",
    strongest_direction="DRUSEN",
    composite_score=1.2,
    layer_c_action="Immediate Review",
    notes="Score continues to rise. Strong DRUSEN alignment."
)

monitor2.add_visit("sim_patient_001", visit1)
print(f"  After visit 1: state = "
      f"{monitor2.patients['sim_patient_001'].current_state}")

monitor2.add_visit("sim_patient_001", visit2)
print(f"  After visit 2: state = "
      f"{monitor2.patients['sim_patient_001'].current_state}")

monitor2.add_visit("sim_patient_001", visit3)
record = monitor2.patients["sim_patient_001"]
print(f"  After visit 3: state = {record.current_state}")
print(f"  Escalated: {record.escalated}")
print(f"  Trend: {monitor2._compute_trend(record)}")

# Test 5: Simulate improving patient for sign-off
print("\nTest 5: Simulated improving patient sign-off")
monitor3 = PatientMonitor(
    config=config,
    consecutive_normal_for_signoff=3,
    verbose=False
)

for i, mah in enumerate([13.0, 11.0, 9.0, 8.5, 8.2]):
    v = VisitRecord(
        patient_id="sim_patient_002",
        visit_id=f"v{i+1}",
        visit_date=f"2025-0{i+1}-15",
        predicted_label="NORMAL",
        mahalanobis=mah,
        layer_a_band="Core Normal" if mah < 9.51 else (
            "Extended Normal" if mah < 10.73 else
            "Atypical Candidate" if mah < 12.54 else
            "Suspicious"
        ),
        layer_a_flag=mah > 10.73,
        layer_b_category=None,
        strongest_direction=None,
        composite_score=0.0,
        layer_c_action="Routine Monitoring",
    )
    monitor3.add_visit("sim_patient_002", v)
    state = monitor3.patients["sim_patient_002"].current_state
    print(f"  Visit {i+1} mah={mah:.1f}: state={state}")

final_record = monitor3.patients["sim_patient_002"]
print(f"  Signed off: {final_record.signed_off}")

# Assertions
assert len(monitor.patients) == 968
assert summaries.shape[0] == 968
print("\nAll assertions passed.")
print("\nPatient monitor complete.")