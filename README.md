# OCT Early Warning System (OCT-EWS)

An embedding-based post-classification safety layer for stratifying NORMAL predictions in automated OCT screening by their position within the learned representation space.

Most classifier evaluation stops at the predicted label. This project asks a different question: for scans a classifier labels NORMAL, where does each one actually sit within the model's learned NORMAL representation, and does that position mean anything reproducible across different models. The repository is organised around two related pieces of work addressing this, in sequence.

- **`01_ews_framework/`** — the original three-layer EWS implementation and its local, single-model experiments.
- **`02_reproducibility_study/`** — a cross-architecture reproducibility investigation of the same underlying geometry, built directly on questions the first piece of work raised but did not test.

---

## 01_ews_framework

The original EWS framework: *An Embedding-Based Post-Classification Safety Layer for Identifying Unsafe Normal Predictions in OCT Screening.*

The EWS operates as a post-classification module alongside a trained deep learning classifier. It identifies structurally atypical NORMAL predictions by analysing representation-space geometry, rather than relying on output probabilities alone.

**Framework — three sequential layers:**
- **Layer A** — Mahalanobis distance-based atypicality detection
- **Layer B** — Disease-direction geometry analysis
- **Layer C** — Clinical prioritisation and workflow decisions

**Phase design:**
- **Phase 1** — Empirical thresholds (current implementation)
- **Phase 2** — Clinically validated thresholds (future work)

> PRELIMINARY RESEARCH OUTPUT. Thresholds are not clinically validated. Not for clinical decision-making.

This is an earlier, exploratory stage of the work, evaluated locally on a single set of trained classifiers. Its results provided the base observations motivating `02_reproducibility_study/`.

```
01_ews_framework/
├── config/          Phase configuration and defaults
├── src/             Core modules
│   ├── data_manager.py
│   ├── reference_builder.py
│   ├── anomaly_scorer.py
│   ├── direction_analyzer.py
│   ├── ews_scorer.py
│   ├── patient_monitor.py
│   └── report_generator.py
├── notebooks/       Demonstration and analysis notebooks (01-04)
├── tests/           Test suite
├── data/outputs/    Figures and result CSVs from this stage of the work
├── conftest.py
└── main.py
```

---

## 02_reproducibility_study

*EWS as an Integrated System: Classifier-Specific Calibration of Representation-Space Geometry for NORMAL Stratification in Retinal OCT.*

`01_ews_framework` raised a question it did not test: does the Layer A/B geometry, once computed for one trained classifier, hold for another? This study answers that directly. It trains four independent architectures (ResNet-50, SE-ResNet50, CBAM-ResNet50, ViT-B/16), evaluates an independently pretrained foundation model (RETFound), and tests whether the resulting NORMAL-atypicality geometry — both magnitude (Mahalanobis distance) and direction (disease-direction cosine similarity) — is reproducible across them, at three prespecified network depths.

**Central finding:** representation-space geometry is not a universal, classifier-independent measurement. It is reproducible within a shared architecture lineage at an early-to-mid network depth, and not reproducible across structurally distinct architecture families. EWS geometry must be developed and calibrated jointly with the specific classifier that generates it.

```
02_reproducibility_study/
├── notebooks/
│   └── Master_Evaluation_All_Models_PERSISTENT_ERROR_ANALYSIS.ipynb
│       False-NORMAL audit, single- and multi-layer Mahalanobis and
│       cosine geometry extraction, RETFound integration and comparison.
└── results/
    ├── classification_audit/
    │   ├── geometry/
    │   │   Per-layer Mahalanobis distance and disease-direction cosine
    │   │   results, pairwise reproducibility metrics (Spearman rank
    │   │   correlation, top-decile Jaccard overlap).
    │   ├── persistent_error_analysis/
    │   │   Cross-architecture error overlap, embedding extraction
    │   │   manifest, contact sheets of persistently misclassified images.
    │   ├── sample_level_predictions_*.csv
    │   ├── confusion_matrix_counts_all_runs.csv
    │   └── (per-run classification audit exports)
    └── visualizations/
        Per-run confusion matrices and ROC curves.
```

The written manuscript for this study is not included in this repository at this time.

---

## Trained model checkpoints

Trained checkpoints for all runs (both pieces of work) total approximately 250GB and are not hosted in this repository or on a public server. They are available from the author on reasonable request. The transfer method will be arranged individually depending on what is practical for the requester — for example, a peer with access to an institutional file-transfer service may prefer to use that, rather than relying on a method arranged solely by the author.

## Requirements

- Python 3.10
- PyTorch, torchvision
- timm (required for RETFound, `02_reproducibility_study` only)
- scikit-learn
- numpy, pandas
- matplotlib

## Dataset

Experiments in both `01_ews_framework` and `02_reproducibility_study` use the Kermany OCT2017 dataset. Due to licensing restrictions, raw data is not included. See the associated written work for details on the patient-stratified, leakage-corrected test set used.

## Status

`01_ews_framework`'s thresholds are empirical and not clinically validated. `02_reproducibility_study` is a cross-sectional computational study; it does not establish that representation-space stratification improves patient outcomes, and does not recommend a change to clinical recall-interval practice. Both are preliminary research outputs, not for clinical decision-making.

## Licence

MIT Licence. See LICENSE for details.

## Author

Ajantha Wirasinghe
MSc Computer Science with AI, Keele University
github.com/Ajantha-Wira
