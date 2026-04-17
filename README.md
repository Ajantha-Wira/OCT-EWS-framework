# OCT Early Warning System (OCT-EWS)

An embedding-based post-classification safety layer for identifying 
unsafe normal predictions in automated OCT screening.

## Overview

This repository contains the full implementation of the OCT Early 
Warning System (EWS) framework introduced in:

> Wirasinghe, A. (2026) *An Embedding-Based Post-Classification Safety 
> Layer for Identifying Unsafe Normal Predictions in OCT Screening*. 
> [Under review]

The EWS operates as a post-classification module alongside a trained 
deep learning classifier. It identifies structurally atypical normal 
predictions by analysing representation-space geometry rather than 
relying on output probabilities alone.

## Framework

The system consists of three sequential layers:

- **Layer A** — Mahalanobis distance-based atypicality detection
- **Layer B** — Disease-direction geometry analysis
- **Layer C** — Clinical prioritisation and workflow decisions

## Repository Structure

```
OCT_EWS/
├── config/          # Phase configuration and defaults
├── src/             # Core modules
│   ├── data_manager.py
│   ├── reference_builder.py
│   ├── anomaly_scorer.py
│   ├── direction_analyzer.py
│   ├── ews_scorer.py
│   ├── patient_monitor.py
│   └── report_generator.py
├── notebooks/       # Demonstration and analysis notebooks
├── tests/           # Test suite
└── main.py
```
## Requirements

- Python 3.10
- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib

## Dataset

Experiments use the Kermany OCT2017 dataset. Due to licensing 
restrictions, raw data is not included. See the paper for details 
on the leakage-corrected preprocessing procedure.

## Phase Design

- **Phase 1** — Empirical thresholds (current implementation)
- **Phase 2** — Clinically validated thresholds (future work)

> PRELIMINARY RESEARCH OUTPUT. Thresholds are not clinically 
> validated. Not for clinical decision making.

## Citation

If you use this code in your research, please cite:

Wirasinghe, A. (2026) An embedding-based post-classification safety layer for detecting unsafe normal predictions in automated OCT screening.
[Under review]

## Licence

MIT Licence. See LICENSE for details.

## Author

Ajantha Wirasinghe  
MSc Computer Science with AI, Keele University  
github.com/Ajantha-Wira
