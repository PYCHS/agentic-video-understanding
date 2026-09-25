# Authoritative proposal and attribution

Specification: **Watch Wisely: Budget-Aware Adaptive Frame Selection for Video Question
Answering**, NCKU Group 128, supplied as `HW1_128_CVPR2026.pdf` (two pages).
Proposal authors: 劉邦佑、蔡源慶、彭以呈、部政佑.
Repository maintainer: Peng Yi Cheng (彭以呈), GitHub PYCHS.

Source SHA-256: `51655df546f391b98ce87d1eeabb1282724a826a312f1d9ca91de6ed959b50b7`.
The PDF was read directly for this foundation. It is not redistributed because its front
page includes student identifiers and contact information. The filename is not evidence of
conference submission or acceptance. Do not imply sole authorship of team work.

| Proposal section | Repository mapping |
|---|---|
| 1: matched fixed-budget comparison | README; configs/main.json; evaluation/README.md |
| 2.1: 448px, survey, tools | sampling.py; controller.py; docs/architecture.md |
| 2.2: four rounds, evidence/confidence stop | controller.py; tests/test_foundation.py |
| 2.3: screening, baselines, costs, statistics | evaluation/README.md; docs/data_screening.md |
| 3: gain/comparability criteria | evaluation/README.md; pending README table |

Mappings distinguish implemented controller behavior from planned inference/evaluation.
Midpoints, tie-breaking, bootstrap replicate count and always-24 feasibility are explicit
engineering choices to freeze before evaluation. Expected outcomes are not measured results.
