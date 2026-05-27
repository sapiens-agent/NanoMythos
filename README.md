# Nanowhale T=1 Experiment Code, Wiki Results, and Report Materials

This curated package is intended for technical review, result presentation, and external communication around the Nanowhale T=1 recurrent-core experiment. It includes core code, experiment configurations, result tables, figures, Wiki experiment materials, and the historical report chain. The goal is to allow readers to understand from the root directory: what experiment was conducted, where the evidence is located, how to reproduce it, and which reports can be used for further explanation.

## One-Sentence Summary

In the FineWeb-Edu 10K cross-seed experiment, Nanowhale T=1 achieved lower held-out PPL than the baseline across 3/3 random seeds. In the Wiki 50-character short-context experiment, a more noticeable PPL improvement was also observed.

## Recommended Reading Order

1. `00_README.md` / `README.md`: Overview.
2. `01_quickstart/EXPERIMENT_INDEX.md`: Experiment map.
3. `06_reports/main/external_communication_brief_zh.md`: Formal external communication brief.
4. `06_reports/wiki/WIKI_REPORTS_INDEX.md`: Entry point for Wiki experiments and reports.
5. `06_reports/all_reports/`: Complete historical report chain.
6. `02_code/scripts/wiki/`: Wiki-related run scripts.

## Core Result Table

| Experiment | Baseline PPL | T=1 PPL | Conclusion |
|---|---:|---:|---|
| FineWeb-Edu 10K seed 2025 @ 5K | 156.7 | **147.7** | T=1 is lower |
| FineWeb-Edu 10K seed 2027 @ 5K | 187.7 | **151.4** | T=1 is lower |
| FineWeb-Edu 10K seed 2048 @ 5K | 154.2 | **145.7** | T=1 is lower |
| Wiki 50 characters | 2398 | **1985** | Approx. 17% improvement |

## Directory Structure

```text
00_README.md                         # First-read guide; same content as README.md
README.md                            # Project overview
01_quickstart/                       # Dependencies, experiment index, reproduction steps
02_code/                             # Core model, training entry point, experiment scripts; includes scripts/wiki
03_configs/                          # Configurations for baseline, T=1, T=2, T=4, etc.
