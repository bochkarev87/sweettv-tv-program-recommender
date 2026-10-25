# SWEET.TV TV program recommender (LLM4Rec midterm)

Top-5 TV program recommendation for weeks 31-42 of 2020 (Kaggle: `sweettv-tv-program-recommender`).

**Final submission:** `submission_v2_airprior.csv` (hybrid: decayed own-history + PureSVD + airing prior) -
Kaggle private 0.30404 / public 0.33326. `submission.csv` (= `submission_v1_hybrid.csv`) is the first version
(private 0.25025 / public 0.28799). Report: `report/bochkarev_ilya_midterm_report.pdf`.

## Pipeline (run in this order, Python 3.14, pandas 3.0, numpy 2.5, scipy 1.18, scikit-learn 1.9, kagglehub)
1. `download.py` - download competition data with kagglehub; set the data folder `D` in `common.py` to the printed path.
2. `02_build_interactions.py` - join logs with the schedule (channel + time overlap >= 60 s, `tv_show_id=0` dropped) -> `data/*.pkl`.
3. `03_experiments.py` - validation (train < 2020-06-01, validate 2020-06-01..07-26): popularity, own history, CBF, ALS, SVD, hybrid -> `results_validation.csv`.
4. `05_cbf_improve_hybrid.py`, `07_airing_prior.py` - CBF ablation, hybrid re-tuning, airing prior (8- and 12-week horizon; env `SWEET_CUTOFF=2020-05-04 SWEET_TAG=_h12w`).
5. `04_final_submission.py` (v1) and `04_final_submission.py air` (v2) - retrain on all logs, predict weeks 31-42.
6. Checks: `08_cbf_sanity.py`, `09_unit_checks.py`, `10_match_one_session.py`; case study: `06_case_study.py`; report: `report/build_report.py`.

`*_output.txt` and `results_*.csv` are the saved outputs quoted in the report. Raw data and `data/` are not in the repository.
