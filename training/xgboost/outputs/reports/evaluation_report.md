# XGBoost Entity Matching - Evaluation Report

**Experiment:** xgboost_baseline
**Timestamp:** 2026-09-27T15:07:58.738195
**Config:** {
  "data_root": "D:\\Amazon_challenge_ML\\Amazon-ML-dataset",
  "candidates_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\blocking\\candidate_pairs.tsv",
  "ground_truth_path": "D:\\Amazon_challenge_ML\\Amazon-ML-dataset\\student_resource\\dataset\\train\\train_ground_truth.tsv",
  "train_s1_path": "D:\\Amazon_challenge_ML\\Amazon-ML-dataset\\student_resource\\dataset\\train\\train_source1.tsv",
  "train_s2_path": "D:\\Amazon_challenge_ML\\Amazon-ML-dataset\\student_resource\\dataset\\train\\train_source2.tsv",
  "train_s3_path": "D:\\Amazon_challenge_ML\\Amazon-ML-dataset\\student_resource\\dataset\\train\\train_source3.tsv",
  "output_dir": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs",
  "model_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\models\\xgboost_model.json",
  "model_pkl_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\models\\xgboost_model.pkl",
  "val_predictions_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\predictions\\validation_predictions.csv",
  "metrics_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\metrics\\metrics.json",
  "threshold_results_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\metrics\\threshold_results.csv",
  "best_threshold_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\metrics\\best_threshold.json",
  "eval_report_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\reports\\evaluation_report.md",
  "error_analysis_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\reports\\error_analysis.csv",
  "feature_importance_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\feature_importance\\feature_importance.csv",
  "experiment_config_path": "D:\\Amazon_challenge_ML\\amazon-ml-hackathon-26\\training\\xgboost\\outputs\\metrics\\experiment_config.json",
  "val_split_ratio": 0.15,
  "val_split_seed": 20260926,
  "stratified_by_country": true,
  "use_name_exact": true,
  "use_name_jaro": true,
  "use_name_jarowinkler": true,
  "use_name_levenshtein": true,
  "use_name_token_jaccard": true,
  "use_name_token_dice": true,
  "use_name_token_set_ratio": true,
  "use_name_token_sort_ratio": true,
  "use_name_char_ngram_jaccard": true,
  "use_name_tfidf_cosine": false,
  "use_addr_exact": true,
  "use_addr_token_jaccard": true,
  "use_addr_token_dice": true,
  "use_addr_char_jaccard": true,
  "use_addr_levenshtein": true,
  "use_addr_tfidf_cosine": false,
  "use_addr_city_match": true,
  "use_addr_postal_match": true,
  "use_addr_housenumber_match": true,
  "use_phone_exact": true,
  "use_phone_normalized": true,
  "use_phone_suffix": true,
  "use_phone_missing": true,
  "use_country_match": false,
  "use_cross_script": true,
  "use_transliteration_sim": true,
  "use_blocker_rank": true,
  "use_blocker_score": true,
  "use_blocker_passes": true,
  "use_missing_indicators": true,
  "use_group_rank": true,
  "use_group_margin_to_2nd": true,
  "use_group_score_ratio_to_max": true,
  "use_group_size": true,
  "use_is_mutual_best": true,
  "use_n_competitors": true,
  "use_name_high_addr_low": true,
  "use_name_low_addr_high": true,
  "use_sim_disagreement": true,
  "encode_missing_as_nan": true,
  "xgb_params": {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 20260926,
    "n_jobs": -1,
    "tree_method": "hist",
    "enable_categorical": false,
    "verbosity": 1
  },
  "num_boost_round": 500,
  "early_stopping_rounds": 50,
  "use_scale_pos_weight": true,
  "threshold_grid": [
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9
  ],
  "compute_macro_f05": true,
  "compute_pr_auc": true,
  "compute_top_k": true,
  "top_k_values": [
    1,
    3,
    5,
    10
  ],
  "seed": 20260926,
  "experiment_name": "xgboost_baseline",
  "experiment_timestamp": "2026-09-27T15:04:56.244737"
}

## Dataset Summary

- Training pairs: 254,997
- Validation pairs: 44,999
- Training entities: 1,700
- Validation entities: 300
- Features: 54
- Countries: 1

## Model Performance (Best Threshold)

**Best Threshold:** 0.9000

- **Macro F0.5:** 0.964323
- Macro Precision: 0.975135
- Macro Recall: 0.953135
- Macro F1: 0.964009
- Global Precision: 0.970641
- Global Recall: 0.953671
- Global F1: 0.962081
- PR-AUC: 0.989524
- ROC-AUC: 0.999081

## Confusion Matrix

- TP: 1,091
- FP: 33
- FN: 53
- TN: 43,822

## Per-Entity Analysis

- Singletons: 0 (F0.5: 0.000000)
- Non-singletons: 300 (F0.5: 0.964323)
- Empty predictions: 1
- Multi-predictions: 280

## Top-K Accuracy

- Top-1: 1.000000
- Top-3: 1.000000
- Top-5: 1.000000
- Top-10: 1.000000

## Ranking Metrics

- MRR: 1.000000
- NDCG@5: 0.990818
- NDCG@10: 0.994691

## Threshold Sweep Results

|   threshold |   macro_f05 |   macro_precision |   macro_recall |   macro_f1 |   global_precision |   global_recall |   global_f1 |   pr_auc |   n_predicted |   n_singletons |
|------------:|------------:|------------------:|---------------:|-----------:|-------------------:|----------------:|------------:|---------:|--------------:|---------------:|
|        0.1  |    0.868865 |          0.853172 |       0.984833 |   0.914287 |           0.807445 |        0.986014 |    0.887839 | 0.989524 |          1397 |              0 |
|        0.15 |    0.896951 |          0.885576 |       0.984167 |   0.932272 |           0.846732 |        0.98514  |    0.910707 | 0.989524 |          1331 |              0 |
|        0.2  |    0.907752 |          0.897807 |       0.983333 |   0.938626 |           0.868157 |        0.984266 |    0.922573 | 0.989524 |          1297 |              0 |
|        0.25 |    0.922177 |          0.914043 |       0.983333 |   0.947423 |           0.892942 |        0.984266 |    0.936383 | 0.989524 |          1261 |              0 |
|        0.3  |    0.926683 |          0.919812 |       0.981167 |   0.949499 |           0.900561 |        0.981643 |    0.939356 | 0.989524 |          1247 |              0 |
|        0.35 |    0.930636 |          0.924536 |       0.9805   |   0.951696 |           0.909238 |        0.980769 |    0.94365  | 0.989524 |          1234 |              0 |
|        0.4  |    0.934488 |          0.929846 |       0.978278 |   0.953447 |           0.914286 |        0.979021 |    0.945547 | 0.989524 |          1225 |              0 |
|        0.45 |    0.939142 |          0.939229 |       0.974111 |   0.956352 |           0.923967 |        0.977273 |    0.949873 | 0.989524 |          1210 |              0 |
|        0.5  |    0.942071 |          0.943272 |       0.971778 |   0.957313 |           0.929942 |        0.97465  |    0.951771 | 0.989524 |          1199 |              0 |
|        0.55 |    0.945393 |          0.947811 |       0.968968 |   0.958273 |           0.937605 |        0.972028 |    0.954506 | 0.989524 |          1186 |              0 |
|        0.6  |    0.947514 |          0.950904 |       0.966746 |   0.958759 |           0.942226 |        0.969406 |    0.955623 | 0.989524 |          1177 |              0 |
|        0.65 |    0.951743 |          0.955792 |       0.966746 |   0.961238 |           0.948674 |        0.969406 |    0.958928 | 0.989524 |          1169 |              0 |
|        0.7  |    0.953769 |          0.958865 |       0.963746 |   0.961299 |           0.952586 |        0.965909 |    0.959201 | 0.989524 |          1160 |              0 |
|        0.75 |    0.956097 |          0.962151 |       0.962357 |   0.962254 |           0.955806 |        0.964161 |    0.959965 | 0.989524 |          1154 |              0 |
|        0.8  |    0.958898 |          0.965762 |       0.961246 |   0.963499 |           0.959059 |        0.962413 |    0.960733 | 0.989524 |          1148 |              0 |
|        0.85 |    0.959665 |          0.967992 |       0.95669  |   0.962308 |           0.963061 |        0.957168 |    0.960105 | 0.989524 |          1137 |              0 |
|        0.9  |    0.964323 |          0.975135 |       0.953135 |   0.964009 |           0.970641 |        0.953671 |    0.962081 | 0.989524 |          1124 |              0 |

## Feature Importance (Top 20 by Gain)

| feature                     |   importance |   rank |   importance_pct |
|:----------------------------|-------------:|-------:|-----------------:|
| addr_containment_cand_in_s1 |    4300.68   |      1 |        36.541    |
| addr_token_dice_nostop      |    1595.5    |      2 |        13.5563   |
| name_token_jaccard_nosuffix |    1114.3    |      3 |         9.46774  |
| addr_token_jaccard_nostop   |    1048.91   |      4 |         8.91212  |
| name_token_dice_nosuffix    |     455.461  |      5 |         3.86986  |
| name_jarowinkler            |     377.627  |      6 |         3.20853  |
| addr_missing_either         |     345.972  |      7 |         2.93957  |
| addr_token_jaccard          |     223.063  |      8 |         1.89527  |
| addr_missing_cand           |     216.857  |      9 |         1.84254  |
| addr_len_cand               |     188.731  |     10 |         1.60356  |
| transliteration_sim         |     181.717  |     11 |         1.54397  |
| addr_city_match             |     159.8    |     12 |         1.35775  |
| addr_token_dice             |     124.826  |     13 |         1.06059  |
| name_char3gram_jaccard      |     113.799  |     14 |         0.966898 |
| name_containment_s1_in_cand |     101.266  |     15 |         0.860415 |
| name_token_dice             |      99.1024 |     16 |         0.842031 |
| addr_token_count_cand       |      94.6783 |     17 |         0.804442 |
| name_token_jaccard          |      81.9425 |     18 |         0.69623  |
| addr_postal_match           |      77.6177 |     19 |         0.659485 |
| addr_housenumber_match      |      74.7011 |     20 |         0.634704 |

## Timing

- Training time: 5.61 seconds
- Prediction time: 0.08 seconds

## XGBoost Parameters

```json
{
  "objective": "binary:logistic",
  "eval_metric": "logloss",
  "learning_rate": 0.03,
  "max_depth": 6,
  "min_child_weight": 1,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_alpha": 0.0,
  "reg_lambda": 1.0,
  "random_state": 20260926,
  "n_jobs": -1,
  "tree_method": "hist",
  "enable_categorical": false,
  "verbosity": 1
}
```
