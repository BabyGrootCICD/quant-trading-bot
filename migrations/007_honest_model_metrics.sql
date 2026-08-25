-- 007: record whether a model is actually any good.
--
-- `skill_metrics()` computes up_rate, majority_baseline, balanced_accuracy and
-- edge_over_baseline every run, and `log_model_metrics()` wrote none of them.
-- Two consequences:
--
--   * `check_auto_upgrade()` gates on `edge_over_baseline`, which was never in
--     the table, so the promotion ladder (logistic -> neural -> xgboost) could
--     never fire. It has been dead code since it was written; the active model
--     only stays xgboost because the gate re-reads whatever was last logged.
--   * There was no stored record of skill. `accuracy` 0.5301 is meaningless
--     without the majority baseline sitting next to it -- TRX once scored 0.65
--     on a 0.64 baseline, which reads as strong and is noise.
--
-- Also adds `roc_auc`, which is invariant under any monotone transform and so
-- measures discrimination *before* calibration can flatten or inflate it. That
-- is the right thing to gate promotion on: it cannot be gamed by a calibration
-- artifact the way accuracy and the old in-sample-flavoured sharpe were.

ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS up_rate DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS majority_baseline DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS balanced_accuracy DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS edge_over_baseline DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS roc_auc DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS symbol VARCHAR(20);
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS calibration_error DOUBLE PRECISION;
ALTER TABLE model_metrics ADD COLUMN IF NOT EXISTS oos_rows INT;

CREATE INDEX IF NOT EXISTS idx_model_metrics_evaluated
    ON model_metrics(evaluated_at DESC);
