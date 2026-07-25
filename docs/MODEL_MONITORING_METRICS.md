# Model Monitoring Metrics

Tai lieu nay giai thich cac chi so ma phan model monitoring dang do trong repo.

## 1. Data Drift

File chinh: `monitoring/model/drift_report.py`

Muc tieu cua nhom metric nay la kiem tra xem du lieu production/serving co lech so voi baseline reference hay khong.

### Numeric features

Danh sach feature numeric dang duoc monitor:

- `amount`
- `risk_score`
- `ml_score`

Metric su dung: `ks_statistic` tu Kolmogorov-Smirnov test.

Y nghia:

- So sanh phan phoi cua tung feature giua reference data va serving data.
- `ks_statistic` cang lon thi hai phan phoi cang khac nhau.
- Code danh dau drift neu `ks_statistic >= 0.2`.

Bao cao numeric drift con luu them:

- `ks_pvalue`: p-value cua KS test.
- `reference_mean`: gia tri trung binh trong reference data.
- `serving_mean`: gia tri trung binh trong serving data.
- `mean_delta`: chenhlech trung binh, tinh bang `serving_mean - reference_mean`.

### Categorical features

Danh sach feature categorical dang duoc monitor:

- `txn_type`
- `severity`
- `is_alert`

Metric su dung: `total_variation_distance`.

Y nghia:

- So sanh ty le tung category giua reference va serving.
- Vi du: ty le `TRANSFER`, `CASH_OUT`, `HIGH`, `LOW`, `True`, `False` thay doi bao nhieu.
- Code danh dau drift neu `total_variation_distance >= 0.1`.

Bao cao categorical drift con luu `distribution`, gom ty le cua tung category trong reference va serving.

### Drift summary

Drift report tong hop cac truong:

- `reference_rows`: so dong baseline reference.
- `serving_rows`: so dong serving/production duoc kiem tra.
- `monitored_features`: danh sach feature duoc monitor.
- `drifted_feature_count`: so feature bi drift.
- `drifted_features`: ten cac feature bi drift.
- `features`: chi tiet metric cua tung feature.

## 2. Model Performance

File chinh: `monitoring/model/performance_report.py`

Performance duoc tinh tu prediction log ket hop voi label that hoac analyst review. Phan ghep label nam trong `monitoring/model/metrics_store.py`.

### Confusion matrix

He thong tinh cac thanh phan confusion matrix tren nhung row da co label:

- `tp`: model bao fraud va thuc te la fraud.
- `fp`: model bao fraud nhung thuc te legit.
- `tn`: model khong bao fraud va thuc te legit.
- `fn`: model khong bao fraud nhung thuc te fraud.

Trong code:

- `predicted_positive` duoc suy ra tu `is_alert`.
- `actual_positive` la label hieu luc bang `fraud`.
- `actual_negative` la label hieu luc bang `legit`.
- Label hieu luc uu tien `actual_label`, neu khong co thi dung `review_label`.

### Performance metrics

- `precision = tp / (tp + fp)`: trong cac alert model bao fraud, bao nhieu alert la dung. Precision thap nghia la nhieu false alarm.
- `recall = tp / (tp + fn)`: trong cac fraud that, model bat duoc bao nhieu. Recall thap nghia la bo sot fraud.
- `f1 = 2 * precision * recall / (precision + recall)`: diem can bang giua precision va recall.
- `false_positive_rate = fp / (fp + tn)`: ty le giao dich legit bi bao nham la fraud.

Neu khong du mau de tinh mau so, cac metric nhu `precision`, `recall`, `f1`, `false_positive_rate` co the la `null` trong JSON report.

### Label coverage

Performance report cung theo doi do phu label:

- `prediction_rows`: tong so prediction duoc dua vao bao cao.
- `review_rows`: so review tu analyst.
- `labeled_rows`: so prediction co label hieu luc.
- `unlabeled_rows`: so prediction chua co label.
- `label_coverage = labeled_rows / prediction_rows`: ty le prediction co label de danh gia.

`label_coverage` rat quan trong vi precision/recall chi dang tin khi co du label. Neu coverage thap, report se them warning.

### Rolling windows

Performance duoc tinh theo:

- `overall`: toan bo du lieu co label.
- `rolling_windows.1d`: cua so 1 ngay gan nhat.
- `rolling_windows.7d`: cua so 7 ngay gan nhat.
- `rolling_windows.30d`: cua so 30 ngay gan nhat.

Moi rolling window deu co cac chi so `labeled_rows`, `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, `f1`, `false_positive_rate`, `window_start`, `window_end`.

### Warnings

Report co the tao warning trong cac truong hop:

- Chua co labeled prediction, nen precision/recall chua phan anh chat luong production.
- Khong co labeled negative prediction, nen recall co the qua lac quan vi false negative chua duoc quan sat day du.

## 3. Retraining Trigger

File chinh: `monitoring/model/check_retraining_trigger.py`

Policy: `monitoring/model/retraining_policy.json`

Phan nay tong hop drift report va performance report de quyet dinh co can retrain model hay khong.

### Policy thresholds hien tai

- `amount_ks_stat_threshold = 0.2`: trigger neu drift cua `amount` qua cao.
- `feature_drift_ratio_threshold = 0.3`: trigger neu ty le feature bi drift dat toi 30% so feature dang monitor.
- `precision_7d_min = 0.8`: trigger neu precision 7 ngay thap hon 0.8.
- `recall_7d_min = 0.7`: trigger neu recall 7 ngay thap hon 0.7.
- `f1_7d_min = 0.75`: trigger neu F1 7 ngay thap hon 0.75.
- `label_coverage_min = 0.5`: trigger/canh bao neu label coverage duoi 50%.
- `alert_rate_change_threshold = 0.1`: trigger neu ty le alert thay doi tu 10 diem phan tram tro len so voi baseline.
- `minimum_labeled_rows_7d = 25`: can it nhat 25 labeled rows trong 7 ngay de performance trigger dang tin.
- `require_minimum_sample_for_trigger = true`: neu chua du sample thi danh gia performance trigger theo huong conservative.

### Retraining decision output

Ket qua retraining gom:

- `retrain_required`: `true` neu co it nhat mot ly do trigger.
- `reason_count`: so ly do trigger.
- `reasons`: danh sach ly do, gom `type`, `message`, `observed`, `threshold`.
- `warnings`: canh bao ve do tin cay cua du lieu monitoring.
- `policy_snapshot`: policy duoc dung tai thoi diem chay.
- `observations`: cac gia tri quan sat duoc dung de ra quyet dinh.

`observations` hien gom:

- `drifted_feature_count`
- `monitored_feature_count`
- `drift_ratio`
- `reference_alert_rate`
- `serving_alert_rate`
- `alert_rate_delta`
- `label_coverage`
- `rolling_7d_labeled_rows`
- `rolling_7d_precision`
- `rolling_7d_recall`
- `rolling_7d_f1`

## 4. Streaming Window Metrics

Ngoai model monitoring, pipeline streaming con co window metrics duoc ghi tu Spark job.

File lien quan:

- `spark-app/stream_job.py`
- `fraud_pipeline/windows.py`
- `fraud_pipeline/models.py`

Bang Cassandra lien quan: `metrics_by_window`.

Metric streaming gom:

- `event_count`: so giao dich trong cua so thoi gian.
- `fraud_count`: so giao dich bi danh dau fraud trong cua so.
- `total_amount`: tong gia tri giao dich trong cua so.
- `fraud_rate = fraud_count / event_count`: ty le fraud trong cua so.

Cac metric nay theo doi luong giao dich va ty le fraud theo thoi gian. Chung khac voi drift/performance metrics, vi khong danh gia truc tiep chat luong model ma phan anh trang thai stream giao dich.
