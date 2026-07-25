# Kịch Bản Demo Pipeline Phát Hiện Gian Lận Real-time

Tài liệu này là **kịch bản trình diễn** (không phải runbook vận hành đầy đủ — xem [FULL_PIPELINE_RUN_GUIDE.md](FULL_PIPELINE_RUN_GUIDE.md) và [LOCAL_STEP_BY_STEP_RUNBOOK.md](LOCAL_STEP_BY_STEP_RUNBOOK.md) cho việc đó). Mục tiêu: trình diễn được toàn bộ vòng đời một giao dịch — từ lúc phát sinh tới lúc bị chấm điểm, cảnh báo, con người duyệt lại, và hệ thống tự giám sát chất lượng model — trong khoảng **15–20 phút**.

## 0. Đối tượng & phạm vi

- Đối tượng: giảng viên / stakeholder không cần biết sâu Spark/Kafka.
- Không demo phần nạp 500MB dữ liệu gốc (quá lâu) — dùng bộ dữ liệu mẫu nhỏ đã tách sẵn tại `Data/logical_sources/` (100 dòng) và traffic tổng hợp qua API.
- Toàn bộ lệnh dưới đây đã được chạy thử thực tế trên máy này trong lúc soạn tài liệu.

## 1. Kiến trúc tóm tắt (nói trong 1 phút, chiếu README)

```
3 CSV nguồn (transaction / sender_state / receiver_state)
    -> Kafka (3 topic độc lập)
    -> Spark Structured Streaming (3-way interval join + Rule Engine + ML hybrid scoring)
    -> Cassandra (lưu vĩnh viễn) + Redis (cache nóng cho dashboard)
    -> API FastAPI (scoring đồng bộ, endpoint /score)
    -> Streamlit dashboard (Live Alerts / Review Queue / Case Details / Monitoring)
    -> Model Monitoring batch (drift, performance, retraining trigger)
```

Điểm nhấn cần nói: hệ thống có **2 đường vào** cùng chấm điểm bằng chung 1 RuleEngine/model — đường streaming thật (Kafka→Spark, mô phỏng luồng giao dịch ngân hàng) và đường đồng bộ (API, dùng cho tích hợp trực tiếp) — cả hai đều ghi vào cùng Cassandra để dashboard và monitoring dùng chung.

## 2. Chuẩn bị trước buổi demo (làm trước 10–15 phút)

```powershell
docker-compose up -d
docker-compose ps
```

Đợi tới khi các service chính báo `healthy`: `kafka`, `cassandra`, `redis`, `spark-master`, `spark-worker`, `streamlit`, `fraud-api`.

**Cảnh báo quan trọng:** TUYỆT ĐỐI không chạy `docker-compose down -v` giữa hoặc ngay trước buổi demo — cờ `-v` xóa sạch volume Cassandra/Kafka, mất hết dữ liệu đã tích lũy (đã gặp đúng lỗi này trong lúc test tài liệu này: 581 dòng prediction bị xóa sạch về 0 chỉ vì `down -v`). Chỉ dùng `docker-compose down` (không `-v`) nếu cần tắt.

Kiểm tra nhanh API và dữ liệu tham chiếu monitoring đã có chưa:

```powershell
curl http://localhost:8000/health
python monitoring/model/reference_builder.py --max-rows 5000   # chạy 1 lần, bỏ qua nếu file đã tồn tại
```

Danh sách UI sẽ dùng trong demo:

| UI | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| Kafka UI | http://localhost:8085 |
| Spark Master UI | http://localhost:8080 |
| Spark App UI (streaming query) | http://localhost:4040 |
| Grafana (hạ tầng) | http://localhost:3001 |
| API docs (Swagger) | http://localhost:8000/docs |

## 3. Act 1 — Chấm điểm tức thời qua API (2 phút)

**Mục tiêu:** cho thấy model phản hồi ngay lập tức cho 1 giao dịch đơn lẻ, giống một ngân hàng gọi API tích hợp trực tiếp.

Mở http://localhost:8000/docs song song, hoặc chạy:

```powershell
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d "{\"step\":1,\"type\":\"TRANSFER\",\"amount\":260000,\"nameOrig\":\"C1\",\"oldbalanceOrg\":300000,\"newbalanceOrig\":100,\"nameDest\":\"C2\",\"oldbalanceDest\":1000,\"newbalanceDest\":261000,\"isFraud\":1}"
```

**Chỉ ra trên kết quả JSON:** `risk_score`, `hybrid_score` (kết hợp rule + ML), `decision`, `triggered_rules` — giải thích hybrid_score = rule_weight × rule_score + ml_weight × ml_score.

## 4. Act 2 — Luồng streaming thật: Kafka → Spark → Cassandra (5 phút)

**Mục tiêu:** chứng minh pipeline streaming thật đang chạy, không phải giả lập.

Mở sẵn 2 tab trình duyệt: Kafka UI (8085) và Spark App UI (4040, tab "Structured Streaming").

Bơm 100 giao dịch mẫu (đã tách sẵn 3 nguồn) qua Kafka:

```powershell
python scripts/publish_logical_sources_parallel.py --max-events 100 --rate 20
```

**Chỉ ra trên UI trong lúc lệnh chạy (~5-10 giây):**
- Kafka UI: 3 topic `transaction_topic`, `sender_state_topic`, `receiver_state_topic` có message mới.
- Spark App UI (4040): query streaming đang active, batch mới được xử lý.
- Streamlit tab **Live Alerts**: danh sách cảnh báo mới xuất hiện gần thời gian thực (Redis cache nóng).

Nếu muốn kiểm tra dữ liệu đã tới Cassandra:

```powershell
docker exec cassandra cqlsh -e "SELECT COUNT(*) FROM fraud_detection.model_predictions_by_day;"
```

## 5. Act 3 — Con người trong vòng lặp: Review Queue (3 phút)

**Mục tiêu:** cho thấy alert không tự động kết luận — có analyst duyệt lại, và việc duyệt này chính là nguồn nhãn (label) nuôi lại cho Model Monitoring ở Act 5.

Trong Streamlit, chuyển tab **Review Queue**:
1. Lọc theo Severity = `high`, chọn 1 case trong bảng.
2. Bấm **Mark Fraud** hoặc **Mark Legit** (hoặc điền form Reviewer/Notes rồi **Save Review**).
3. Nói rõ: hành động này ghi trực tiếp vào bảng Cassandra `alert_reviews` — đây chính là nhãn thật mà `performance_report.py` ở Act 5 sẽ đọc để tính precision/recall.

Có thể mở nhanh tab **Case Details** để xem chi tiết 1 case (rule đã trigger, số dư trước/sau).

## 6. Act 4 — Đổ traffic để tạo drift thật (3 phút)

**Mục tiêu:** mô phỏng hành vi giao dịch thay đổi đột ngột (ví dụ: đợt tấn công mới, hoặc thay đổi hành vi khách hàng) để phần Monitoring ở Act 5 có tín hiệu drift rõ ràng, thay vì demo trên dữ liệu "sạch" nhàm chán.

```powershell
python scripts/demo_api_drift_traffic.py --drift-count 200
```

Script này gửi 200 giao dịch tổng hợp qua **API thật** (`/score/batch`), amount bị đẩy cao gấp 5 lần và tỷ trọng `TRANSFER` tăng vọt so với baseline — mỗi giao dịch đi qua đúng luồng scoring thật và được ghi vào Cassandra `model_predictions_by_day`, giống hệt traffic production thật, chỉ khác là dữ liệu tổng hợp.

## 7. Act 5 — Model Monitoring: drift, performance, quyết định retrain (5 phút)

**Mục tiêu:** đây là phần đặc trưng nhất — cho thấy hệ thống tự phát hiện khi model "lệch" so với baseline và tự đề xuất retrain.

```powershell
python monitoring/model/drift_report.py --cassandra-host localhost --cassandra-port 9042 --cassandra-keyspace fraud_detection --day-bucket <YYYY-MM-DD-hom-nay>
python monitoring/model/performance_report.py --cassandra-host localhost --cassandra-port 9042 --cassandra-keyspace fraud_detection --day-bucket <YYYY-MM-DD-hom-nay>
python monitoring/model/check_retraining_trigger.py
```

> Luôn truyền `--day-bucket` đúng ngày hôm nay. Nếu bỏ qua, script sẽ dò ngược tối đa 400 ngày để tìm dữ liệu (`discover_prediction_days`), có thể mất hơn 1-2 phút — dễ khiến người xem tưởng bị treo.
>
> **Quan trọng:** `day_bucket` được API/Spark ghi theo **giờ UTC của container** (docker-compose không set timezone VN), không phải giờ máy tính cá nhân. Việt Nam UTC+7 nên có thể lệch 1 ngày — ví dụ 01:00 sáng giờ VN ngày 25 vẫn là 18:00 UTC ngày 24. Kiểm tra bucket thật trước khi truyền: `docker exec cassandra cqlsh -e "SELECT day_bucket, COUNT(*) FROM fraud_detection.model_predictions_by_day GROUP BY day_bucket;"` rồi dùng đúng giá trị đó cho `--day-bucket`.

**Reload trình duyệt** (F5) tab **Monitoring** trong Streamlit — đây là bước hay quên nhất, Streamlit không tự poll file, phải F5 hoặc bấm menu ⋮ → Rerun.

**Chỉ ra trên dashboard:**
- 4 ô metric trên cùng: **Drifted Features**, **Label Coverage**, **7D Precision**, **Retrain Required**.
- Biểu đồ **Drift Summary**: feature nào vượt ngưỡng KS-statistic (numeric) hoặc total variation distance (categorical).
- Khối **Retraining Decision**: liệt kê chính xác lý do (`reasons`) vì sao hệ thống đề xuất retrain — đây là điểm khác biệt so với chỉ hiển thị số liệu suông.
- **Rolling Performance**: precision/recall/f1 theo cửa sổ 1d/7d/30d — nối lại với việc duyệt case ở Act 3 (label càng nhiều, số liệu càng đáng tin — nhắc lại `label_coverage` thấp vì demo mới duyệt 1-2 case).

### Phương án dự phòng nếu không có traffic Cassandra kịp lúc

Nếu vì lý do gì đó Cassandra không có dữ liệu (mạng lỗi, do quên chạy Act 4), dùng đường offline không phụ thuộc hạ tầng:

```powershell
python scripts/make_drifted_serving_csv.py
python monitoring/model/drift_report.py --reference-csv monitoring/reference/reference_dataset.csv --serving-csv monitoring/reference/serving_drifted.csv
python monitoring/model/check_retraining_trigger.py
```

Cho kết quả tương đương (drift rõ ràng ở `amount`/`txn_type`) mà không cần API/Cassandra sống — dùng làm phương án B nếu demo trực tiếp gặp sự cố hạ tầng.

## 8. Act 6 (tuỳ chọn) — Hạ tầng tự giám sát: Grafana (2 phút)

Mở http://localhost:3001, chỉ ra panel Throughput (EPS), JVM Heap Memory, Spark Executor Metrics — nhấn mạnh đây là tầng giám sát **hạ tầng** (Spark/JVM), khác với tầng giám sát **chất lượng model** vừa demo ở Act 5.

## 9. Xử lý sự cố nhanh trong lúc demo

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| Dashboard Monitoring không đổi số liệu | Streamlit không tự rerun khi file thay đổi | F5 trang, hoặc menu ⋮ → Rerun |
| `drift_report.py` báo "No serving rows available" | Cassandra rỗng (có thể do lỡ `down -v`) | Chạy lại Act 4 (`demo_api_drift_traffic.py`) để bơm lại dữ liệu |
| Lệnh monitoring chạy rất lâu (>1 phút) không thấy gì | Thiếu `--day-bucket`, script dò ngược 400 ngày | Luôn truyền `--day-bucket <hôm-nay>` |
| `drift_report.py` báo "No serving rows available" dù Act 4 đã chạy | `day_bucket` lưu theo giờ UTC container, lệch với giờ VN (UTC+7) | Kiểm tra bucket thật bằng cqlsh (xem Act 5) trước khi truyền `--day-bucket` |
| `performance_report.py` báo lỗi `KeyError: label_available` | Bug cũ khi Cassandra 0 dòng — đã vá trong `performance_report.py` | Đảm bảo đang chạy bản code hiện tại của repo (đã fix) |
| Muốn dữ liệu drift ổn định, lặp lại y hệt mỗi lần tổng duyệt trước | `make_drifted_serving_csv.py` mặc định `--seed 42` | Đổi `--seed` nếu muốn kết quả khác mỗi lần |

## 10. Checklist tổng kết cuối demo

- [ ] Act 1: API trả JSON tức thời cho 1 giao dịch
- [ ] Act 2: Kafka UI + Spark UI + Streamlit Live Alerts đều cập nhật sau khi bơm 100 giao dịch
- [ ] Act 3: 1 case được duyệt (Mark Fraud/Legit), thấy `alert_reviews` được ghi
- [ ] Act 4: 200 giao dịch drift đã gửi qua API thật
- [ ] Act 5: dashboard Monitoring hiển thị drift + retraining decision đúng dữ liệu vừa tạo
- [ ] (tuỳ chọn) Act 6: Grafana cho thấy tầng giám sát hạ tầng riêng biệt

## 11. Tài liệu liên quan

- [FULL_PIPELINE_RUN_GUIDE.md](FULL_PIPELINE_RUN_GUIDE.md) — vận hành full 500MB dữ liệu thật (không dùng cho demo nhanh)
- [LOCAL_STEP_BY_STEP_RUNBOOK.md](LOCAL_STEP_BY_STEP_RUNBOOK.md) — runbook kỹ thuật đầy đủ, dùng khi debug sâu
- [MODEL_DEPLOYMENT.md](MODEL_DEPLOYMENT.md) — runbook riêng cho API + monitoring scripts
- `scripts/make_drifted_serving_csv.py`, `scripts/demo_api_drift_traffic.py` — 2 script hỗ trợ demo drift (offline CSV và live API)
