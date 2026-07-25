#!/bin/bash
set -euo pipefail

python3 -c "import socket, time
targets=[('kafka', 29092), ('cassandra', 9042), ('redis', 6379), ('spark-master', 7077)]
for host, port in targets:
    deadline = time.time() + 180
    while True:
        try:
            with socket.create_connection((host, port), timeout=3):
                break
        except OSError:
            if time.time() > deadline:
                raise SystemExit(f'Timeout waiting for {host}:{port}')
            time.sleep(2)
"

mkdir -p /tmp/spark-events /tmp/spark_checkpoints

/opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --deploy-mode client \
  --driver-memory "${SPARK_DRIVER_MEMORY:-1g}" \
  --executor-memory "${SPARK_EXECUTOR_MEMORY:-1g}" \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  --conf spark.sql.streaming.forceDeleteTempCheckpointLocation=true \
  --conf spark.eventLog.enabled=true \
  --conf spark.eventLog.dir=file:///tmp/spark-events \
  --conf spark.sql.shuffle.partitions=4 \
  --conf spark.sql.constraintPropagation.enabled=false \
  /opt/spark/app/stream_job.py
