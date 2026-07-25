FROM apache/spark:3.5.1

USER root

# apache/spark:3.5.1 ships Python 3.8 (Ubuntu 20.04 "focal"), which cannot install
# numpy>=2.0 / scikit-learn>=1.8 (the versions the fraud model artifact was trained and
# pinned with). deadsnakes no longer publishes builds for focal, so apt can't provide
# python3.11 either. Use uv to fetch a standalone Python 3.11 build instead, put the ML
# deps in a venv on top of it, and point the PySpark driver at that interpreter. Executors
# keep using the stock Python 3.8 — this job has no Python UDFs, ML scoring runs inside
# foreachBatch on the driver only, so executors never need the ML libraries.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

RUN uv venv /opt/venv311 --python 3.11 \
    && uv pip install --python /opt/venv311/bin/python --no-cache \
       cassandra-driver redis kafka-python \
       numpy==2.4.4 joblib==1.5.3 scikit-learn==1.8.0 \
       pandas==3.0.3 xgboost==3.2.0 lightgbm==4.5.0

ENV PYSPARK_PYTHON=/opt/venv311/bin/python
ENV PYSPARK_DRIVER_PYTHON=/opt/venv311/bin/python

WORKDIR /opt/project

COPY fraud_pipeline /opt/project/fraud_pipeline
COPY model /opt/project/model
COPY spark-app /opt/project/spark-app-build

ENV PYTHONPATH=/opt/project:/opt/spark/app

RUN sed -i 's/\r$//' /opt/project/spark-app-build/entrypoint.sh \
    && chmod +x /opt/project/spark-app-build/entrypoint.sh

CMD ["/opt/project/spark-app-build/entrypoint.sh"]



