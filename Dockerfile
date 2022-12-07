FROM --platform=linux/amd64 tensorflow/tensorflow
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN mkdir data
COPY data/features_metadata.csv /data/features_metadata.csv
COPY data/train_agbm_metadata.csv /data/train_agbm_metadata.csv
COPY biomassters /biomassters
CMD uvicorn biomassters.api.fast:app --host 0.0.0.0 --port $PORT
