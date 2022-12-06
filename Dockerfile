FROM --platform=linux/amd64 tensorflow/tensorflow
COPY biomassters /biomassters
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
CMD uvicorn biomassters.api.fast:app --host 0.0.0.0 --port $PORT
