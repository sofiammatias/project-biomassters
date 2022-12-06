from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from biomassters.ml_logic.registry import load_model
from biomassters.ml_logic.model import image_to_np
from typing import List
from tensorflow import image, expand_dims
import tensorflow
import pandas as pd
import numpy as np

app = FastAPI()

app.state.model = load_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/predict/")
async def create_upload_file(files: List[UploadFile]):

    content = await files[0].read()
    array = np.frombuffer(content, dtype=np.float32)
    array = array.reshape((256,256,4))

    img1 = image.per_image_standardization(array)
    X1_pred = expand_dims(img1, axis=0)


    content = await files[1].read()
    array = np.frombuffer(content, dtype=np.int16)
    array = array.reshape((256,256,11))
    img2 = image.per_image_standardization(array)
    X2_pred = expand_dims(img2, axis=0)

    prediction = app.state.model.predict([X1_pred, X2_pred])
    y_pred = tensorflow.squeeze(prediction)
    print (y_pred)
    return {'file': np.array(y_pred).tolist()}


@app.get("/")
def root():
    # Define a root `/` endpoint
    return {'greeting': 'Hello'}
