from fastapi import FastAPI, File, UploadFile, Request
app = FastAPI()
import cProfile
# import pickle
# import numpy as np
# # from keras import preprocess_input
# from keras.applications.imagenet_utils import preprocess_input
# from PIL import Image
# from keras.applications import VGG16
# # from keras.models import load_model
# from joblib import load
# from sklearn.decomposition import PCA
import asyncio
import time 
from starlette.concurrency import run_in_threadpool
import concurrent.futures
from fastapi import BackgroundTasks

# from celery import Celery

# celery_app = Celery('tasks', broker='pyamqp://guest@localhost//')

# @celery_app.task
# def add():
#     calculate_pi(1000000)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

import asyncio
import functools
import multiprocessing

async def take_time():
    time.sleep(1)
    return {"message": "Hello World"}

def worker(num_iterations):
    pi = 0
    sign = 1
    for i in range(num_iterations):
        pi += sign * 4 / (2 * i + 1)
        sign = -sign
    print(pi)
    return pi

from aiohttp import ClientSession

async def calculate_pi(num_iterations):
    return await asyncio.to_thread(worker, num_iterations)

@app.get("/")
async def root(background_tasks: BackgroundTasks):
    pr = cProfile.Profile()
    # pr.enable()
    start_time = time.time()
    x = await calculate_pi(1000000)
    y = await calculate_pi(1000000000000)
    end_time = time.time()
    # pr.disable()
    # pr.print_stats(sort='time')
    # pr.dump_stats('profile.prof')
    return (end_time - start_time) * 1000

# route to add add latency to the response, will be called by / route
@app.get("/2")
async def root2():
    time.sleep(5)
    return {"message": "Hello World"}

@app.post("/input-bird-photo/")
def input_bird_photo(bird_photo: UploadFile = File(...)):
    print(bird_photo.filename)
    with open(bird_photo.filename, 'wb') as f:
        f.write(bird_photo.file.read())
    bird_photo = Image.open(bird_photo.filename)
    bird_name, prediction_conf = identify_bird(bird_photo)
    return {"bird_type": bird_name, "confidence": prediction_conf}

def identify_bird(bird_photo):
#    return "bird_name"
    pca = PCA(n_components = 1000)
    # loads train_image.pkl
    with open('runner/pca_fit.pkl', 'rb') as pca_fit:
        pca_fit = pickle.load(pca_fit)
    

    # pca_fit=pca.fit(train_images)
    # model = VGG16(weights='imagenet', include_top=False)
    # # predicted_species = classifier.predict(img)
    with open('runner/svc_model.pkl', 'rb') as file:
        pickle_model = pickle.load(file)

    #load cookapoo.jpg and convert it to a numpy array to be used in the model
    # with open('/Users/tyler/Desktop/PetProjects/bird-watch/cookapoo.jpeg', 'rb') as file:
    #     # Load the image
    img = bird_photo
    img = img.resize((224, 224))
    img = np.array(img)
    img = img.reshape(1, -1)

    # code to fix error: X has 150528 features, but SVC is expecting 1000 features as input.
    # This is because the model was trained on 1000 features, but the image we are trying to predict is 150528 features.
    # We need to reduce the image to 1000 features.
    img = pca_fit.transform(img)
    print(img.shape)

        
    # Predict the species of the bird 
    predicted_species = pickle_model.predict(img)[0]
    print(predicted_species)
    prediction_confidence = max(pickle_model.predict_proba(img)[0])*100
    print(prediction_confidence)



    return predicted_species, prediction_confidence