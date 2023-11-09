#!/bin/bash

# Set the python path
export PYTHONPATH=$PYTHONPATH:/Users/tyler/Desktop/PetProjects/bird-watch/venv/lib/python3.9/site-packages

# Start the Uvicorn server
uvicorn main:app --host 0.0.0.0 --port 80 --workers 10
