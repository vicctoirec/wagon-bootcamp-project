# # TODO: select a base image
# # Tip: start with a full base image, and then see if you can optimize with
# #      a slim or tensorflow base

# #      Standard version
# FROM python:3.12

# #      Slim version
# # FROM python:3.12-slim

# #      Tensorflow version (attention: won't run on Apple Silicon)
# # FROM tensorflow/tensorflow:2.16.1

# # Install requirements
# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy our code
# COPY packagename packagename
# COPY api api

# # Make directories that we need, but that are not included in the COPY
# RUN mkdir /raw_data
# RUN mkdir /models

# # COPY credentials.json credentials.json

# # TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# # RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'

# CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT



FROM python:3.10.6-slim

WORKDIR /app

COPY requirements_prod.txt requirements_prod.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements_prod.txt
RUN pip install --no-cache-dir --no-deps sentence-transformers==4.1.0
RUN pip install -U pip

COPY models models
COPY raw_data raw_data
COPY api api
COPY ai_spotify_lyrics ai_spotify_lyrics

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
