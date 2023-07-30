FROM python:3.8.12-slim

RUN apt-get update && apt-get install -y git
RUN apt-get -y install libpq-dev gcc
RUN apt-get install -y python3-psycopg2
RUN mkdir ml-ops-zoomcamp-project
COPY . /ml-ops-zoomcamp-project 
WORKDIR /ml-ops-zoomcamp-project

# install pipenv and dependencies
ENV PIPENV_VENV_IN_PROJECT=1
ENV PYTHONPATH "${PYTHONPATH}:/ml-ops-zoomcamp-project"
RUN pip install pipenv
RUN pipenv install --dev
RUN pipenv sync --system
