FROM jupyter/r-notebook:x86_64-ubuntu-22.04

USER root
RUN sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /home/jupyter/work
    
COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install --no-interaction --no-ansi
    
EXPOSE 8888

CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
