FROM jupyter/r-notebook:latest

USER root
RUN sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /home/jupyter/work

COPY requirements.txt requirements-base.txt
COPY ../requirements.txt requirements.txt

RUN pip install -r requirements-base.txt
RUN pip install -r requirements.txt

RUN rm requirements-base.txt
RUN rm requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
