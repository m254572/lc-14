FROM jupyter/r-notebook:x86_64-ubuntu-22.04

USER root
RUN sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /home/jupyter/work
    
COPY requirements.txt .
    
RUN pip install -r requirements.txt
    
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
