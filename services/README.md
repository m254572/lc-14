## Overview

This repository includes a set of pre-defined services designed to be run directly on your issued laptop.

For resource intensive applications, these services can also be hosted remotely (on a machine connected to the local network). If hosted remotely, the services will be accessible through your browser, but the data will be stored on the remote machine.

### Requirements

These services require only one dependency on your machine when you initally set up your project. The rest of the dependencies will be managed through the services indirectly. At this point in your academic career, you are familiar with Python, but Docker may be new to you. Docker is a platform that allows you to develop, ship, and run applications in a containerized environment. Containers are lightweight, standalone, and executable packages that include everything needed to run an application.

#### [Docker](https://www.docker.com/)

The easiest way to install Docker is to download Docker Desktop. Docker Desktop is an easy-to-install application for your Windows environment that enables you to build the containerized services. In order for the docker engine to be available, Docker Desktop must be running on your machine.

Once you've downloaded Docker Desktop, you can verify that the installation was successful by running the following command in your terminal:

```bash
docker --version
```

If you see a version number, the docker engine is available to build the services locally.

## Services

### Frontend services

#### [Jupyter Notebook](https://jupyter.org/)

Jupyter notebooks make it easy to quickly expirament with different approaches to solving a problem. In addition to running code, Jupyter notebooks can display visualizations and text. This makes it easy to share your work with others, and to understand the state of your expiraments and models at each phase of their development.

They also make running code and expiraments repeatable and reproducible. While not strictly required, Jupyter environements provide an excellent foundation for getting up to speed on your project and tracking progress over time.

Once the services are running, you can access the Jupyter notebook by navigating to `http://localhost:8888` in your browser. Alternatively, if you prefer developing in an IDE such as VSCode, you can install the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-jupyter) and connect to the Jupyter notebook server by using `existing server` and entering `http://localhost:8888` when selecting your python interpreter.

#### [MLFlow](https://mlflow.org/)

MLFlow is an open-source platform for managing the end-to-end machine learning lifecycle. It allows you to track experiments, package code into reproducible runs, and share and deploy models. MLFlow is designed to work with any machine learning library and language, and integrates with the tools and libraries you're already using. There are especially strong integations with popular libraries like PyTorch, and Scikit-Learn.

Both MLFlow and the MLFlow UI are not required for the project, but they are available for you to use if you would like. Once the services are running, you can access the MLFlow UI by navigating to `http://localhost:5000` in your browser. The MLFlow UI is an intuitive and visual way to compare different runs and track the performance of your models over time. This data is also accessible through the MLFlow API, which is avaialbe to you in your Jupyter notebook.

### Backend services

You will not need to interact with these backend services directly in completing your project, but it is somewhat useful to have a broad understanding of their purpose and usage. These services are used to store data and serve the frontend services, and are more important when the services are hosted remotely.

#### [MinIO](https://min.io/)



#### [NGINX](https://www.nginx.com/)

### Usage

To start the services, run the following command:

```bash
docker-compose up
```

To stop the services, run the following command:

```bash
docker-compose down
```


