## Datasets

This directory contains the datasets used in the project. The preferred pattern for adding new datasets is under a descriptively-named subdirectory. For example, the `aircraft` dataset is stored under `data/datasets/aircraft`.

The datasets are not directly tracked in git, so any analysis or data loading code should be omited from the `data/datasets` directory. Through the magic of Docker, the datasets will be mounted into your Jupyter lab server at runtime. This decouples the data from the code, and ensures that everyone on the team build, run, and collaborate on the project across different machines.
