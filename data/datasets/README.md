## Datasets

This directory contains the datasets used in the project. The preferred pattern for adding new datasets is under a descriptively-named subdirectory. For example, the `aircraft` dataset is stored under `data/datasets/aircraft`.

The datasets are not directly tracked in git, so any analysis or data loading code should be omited from the `data/datasets` directory. Through the magic of Docker, the datasets will be mounted into your Jupyter lab server at runtime. This decouples the data from the code, and ensures that everyone on the team build, run, and collaborate on the project across different machines.

### The `aircraft` dataset

The `aircraft` dataset is distributed through [Kaggle](https://www.kaggle.com/docs), which is itself an excellent learning resource. Please download the dataset from [this link](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset) (requires registration or a Google account).

Once downloaded, extract the contents of the zipped file to a folder called `aircraft` in the `data/datasets/` directory, such that the structure matches the example below:

```
data/
├── datasets/
  ├── aircraft/
  │   ├── annotations.csv
  │   ├── extras/
  │   ├── images/
  │   ├── ...
```

### The `tornado` dataset

The `tornado` dataset is distributed through [the FEMA ARC-GIS Hub](https://gis-fema.hub.arcgis.com/datasets/e75412d18bdc469dbf89bf7e929475cc/explore). While the data is available in multple formats. please select the GeoJSON format.

Once downloaded, place the `Tornado_Tracks.geojson` file in a folder called `tornado` in the `data/datasets/` directory, such that the structure matches the example below:

```
data/
├── datasets/
  ├── tornado/
  │   ├── Tornado_Tracks.geojson
  │   ├── ...
```


