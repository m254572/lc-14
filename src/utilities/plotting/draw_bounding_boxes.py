from typing import List, Tuple, Optional, Dict, Union

import cv2
import pandas as pd
from IPython.display import Image, display


def bounds_from_geometry(geometry: List[Tuple[int, int]]) -> Union[Tuple[int, int, int, int], None]:
  """
  Obtain the coordinates of the bounding box from a GEOJson geometry.

  Args:
    geometry (list): The GEOJson geometry coordinates

  Returns:
    x_min, y_min, x_max, y_max (int, int, int, int): The coordinates of the bounding box.
  """
  if not geometry:
    return None

  # Unpack coordinates, ensuring they are integers
  x_coords, y_coords = zip(*((int(x), int(y)) for x, y in geometry))

  # Calculate min/max values
  x_min, x_max = min(x_coords), max(x_coords)
  y_min, y_max = min(y_coords), max(y_coords)

  return x_min, y_min, x_max, y_max

def draw_bounding_boxes(image_filepath, df: Optional[pd.DataFrame] = None, prediction: Optional[Dict[str, Union["np.ndarray", "torch.Tensor"]]] = None):
  """
  Displays an image with bounding boxes from a Pandas DataFrame.

  Args:
    image_filepath (str): Path to the image file.
    df (pd.DataFrame): Optional ground-truth DataFrame with GEOJson geometry from which to derive bounding boxes.
    prediction (dict): Optional predictions to draw on the image.

  Returns:
    None: An image is printed 
  """

  def _ground_truth_for_image(df: pd.DataFrame, image_filepath: str) -> None:
    """
    Filter the DataFrame for the current image.

    Args:
      df (pd.DataFrame): The DataFrame with ground-truth data.
      image_filepath (str): The path to the current image.

    Returns:
      None: the ground truth data for the current image is added to the plot.
    """
    nonlocal img

    image_name = image_filepath.split("/")[-1]
    image_df = df[df["image_id"] == image_name]

    # Draw bounding boxes for ground-truth if available
    for _, row in image_df.iterrows():
      label = row["class"]
      geometry = row["geometry"]

      x_min, y_min, x_max, y_max = bounds_from_geometry(geometry=geometry)

      # Add rectangle
      cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 200), 2)  # Red rectangle

      # Add label text
      cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
  
  def _predictions_for_image(prediction: Dict[str, Union["np.ndarray", "torch.Tensor"]], score_threshold: float = 0.8) -> None:
    """
    Filter the predictions for the current image.

    Args:
      prediction (dict): The predictions array.
      image_filepath (str): The path to the current image.

    Returns:
      None: the predictions for the current image is added to the plot.
    """
    nonlocal img

    # Iterate over the detections
    for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
      if score < score_threshold:
          continue

      x_min, y_min, x_max, y_max = map(int, box)

      # Add prediction bounding box
      cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (200, 0, 0), 2)  # Blue rectangle

      # Add label text
      label_text = f"{label}: {score:.2f}"
      cv2.putText(img, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

  # Load the image
  img = cv2.imread(image_filepath)

  if df is not None:
    # Filter the DataFrame for the current image
    _ground_truth_for_image(df, image_filepath)
  
  if prediction:
    # Filter the predictions for the current image
    _predictions_for_image(prediction)

  # Display the image using Ipython display
  _, encoded_img = cv2.imencode(".png", img)
  display(Image(data=encoded_img))
