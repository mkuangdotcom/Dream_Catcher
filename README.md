## DREAM CATCHER 🛏️

The Dream Catcher project analyzes head movements during sleep using a combination of computer vision and data analysis tools. It leverages OpenCV for image and video processing, MediaPipe for face mesh detection, and NumPy for numerical data manipulation. The head movements are tracked and analyzed, and the data is organized into tables using Pandas. Finally, the results are visualized using Matplotlib.
<br> <br>

## Installation

To get started, you need to install the required packages. You can do this using pip:

```sh
pip install opencv-python
pip install numpy
pip install mediapipe
pip install pandas
pip install matplotlib
```
<br>

## How It Works?  
Video Input: The system takes a video of a person sleeping as input. <br>
Head Detection: Using OpenCV and MediaPipe, the program detects the head and tracks its movement. <br>
Data Collection: The head movement data is collected and organized into a table using Pandas. <br>
Visualization: The collected data is then visualized using Matplotlib to show the patterns and extent of head movements during sleep. <br>
<br>

# Key Points in the Code
**Import Libraries:** Essential libraries for computer vision, numerical computations, data manipulation, and visualization are imported.

```sh
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt
from head_detection import detect_head
```

<br>

**Head Detection:** The detect_head function from head_detection.py is used to detect the head in each frame of the video.

```sh
def detect_head(frame):
    # Code for head detection using MediaPipe
    ...
```

<br>

**Data Processing:** The detected head positions are stored in a Pandas DataFrame for further analysis.

```sh
data = pd.DataFrame(columns=['timestamp', 'head_position'])
# Code to populate the DataFrame
...
```

<br>

**Visualization:** Matplotlib is used to plot the head movements over time.

```sh
plt.plot(data['timestamp'], data['head_position'])
plt.xlabel('Time')
plt.ylabel('Head Position')
plt.title('Head Movements During Sleep')
plt.show()
```

<br>
