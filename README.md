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
- Video Input: The system takes a video of a person sleeping as input. <br>
- Head Detection: Using OpenCV and MediaPipe, the program detects the head and tracks its movement. <br>
- Data Collection: The head movement data is collected and organized into a table using Pandas. <br>
- Visualization: The collected data is then visualized using Matplotlib to show the patterns and extent of head movements during sleep. <br>
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
<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/mkuangdotcom/Dream_Catcher/assets/150900178/94920b2f-52fa-4a09-a60b-4292ec7900e5" width="450"/>
        <br>
        Looking Forward
      </td>
      <td align="center">
        <img src="https://github.com/mkuangdotcom/Dream_Catcher/assets/150900178/34e7b991-53e3-40e8-9c65-2d586797b349" width="450"/>
        <br>
        Looking Left
      </td>
    </tr>
  </table>
</div>
<br>


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

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/mkuangdotcom/Dream_Catcher/assets/150900178/e260c146-f2ac-456a-9b1d-4487aacc4074" width="700"/>
        <br>
        Example Output
      </td>
    </tr>
  </table>
</div>
<br>




## Credits

This project was inspired by the work of Irfan Alghani Khalid on head pose estimation. You can read more about it on [Towards Data Science](https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600).

This project was also inspired by the work of Sergio Canu, the founder of Pysource. You can read more about it on [Pysource](https://pysource.com/2020/09/17/build-a-sleep-tracker-with-opencv-and-python/).

