# AR Tag Detection and Tracking

## Description

Given a video where an AR Tag moves with respect to the camera, this project attempts to detect the tag, warp it, find its orientation and superimpose a 'Testudo' image as well as a 3D cube on it.


## Data

* AR Tag Video

![alt text](/data/tagvideo.gif)

* Testudo Image for Superimposition

![alt text](/data/testudo.png)

* Intrinsic Parameters of the Camera


## Approach

* Read the video frame by frame.

* Isolate the AR tag and find the coordinates of its corners.

* Warp the tag using Homography without using inbuilt function and implementing a vectorized approach.

* Find the orientation of the tag.

* Roatate the image based on the orientation.

* Superimpose the image on the AR tag without using inbuilt function and a vectorized approach.

* Based on the corners and the intrinsic camera parameters, find the Projection Matrix.

* Draw a cube considering all the points found using the Projection Matrix.


## Output

* Isolating the AR Tag

![alt text](/output/output1.png)

* Find Corners

![alt text](/output/output2.png)

* Homography

![alt text](/output/output3.png)

* Warping

![alt text](/output/output4.png)

* Orienting the Tag

![alt text](/output/output5.png)

* Superimposing the 'Testudo' image

![alt text](/output/output6.png)

* Superimposing the 3D Cube

![alt text](/output/output7.png)

* Final Output for image superimposition [Link](https://drive.google.com/file/d/1rMePdOe4IaK8dgb-byewfD2OvjbIvS2t/view?usp=sharing)

![alt text](/output/outputvid1.gif)

* Final Output for 3D Cube superimposition [Link](https://drive.google.com/file/d/1TePCedJXcSskpTHvCKJ925YS9FoHwXCV/view?usp=sharing)

![alt text](/output/outputvid2.gif)


## Getting Started

### Dependencies

<p align="left"> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>&ensp; </a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://www.codebykelvin.com/learning/python/data-science/numpy-series/cover-numpy.png" alt="numpy" width="40" height="40"/>&ensp; </a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://avatars.githubusercontent.com/u/5009934?v=4&s=400" alt="opencv" width="40" height="40"/>&ensp; </a>

* [Python 3](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)


### Executing program

* Clone the repository into any folder of your choice.
```
git clone https://github.com/ninadharish/AR-Tag-Detection-and-Tracking.git
```

* Open the repository and navigate to the `src` folder.
```
cd AR-Tag-Detection-and-Tracking/src
```
* Depending on whether you want to superimpose athe image or 3D cube on the tag, comment/uncomment the proper line.

* Run the program.
```
python main.py
```


## Authors

👤 **Ninad Harishchandrakar**

* [GitHub](https://github.com/ninadharish)
* [Email](mailto:ninad.harish@gmail.com)
* [LinkedIn](https://linkedin.com/in/ninadharish)
