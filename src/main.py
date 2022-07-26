from track_image import *
from track_3D_cube import *


if __name__ == "__main__":

    video = './data/tagvideo.mp4'
    image = './data/testudo.png'

    track_image(video, image)

    # track_cube(video)