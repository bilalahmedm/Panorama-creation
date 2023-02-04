#####TUWIEN - WS2022 CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List, Tuple
from matplotlib.pyplot import angle_spectrum
import numpy as np
# from cyvlfeat.sift.sift import sift
# import cyvlfeat
import os
import cv2


def get_panorama_data(path: str) -> Tuple[List[np.ndarray], List[cv2.KeyPoint], List[np.ndarray]]:
    """
    Loop through images in given folder, extract SIFT points
    and return images, keypoints and descriptors
    This time we need to work with color images. Since OpenCV uses BGR you need to swap the channels to RGB
    HINT: os.walk(..), cv2.imread(..), sift=cv2.SIFT_create(), sift.detectAndCompute(..)
    
    Parameters
    ----------
    path : str
        path to image folder

    Returns
    ---------
    List[np.ndarray]
        img_data : list of images
    List[cv2.Keypoint]
        all_keypoints : list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    List[np.ndarray]
        all_descriptors : list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    """
    
    # student_code start
    images = os.listdir(path)
    img_data = []
    all_keypoints = []
    all_descriptors = []
    
    for img in images:
        image = cv2.imread(os.path.join(path,img))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        sift = cv2.xfeatures2d.SIFT_create()

        kp, des = sift.detectAndCompute(image,None)

        img_data.append(image)
        all_keypoints.append(kp)
        all_descriptors.append(des)
    #raise NotImplementedError("TO DO in dataset.py")
       
    # student_code end

    return img_data, all_keypoints, all_descriptors
