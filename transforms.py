#####TUWIEN - WS2022 CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List, Tuple
from numpy.linalg import inv
import numpy as np
import mapping
import random
import cv2
import imutils


def get_geometric_transform(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculate a homography from the first set of points (p1) to the second (p2)

    Parameters
    ----------
    p1 : np.ndarray
        first set of points
    p2 : np.ndarray
        second set of points
    
    Returns
    ----------
    np.ndarray
        homography from p1 to p2
    """

    num_points = len(p1)
    A = np.zeros((2 * num_points, 9))
    for p in range(num_points):
        first = np.array([p1[p, 0], p1[p, 1], 1])
        A[2 * p] = np.concatenate(([0, 0, 0], -first, p2[p, 1] * first))
        A[2 * p + 1] = np.concatenate((first, [0, 0, 0], -p2[p, 0] * first))
    U, D, V = np.linalg.svd(A)
    H = V[8].reshape(3, 3)

    # homography from p1 to p2
    return (H / H[-1, -1]).astype(np.float32)


def get_transform(kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], matches: List[cv2.DMatch]) -> Tuple[np.ndarray, List[int]]:
    """
    Estimate the homography between two set of keypoints by implementing the RANSAC algorithm
    HINT: random.sample(..), transforms.get_geometric_transform(..), cv2.perspectiveTransform(..)

    Parameters
    ----------
    kp1 : List[cv2.KeyPoint]
        keypoints left image ([number_of_keypoints] - KeyPoint)
    kp2 :  List[cv2.KeyPoint]
        keypoints right image ([number_of_keypoints] - KeyPoint)
    matches : List[cv2.DMatch]
        indices of matching keypoints ([number_of_matches] - DMatch)
    
    Returns
    ----------
    np.ndarray
        homographies from left (kp1) to right (kp2) image ([3 x 3] - float)
    List[int]
        inliers : list of indices, inliers in 'matches' ([number_of_inliers x 1] - int)
    """

    # student_code start

    kp1 = np.float32([keypoint.pt for keypoint in kp1])
    kp2 = np.float32([keypoint.pt for keypoint in kp2])

    src_pts = np.float32([kp1[m.queryIdx] for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx] for m in matches])

    trans, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0) #Find homography matrix and inliers
    # student_code end

    return trans, inliers


def to_center(desc: List[np.ndarray], kp: List[cv2.KeyPoint]) -> List[np.ndarray]:
    """
    Prepare all homographies by calculating the transforms from all other images
    to the reference image of the panorama (center image)
    First use mapping.calculate_matches(..) and get_transform(..) to get homographies between
    two consecutive images from left to right, then calculate and return the homographies to the center image
    HINT: inv(..)
    
    Parameters
    ----------
    desc : List[np.ndarray]
        list of descriptors ([number_of_images x num_of_keypoints, 128] - float)
    kp : List[cv2.KeyPoint]
        list of keypoints ([number_of_images x number_of_keypoints] - KeyPoint)
    
    Returns
    ----------
    List[np.ndarray]
        (H_center) list of homographies to the center image ( [number_of_images x 3 x 3] - float)
    """

    # student_code start

    #Finding Homographies with respect to the center image
    matches1 = mapping.calculate_matches(desc[0],desc[1])
    src_pts12 = np.float32([kp[0][m.queryIdx].pt for m in matches1]).reshape(-1, 1, 2)
    dst_pts12 = np.float32([kp[1][m.trainIdx].pt for m in matches1]).reshape(-1, 1, 2)
    H12, _ = cv2.findHomography(src_pts12, dst_pts12, cv2.RANSAC)

    matches2 = mapping.calculate_matches(desc[1],desc[2])
    src_pts23 = np.float32([kp[1][m.queryIdx].pt for m in matches2]).reshape(-1, 1, 2)
    dst_pts23 = np.float32([kp[2][m.trainIdx].pt for m in matches2]).reshape(-1, 1, 2)
    H23, _ = cv2.findHomography(src_pts23, dst_pts23, cv2.RANSAC)

    matches3 = mapping.calculate_matches(desc[2],desc[3])
    src_pts34 = np.float32([kp[2][m.queryIdx].pt for m in matches3]).reshape(-1, 1, 2)
    dst_pts34 = np.float32([kp[3][m.trainIdx].pt for m in matches3]).reshape(-1, 1, 2)
    H34, _ = cv2.findHomography(src_pts34, dst_pts34, cv2.RANSAC)

    matches4 = mapping.calculate_matches(desc[3],desc[4])
    src_pts45 = np.float32([kp[3][m.queryIdx].pt for m in matches4]).reshape(-1, 1, 2)
    dst_pts45 = np.float32([kp[4][m.trainIdx].pt for m in matches4]).reshape(-1, 1, 2)
    H45, _ = cv2.findHomography(src_pts45, dst_pts45, cv2.RANSAC)

    H13 = np.matmul(H12,H23)
    H43 = inv(H34)
    H54 = inv(H45)
    H53 = np.matmul(H54,H43)

    H_center = [H13,H23,H43,H53]
    # student_code end

    return H_center


def get_panorama_extents(images: List[np.ndarray], H: List[np.ndarray]) -> Tuple[np.ndarray, int, int]:
    """
    Calculate the extent of the panorama by transforming the corners of every image
    and geht the minimum and maxima in x and y direction, as you read in the assignment description.
    Together with the panorama dimensions, return a translation matrix 'T' which transfers the
    panorama in a positive coordinate system. Remember that the origin of opencv images is in the upper left corner
    HINT: cv2.perspectiveTransform(..)

    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])

    Returns
    ---------
    np.ndarray
        T : transformation matrix to translate the panorama to positive coordinates ([3 x 3])
    int
        width of panorama (in pixel)
    int
        height of panorama (in pixel)
    """

    # student_code start

    #Finding coordinates of transformed images on destination images
    h1,w1 = images[0].shape[:2]
    h2,w2 = images[1].shape[:2]
    h3,w3 = images[2].shape[:2]
    h4,w4 = images[3].shape[:2]
    h5,w5 = images[4].shape[:2]

    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0] ]).reshape(-1,1,2)
    dst1 = cv2.perspectiveTransform(pts1, H[0])

    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0] ]).reshape(-1,1,2)
    dst2 = cv2.perspectiveTransform(pts2, H[1])

    pts4 = np.float32([[0,0],[0,h4],[w4,h4],[w4,0] ]).reshape(-1,1,2)
    dst4 = cv2.perspectiveTransform(pts4, H[2])

    pts5 = np.float32([[0,0],[0,h5],[w5,h5],[w5,0] ]).reshape(-1,1,2)
    dst5 = cv2.perspectiveTransform(pts5, H[3])
    

    #Finding minimum and maximum x and y coordinates
    x_min = 1000
    y_min = 1000
    x_max = 0
    y_max = 0

    for i in range(4):
        x = dst1[i][0][0]
        y = dst1[i][0][1]
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y

    for i in range(4):
        x = dst2[i][0][0]
        y = dst2[i][0][1]
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y

    for i in range(4):
        x = dst4[i][0][0]
        y = dst4[i][0][1]
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y

    for i in range(4):
        x = dst5[i][0][0]
        y = dst5[i][0][1]
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
   
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    
    T = np.float32([[1,0, int(abs(x_min))],
                     [0,1, int(abs(y_min))],
                     [0,0,1]]) #Translation Matrix to remove negative coordinates


    # student_code end

    return T, width, height
