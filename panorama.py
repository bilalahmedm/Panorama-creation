#####TUWIEN - WS2022 CV: Task2 - Image Stitching
#####*********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_simple(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Stitch the final panorama with the calculated panorama extents
    by transforming every image to the same coordinate system as the center image. Use the dot product
    of the translation matrix 'T' and the homography per image 'H' as transformation matrix.
    HINT: cv2.warpPerspective(..), cv2.addWeighted(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) panorama image ([height x width x 3])
    """
    
    # student_code start

    stitch1 = cv2.warpPerspective(images[0], np.matmul(T,H[0]), (width, height))
    stitch2 = cv2.warpPerspective(images[1], np.matmul(T,H[1]), (width, height))
    center = cv2.warpAffine(images[2],T[:][:2],(width,height))
    stitch4 = cv2.warpPerspective(images[3], np.matmul(T,H[2]), (width, height))
    stitch5 = cv2.warpPerspective(images[4], np.matmul(T,H[3]), (width, height))
    result = cv2.addWeighted(stitch1,1,stitch2,1,0)
    result = cv2.addWeighted(result,1,stitch4,1,0)
    result = cv2.addWeighted(result,1,stitch5,1,0)
    result = cv2.addWeighted(result,1,center,1,0)

    # student_code end
        
    return result


def get_blended(images: List[np.ndarray], width: int, height: int, H: List[np.ndarray], T: np.ndarray) -> np.ndarray:
    """
    Use the equation from the assignment description to overlay transformed
    images by blending the overlapping colors with the respective alpha values
    HINT: ndimage.distance_transform_edt(..)
    
    Parameters
    ----------
    images : List[np.ndarray]
        list of images
    width : int
        width of panorama (in pixel)
    height : int
        height of panorama (in pixel)
    H : List[np.ndarray]
        list of homographies to center image ([number_of_images x 3 x 3])
    T : np.ndarray
        translation matrix for panorama ([3 x 3])

    Returns
    ---------
    np.ndarray
        (result) blended panorama image ([height x width x 3])
    """
    
    # student_code start
    
    mask = np.zeros((512,384),dtype='uint8')
    mask = cv2.rectangle(mask,(0,0),(384,512),(255,255,255),3)
    mask = cv2.bitwise_not(mask)
    alpha = ndimage.distance_transform_edt(mask)
    alpha = cv2.resize(alpha,(images[0].shape[1],images[0].shape[0]))
    alpha1 = cv2.warpPerspective(alpha, np.matmul(T,H[0]), (width, height))
    alpha2 = cv2.warpPerspective(alpha, np.matmul(T,H[1]), (width, height))
    alpha3 = cv2.warpAffine(alpha,T[:][:2],(width,height))
    alpha4 = cv2.warpPerspective(alpha, np.matmul(T,H[2]), (width, height))
    alpha5 = cv2.warpPerspective(alpha, np.matmul(T,H[3]), (width, height))

    stitch1 = cv2.warpPerspective(images[0], np.matmul(T,H[0]), (width, height))
    stitch2 = cv2.warpPerspective(images[1], np.matmul(T,H[1]), (width, height))
    center = cv2.warpAffine(images[2],T[:][:2],(width,height))
    stitch4 = cv2.warpPerspective(images[3], np.matmul(T,H[2]), (width, height))
    stitch5 = cv2.warpPerspective(images[4], np.matmul(T,H[3]), (width, height))

    result1 = 0
    result2 = 0
    result3 = 0
    a_ = [alpha1,alpha2,alpha3,alpha4,alpha5]
    trans_img = [stitch1,stitch2,center,stitch4,stitch5]
    np.seterr(invalid='ignore')
    for i in range(len(a_)):
        a_sum = sum(a_)
        result1 += ((trans_img[i][:,:,0]*a_[i])/a_sum)
    for i in range(len(a_)):
        a_sum = sum(a_)
        result2 += ((trans_img[i][:,:,1]*a_[i])/a_sum)
    for i in range(len(a_)):
        a_sum = sum(a_)
        result3 += ((trans_img[i][:,:,2]*a_[i])/a_sum)

    result = cv2.merge((result1,result2,result3)).astype('uint8')
    
    #raise NotImplementedError("TO DO in panorama.py")
    # student_code end

    return result
