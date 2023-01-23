# import packages
from skimage.segmentation import slic
from skimage.measure import regionprops

import cv2
import numpy as np
import random
import torch

from detect_pixels import is_pixel_in_box

def get_patch(image, num_segments, ratio):
    segments = slic(image, n_segments = num_segments, sigma = 5, start_label=1)
    num_superpixels = len(np.unique(segments))
    
    # get masks list
    masks = random.sample(range(1, num_superpixels), k = int(ratio * num_superpixels))
    
    # number of not masked superpixels
    num_slected = len(np.unique(segments)) - len(np.unique(masks))
    
    # create masked image
    for i in masks:
        segments[segments == i] = 0

    # mask image
    image[segments == 0] = 0
    
    # test if segments are masked
    assert len(np.unique(segments)) in range(num_slected, num_superpixels + 1)
    
    # test if all pixels are in box
    assert is_pixel_in_box(segments) == True
    
    # define patches container
    patches_container = []
    original_image = image.copy()
    original_segments = segments.copy()
    
    for i in np.unique(segments):
        segments[segments != i] = 0
        image[segments == 0] = 0
        
        for region in regionprops(segments):
            minr, minc, maxr, maxc = region.bbox
            
            # add patches to container
            patches = image[minr:maxr, minc:maxc]
            patches_container.append(patches)
            
        segments = original_segments.copy()
        image = original_image.copy()
    
    # return a list of patches
    return patches_container