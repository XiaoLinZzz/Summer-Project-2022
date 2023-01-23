from skimage.segmentation import slic
from skimage.measure import regionprops

import cv2
import numpy as np
import torch

from detect_pixels import is_pixel_in_box

def patchify_slic(images, n_patches, size):
    '''
        input: images (N, C, H, W)
        
        output: patches (N, num_patches, size * size * C)
    ''' 
    n, c, h, w = images.shape
    
    for idx, image in enumerate(images):
        image = image.permute(1, 2, 0)
        image = image.numpy()
        
        segments = slic(image, n_segments = n_patches, sigma = 5, start_label=1)
        num_segments = len(np.unique(segments))
        num_patches = num_segments
        
        # test if all pixels are in box 
        assert is_pixel_in_box(segments) == True
        patches = torch.zeros(n, num_patches, size * size * c) 
        
        for i in range(num_patches):
            for region in regionprops(segments):
                y1, x1, y2, x2 = region.bbox
                patch = image[y1:y2, x1:x2]
                patch = cv2.resize(patch, (size, size))
                
                # tranform to tensor
                patch = torch.from_numpy(patch)
                patches[idx, i] = patch.flatten()
    
    return patches 