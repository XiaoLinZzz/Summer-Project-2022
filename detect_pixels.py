from skimage.measure import regionprops

def is_pixel_in_box(segments):
    for region in regionprops(segments):
        # get the coordinates of the region
        minr, minc, maxr, maxc = region.bbox
        coord = region.coords
        
        for i in coord:
            # if coords are not in box, return false
            if i[0] <= minr and i[0] >= maxr and i[1] <= minc and i[1] >= maxc:
                return False
            else:
                return True