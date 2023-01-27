import cv2
from build_dataset import *
from patchify import get_patch
from torch.utils.data import DataLoader

class slic:
    def __init__(self, path, numSegments, ratio, size):
        self.path = path
        self.numSegments = numSegments
        self.ratio = ratio
        self.size = size  
    
    def build_dataset(self):
        dataset = CustomDataset(self.path)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
        return data_loader
    
    def get_patches(self):
        data_loader = self.build_dataset()
        
        images_list = []
        for images, labels in data_loader:
            image = images[0]
            image = image.permute(1, 2, 0)
            
            images_list.append(image)
            num_images = len(images_list)
            # print(num_images)
            
        # get patches & save them
        patches_list = []
        num_patch_list = []
        
        for i in range(len(images_list)):          
            images_list[i] = images_list[i].numpy()
            patches = get_patch(images_list[i], self.numSegments, self.ratio)
            
            number_patches = len(patches)
            # print(number_patches)
            num_patch_list.append(number_patches)
            
            for j in range(number_patches):
                patch = patches[j]
                
                # reshape the patches
                patch = cv2.resize(patch, (self.size, self.size))
                
                # i is the image number, j is the patch number
                # cv2.imwrite("patches/patch_{}_{}.png".format(i, j), patch)
                
                patches_list.append(patch)
                
        # patches_list = np.array(patches_list)
        return patches_list, num_patch_list 
        
        
            
        
    
if __name__ == "__main__":
    path = "/home/lma/Summer-project-1/SLIC/animals"
    
    slic = slic(path, numSegments=100, ratio=0.9, size=16)
    patch_list, num_patch_list = slic.get_patches()
    
    print(len(patch_list))
    print(patch_list[0].shape)
    print(num_patch_list)