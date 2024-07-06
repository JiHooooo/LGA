import albumentations as A
#Convert image to torch.Tensor and divide by 255 if image or mask are uint8 type.
# from albumentations.pytorch import ToTensor
import cv2
import numpy as np
 # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
def get_augmentation_gray(image_size, train_flag=True):
    #image_size tuple or list of [height, width]
    small_len = min(image_size)
    argument_list = []
    if train_flag:
        argument_list.extend([
            A.Resize(height=image_size[0], width=image_size[1], p=1.0),
            A.RandomResizedCrop(height=image_size[0], width=image_size[1], 
                               scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), 
                               interpolation=cv2.INTER_LINEAR, p=1.0),
            
            A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=(-0.05,0.05),
                               rotate_limit=10, border_mode=0 , value=0, p=0.3),
                                                            
            
           
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0, hue=0, p=0.5),
            
        ])
    else:
        argument_list.extend([A.Resize(height=image_size[0], width=image_size[1], p=1.0),
                           
                             ])
    print(argument_list)
    return A.Compose(argument_list)