import os
import cv2
import numpy as np

def save_color():
    color_list = []
    for name in os.listdir('./Polyp/train/image/'):
        print(name)
        image     = cv2.imread('./Polyp/train/image/'+name)
        image     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        mean, std = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
        color_list.append((mean, std))

    for folder in os.listdir('./LDPolypVideo/image'):
        print(folder)
        for name in os.listdir('./LDPolypVideo/image/'+folder):
            image     = cv2.imread('./LDPolypVideo/image/'+folder+'/'+name)
            image     = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            mean, std = image.mean(axis=(0,1), keepdims=True), image.std(axis=(0,1), keepdims=True)
            color_list.append((mean, std))
    np.save('color.npy', color_list)

if __name__=='__main__':
    save_color()
