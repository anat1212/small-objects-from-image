import numpy as np
import glob
from skimage import io
import cv2

""" put value from 1-99 in the percentage variable"""
percentage = 55
percentage_used = percentage / 100
counter = 0
print(counter)
""" choose file directory that stores images to be processed by the script"""
images = glob.glob("D:/clusterdelta/global-wheat-detection/train/*.jpg")
for name in images:
    im1 = io.imread(name)

    heit, lengz, num_layers = im1.shape
    pixel_total = (int(heit*lengz*num_layers))

    empty_list = []
    empty_list2 = []
    empty_list3 = []
    empty_list4 = []
    im1_reshaped = np.reshape(im1, (lengz*heit*num_layers))
    """group pixel amounts by value"""
    for i in range(256):
        pixels = np.count_nonzero(im1_reshaped == i)
        empty_list.append(pixels)
        empty_list2.append(i)

    """zip amounts to each value and reverse the array"""
    list_pixels2 = (sorted(zip(empty_list,empty_list2), reverse=True))
    for pixel in list_pixels2:
        """separate pixels to save and replace"""
        if sum(empty_list3) < (pixel_total * percentage_used):
            empty_list3.append(pixel[0])
            empty_list4.append(pixel[1])
        else:
            break

    """replace pixels with value 0"""
    for i in empty_list4:
        """put zero or any other value just before im1 variable"""
        im1 = (np.where(im1[:,:,0:1]==i, 0, im1)) 
        # im1 = cv2.cvtColor(im1, cv2.COLOR_RGBA2GRAY)
        # thresh = 128
        # im1 = cv2.threshold(im1, thresh, 255, cv2.THRESH_BINARY)[1]
        # im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
        print(im1.shape)


    # io.imshow(im1)
    # plt.show()

    counter += 1
     """below adjust if you need to modify output image name"""
    io.imsave(str(name[45:]), im1)
