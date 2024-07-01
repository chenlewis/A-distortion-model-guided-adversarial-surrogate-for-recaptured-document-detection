from skimage import io, data, exposure, img_as_float, filters, util
import os
import cv2
from tqdm.notebook import tqdm

"""
1、HalfTonging image
2、Halftoning image to recaptured image
"""

def saturation(img, increment):
    img = img * 1.0
    img_out = img * 1.0

    # -1 ~ 1
    Increment = increment

    img_min = img.min(axis=2)
    img_max = img.max(axis=2)

    Delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value/2.0

    mask_1 = L < 0.5

    s1 = Delta/(value + 0.001)
    s2 = Delta/(2 - value + 0.001)
    s = s1 * mask_1 + s2 * (1 - mask_1)

    if Increment >= 0 :
        temp = Increment + s
        mask_2 = temp >  1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - Increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1/(alpha + 0.001) -1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    else:
        alpha = Increment
        img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
        img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
        img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)


    img_out = img_out/255.0
    
    mask_1 = img_out  < 0
    mask_2 = img_out  > 1

    img_out = img_out * (1-mask_1)
    img_out = img_out * (1-mask_2) + mask_2

    return img_out


def InkHalfToning(genuine_img_path, save_dir):

    img = cv2.imread(genuine_img_path, cv2.IMREAD_COLOR)
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # error diffusion
    Height = GrayImage.shape[0]
    Width = GrayImage.shape[1]
    channel = GrayImage.shape[2]

    for i in tqdm(range(0, channel)):
        for y in tqdm(range(0, Height)):
            for x in range(0, Width):

                old_value = GrayImage[y, x, i]
                new_value = 0
                if (old_value > 128):
                    new_value = 255

                GrayImage[y, x, i] = new_value

                Error = old_value - new_value

                if (x < Width - 1):
                    NewNumber = GrayImage[y, x + 1, i] + Error * 7 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y, x + 1, i] = NewNumber

                if (x > 0 and y < Height - 1):
                    NewNumber = GrayImage[y + 1, x - 1, i] + Error * 3 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y + 1, x - 1, i] = NewNumber

                if (y < Height - 1):
                    NewNumber = GrayImage[y + 1, x, i] + Error * 5 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y + 1, x, i] = NewNumber

                if (y < Height - 1 and x < Width - 1):
                    NewNumber = GrayImage[y + 1, x + 1, i] + Error * 1 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y + 1, x + 1, i] = NewNumber

    halfTone_save_path = os.path.join(save_dir, genuine_img_path.split("/")[-1].split(".")[0] + "_inkHalftoning" + "." + genuine_img_path.split("/")[-1].split(".")[1])
    io.imsave(halfTone_save_path, GrayImage)
    
# Simulation of recaptured samples printed by inkjet, the halftoning images are input    
def InkHalfToningToRecapturedSample(halftoning_img_path, save_dir, gamma=1.5, increment=-0.5):
   
    # load
    img = io.imread(halftoning_img_path)
    img_out = saturation(img, increment) * 255.0
    img_out = img_out.astype('uint8')

    # Gaussian noise
    img_guassian_noise = util.random_noise(img_out, mode='gaussian')

    # Gaussian blur
    img_guassian = filters.gaussian(img_guassian_noise)

    # Gamma correction
    img_gamma = exposure.adjust_gamma(img_guassian, gamma)

    # save the simulated recaptured sample
    # JPEG compression
    # save_path = os.path.join(save_dir, "" + halftoning_img_path.split("/")[-1].split(".")[0].replace("_inkHalftoning", "_inkRecaptured") + '.jpg')
    save_path = os.path.join(save_dir, "" + halftoning_img_path.split("/")[-1].split(".")[0] + "_inkRecaptured" + '.jpg')
    io.imsave(save_path, img_gamma)

#  LaserHalfToning: We used Photoshop to simulated the laser halftone, which is in "Color Halftone" of "Pixelate" 

# Simulation of recaptured samples printed by laserjet, the halftoning images are input
def LaserHalftoningToRecapturedSample(halftoning_img_path, save_dir, gamma=0.9, increment=-0.5):
    
    # load
    img = io.imread(halftoning_img_path)
    img_out = img

    # Gaussian noise
    img_guassian_noise = util.random_noise(img_out, mode='gaussian', mean=mean, var=var)

    # Gaussian blur
    img_guassian = filters.gaussian(img_guassian_noise, sigma=sigma)                                                                                                                                      
    # Gamma correction
    img_gamma = exposure.adjust_gamma(img_guassian, gamma=gamma)

    # save the simulated recaptured sample
    # JPEG compression
    save_path = os.path.join(save_dir, "" + halftoning_img_path.split("/")[-1].split(".")[0].replace("_laserHalftoning", "_laserRecaptured") + '.jpg')
    io.imsave(save_path, img_gamma)