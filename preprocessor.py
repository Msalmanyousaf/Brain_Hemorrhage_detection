# -*- coding: utf-8 -*-

import numpy as np 
import pydicom
from skimage.transform import resize
from imgaug import augmenters as iaa

class Preprocessor:    
    
    def __init__(self, path, backbone, augment = False):
        """
        This class allows to read DICOM files of CT scans, apply three windows
        (brain, subdural, soft), consider each window as separate channel in
        RGB image, resize the image according to the input of the selected model
        for transfer learning.

        Parameters
        ----------
        path : string
            path of the directory containing the DICOM files of CT scans.
        backbone : Python dictionary
            contains weigths, input shape and preprocess function of the selected 
            model for transfer learning. Example:
            backbone = {"weights": "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                  "nn_input_shape": (224,224),
                  "preprocess_fun": preprocess_resnet_50},
        augment : bool, optional
            Option to apply image augmentation. The default is False.

        Returns
        -------
        None.

        """
        self.path = path
        self.backbone = backbone
        self.nn_input_shape = backbone["nn_input_shape"]
        self.augment = augment
        
    def load_dicom_dataset(self, filename):
        """
        reads the DICOM file

        Parameters
        ----------
        filename : string
            name of DICOM file with extension

        Returns
        -------
        dataset : dcm data set

        """
        dataset = pydicom.dcmread(self.path + filename)
        return dataset        

    def rescale_pixelarray(self, dataset):
        """
        Applies linear transformation to get Hounsfield Unit (HU) values.

        Parameters
        ----------
        dataset : dcm data set

        Returns
        -------
        rescaled_image : numpy array
            pixel values after applying linear transformation

        """
        try:
            image = dataset.pixel_array
        except:
            image = np.ones((self.nn_input_shape[0], self.nn_input_shape[1]))            
        rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
        return rescaled_image

    def window_image(self, dataset, window_center, window_width, normalize = True):
        """
        Applies a window of user specified width and center to DICOM image. 
        
        Parameters
        ----------
        dataset : DICOM data set
        window_center : int
        window_width : int
        normalize : bool, optional
            Normalizes the pixel values between 0 and 1. Default is True.
    
        Returns
        -------
        windowed_image : numpy array
            pixels after applying the specified window. This can now be plotted 
            for visualization.
        
        """
          
        # apply linear transformation
        windowed_image = self.rescale_pixelarray(dataset)
        
        img_min = window_center - window_width//2
        img_max = window_center + window_width//2
        
        # apply the specified window
        windowed_image[windowed_image < img_min] = img_min
        windowed_image[windowed_image > img_max] = img_max
        
        if normalize:
            # normalizing to 0-1
            windowed_image = (windowed_image - img_min) / (img_max - img_min)
        
        return windowed_image

    def bsb_window(self, dataset):
        """
        Applies brain, subdural and soft windows to the image. All three windowed
        images are stacked together to make RGB image.

        Parameters
        ----------
        dataset : dcm data set

        Returns
        -------
        bsb_img : numpy array
            numpy array with pixel values corresponding to brain, subdural and 
            soft windowed images.

        """
        
        brain_img = self.window_image(dataset, 40, 80)
        subdural_img = self.window_image(dataset, 80, 200)
        soft_img = self.window_image(dataset, 40, 380)
        
        bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)
            
        return bsb_img
      
    def resize(self, image):
        """
        resizes the image dimensions according to the input size of the selected
        model for transfer learning.

        Parameters
        ----------
        image : numoy array

        Returns
        -------
        image : numpy array
            resized image.

        """
        
        # image is pixel array
        out_shape = (self.nn_input_shape[0], self.nn_input_shape[1], 3)
        image = resize(image, output_shape = out_shape, anti_aliasing = True)
        return image
    
    def augment_img(self, image): 
        """
        apply augmentation to image to avoid overfitting.

        Parameters
        ----------
        image : numpy array

        Returns
        -------
        image_aug : numpy array

        """
        augment_img = iaa.Sequential([
            #iaa.Crop(keep_size=True, percent=(0.01, 0.05), sample_independently=False),
            #iaa.Affine(rotate=(-10, 10)),
            iaa.Fliplr(0.5)])
        image_aug = augment_img.augment_image(image)
        return image_aug            
        
    def preprocess(self, identifier):
        """
        applies the preprocessing steps defined in the class Preprocessor

        Parameters
        ----------
        identifier : string
            DICOM file name without extension.

        Returns
        -------
        image : numpy array
            preprocessed image

        """
        filename = identifier +  ".dcm"
        dataset = self.load_dicom_dataset(filename)
        windowed_image = self.bsb_window(dataset)
        image = self.resize(windowed_image)
        if self.augment:
            image = self.augment_img(image)
            
        return image       