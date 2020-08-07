# RSNA Intracranial Hemorrhage Detection
This is a [Kaggle project](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview), where the goal is to detect brain hemorrhage and classify into its subtypes by looking at the brain CT scan. Intracranial hemorrhage is the bleeding that occurs inside the cranium. This is a serious medical issue and requires immediate treatment.  It is usually classified into the following 5 types:
1.  Intraparenchymal
2. Intraventricular
3. Sabarachnoid
4. Subdural
5. Epidural    

The details can be found in the project description on Kaggle.

## 1. Training Data    
The [training data](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) consists of DICOM files of brain CT scans. Moreover, a csv file is also provided, where the probability of each hemorrhage type is recorded in front of the image ID. The task is to train a model on DICOM files and then make predictions for the given test data. Finally, a csv file for the test data predictions needs to be uploaded on Kaggle.  
## 2. Digital Imaging and Communications in Medicine (DICOM)
DICOM is a widely used file format in the medical field to record and transfer the data (usually medical scans). The CT scans are recorded in the form of Hounsfield Unit (HU) values. These values can vary between -1000 and +1000. The HU value of a tissue depends on its density. For example, bones are represented by HU values of +1000, while air is represented by HU value of -1000. 
### 2.1. Concept of Window
The most important task in the analysis of CT scans is the choice of appropriate window. The range of -1000 to +1000 contains so many grey scale values that they cannot be distinguished by the human eye. Therefore, the doctors apply windows, where the HU values outside the window range are clipped. In this project, three windows (brain, soft, subdural) are applied. The image arrays obtained after applying windows are stacked together to make an RGB image.

The concept of window along with visualization is explained in the file "Exploratory Data Analysis_EDA.ipynb"
## 2. Transfer Learning
In this project, pretrained ResNet50 and VGG16 models are used by adding the new final layers.  Since the CT scans require some preprocessing, a custom preprocessor along with data loader are used. 
## 3. Acknowledgements
The purpose of this project was to get a hands-on experience with transfer learning using well-known models. Moreover, the project also provided how to do preprocessing on the images especially in the medical field. The concepts applied in the code were derived after carefully going through many great Kernels shared on Kaggle. Some of these Kernels include [David Tang](https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing), [Ryan Epp](https://www.kaggle.com/reppic/gradient-sigmoid-windowing), [Marco Vasquez](https://www.kaggle.com/marcovasquez/basic-eda-data-visualization), [akensert](https://www.kaggle.com/akensert/rsna-inceptionv3-keras-tf1-14-0) and [Laura Fink](https://www.kaggle.com/allunia/rsna-ih-detection-baseline/notebook).
