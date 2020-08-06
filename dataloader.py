# -*- coding: utf-8 -*-

import numpy as np 
from keras.utils import Sequence

class DataLoader(Sequence):
    """
    custom data loaded classderived from Sequence class to take into account the
    custom preprocessing.
    """
    
    def __init__(self, dataframe, preprocessor, batch_size, shuffle,
                 num_classes = 6):
        """        

        Parameters
        ----------
        dataframe : pandas dataframe
        preprocessor : object of Preprocessor class
        batch_size : int
        shuffle : bool
        num_classes : int, optional
            The default is 6.

        Returns
        -------
        None.

        """
        
        self.preprocessor = preprocessor
        self.data_ids = dataframe.index.values
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = self.preprocessor.backbone["nn_input_shape"]
        self.preprocess_fun = self.preprocessor.backbone["preprocess_fun"]
        self.num_classes = num_classes
        self.current_epoch = 0

    def __len__(self):
        """
        defines the number of steps per epoch
        """

        return np.int(np.ceil(len(self.data_ids) / np.float(self.batch_size)))  

    def on_epoch_end(self):
        """
        at the end of an epoch
        """
        
        self.data_ids = self.dataframe.index.values
        if self.shuffle:
            np.random.shuffle(self.data_ids)
        self.current_epoch += 1
    
    def __getitem__(self, item):
        """
        returns a batch of images
        """
        # select the ids of the current batch
        current_ids = self.data_ids[item*self.batch_size:(item+1)*self.batch_size]
        X, y = self.__generate_batch(current_ids)
        return X, y
    
    def __generate_batch(self, current_ids):
        """
        collects the preprocessed images and targets of one batch
        """
        X = np.empty((self.batch_size, *self.input_shape, 3))
        y = np.empty((self.batch_size, self.num_classes))
        for idx, ident in enumerate(current_ids):
            # Store sample
            image = self.preprocessor.preprocess(ident)
            X[idx] = image
            # Store class
            y[idx] = self.__get_target(ident)
        return X, y
    
    def __get_target(self, ident):
        """
        extracts the targets of one image id
        """
        targets = self.dataframe.loc[ident].values
        return targets    
