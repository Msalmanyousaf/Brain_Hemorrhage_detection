# -*- coding: utf-8 -*-

from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

class MyNetwork:
    
    def __init__(self, base_model, loss_fun, metrics_list, train_generator,
                 dev_generator, epochs, 
                 checkpoint_path,
                 num_classes = 6):
        """      

        Parameters
        ----------
        base_model : selected model object for transfer learning
        loss_fun : loss function 
        metrics_list : list containing metrics
        train_generator : object of class Dataloader
        dev_generator : object of class Dataloader
        epochs : int
        checkpoint_path : string
            path to save the best model after every epoch
        num_classes : int, optional
            The default is 6.

        Returns
        -------
        None.

        """
        
        self.base_model = base_model
        self.loss_fun = loss_fun
        self.metrics_list = metrics_list
        self.train_generator = train_generator
        self.dev_generator = dev_generator
        self.epochs = epochs
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path 
              
        # save the best known model after every epoch. this is helpful if you
        # want to continue training at some later stage.
        
        self.checkpoint = ModelCheckpoint(filepath = self.checkpoint_path,
                                          mode = "min", verbose = 1,
                                          save_best_only = True, 
                                          save_weights_only = True,
                                          period = 1,
                                          monitor='val_loss')
        
        self.reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                           factor = 0.5,
                                           patience = 2,
                                           min_lr = 1e-8,
                                           mode = "min")
        
        # if the loss is no longer going down, terminate training.
        self.e_stopping = EarlyStopping(monitor = "val_loss",
                                        min_delta = 0.01,
                                        patience = 5,
                                        mode = "min",
                                        restore_best_weights = True)
        
    def build_model(self):

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.3)(x)
        pred = Dense(self.num_classes,
                     kernel_initializer=he_normal(seed=11),
                     kernel_regularizer=l2(0.05),
                     bias_regularizer=l2(0.05), activation="sigmoid")(x)
        self.model = Model(inputs=self.base_model.input, outputs=pred)
    
    def compile_model(self, lr):
        """
        
        Parameters
        ----------
        lr : float
            learning rate.

        Returns
        -------
        None.

        """
        self.model.compile(optimizer=Adam(learning_rate=lr),
                           loss=self.loss_fun, 
                           metrics=self.metrics_list)
    
    def learn(self):
        return self.model.fit_generator(generator=self.train_generator,
                    validation_data=self.dev_generator,
                    epochs=self.epochs,
                    callbacks=[self.checkpoint, self.reduce_lr, self.e_stopping],
                    #use_multiprocessing=False,
                    workers=8)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def predict(self, test_generator):
        predictions = self.model.predict_generator(test_generator, workers=8)
        return predictions    