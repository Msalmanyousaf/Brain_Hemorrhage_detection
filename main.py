# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16

from trained_models import resnet_50, vgg_16
from preprocessor import Preprocessor
from dataloader import DataLoader
from modelling import MyNetwork
from losses import np_multilabel_loss, multilabel_focal_loss

def turn_pred_to_dataframe(data_df, pred):
    df = pd.DataFrame(pred, columns=data_df.columns, index=data_df.index)
    df = df.stack().reset_index()
    df.loc[:, "ID"] = df.ID.str.cat(df.Subtype, sep="_")
    df = df.drop(["ID", "Subtype"], axis=1)
    df = df.rename({0: "Label"}, axis=1)
    return df


# define paths
INPUT_PATH = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"
train_dir = INPUT_PATH + "stage_2_train/"
test_dir = INPUT_PATH + "stage_2_test/"   
pretrained_models_path = "../input/keras-pretrained-models/"    
MODELOUTPUT_PATH = "best_model.hdf5"

# read csv files
testdf = pd.read_csv(INPUT_PATH + 'stage_2_sample_submission.csv')
traindf = pd.read_csv(INPUT_PATH + 'stage_2_train.csv')

#######################################################
############## Dataframes preprocessing ###############

labels = traindf.Label
traindf = traindf.ID.str.rsplit(pat = '_', n = 1, expand = True)
traindf.loc[:,'Label'] = labels.values
traindf = traindf.rename({0:'ID', 1: 'Subtype'}, axis = 1)

testdf = testdf.ID.str.rsplit(pat = '_', n = 1, expand = True)
testdf = testdf.rename({0:'ID', 1: 'Subtype'}, axis = 1)
testdf.loc[:, 'Label'] = 0

traindf = pd.pivot_table(traindf, index="ID", columns="Subtype", values="Label")
testdf = pd.pivot_table(testdf, index="ID", columns="Subtype", values="Label")

#######################################################
################## validation #########################

split_seed = 1
kfold = StratifiedKFold(n_splits = 5, random_state = split_seed).split(
    np.arange(traindf.shape[0]), traindf["any"].values)

train_idx, dev_idx = next(kfold)

train_data = traindf.iloc[train_idx]
dev_data = traindf.iloc[dev_idx]

       
pretrained_models = {
    "resnet_50": {"weights": "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                  "nn_input_shape": (224,224),
                  "preprocess_fun": preprocess_resnet_50},
    "vgg16": {"weights": "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
              "nn_input_shape": (224,224),
              "preprocess_fun": preprocess_vgg_16}
}        


#######################################################
#################### Run the model ####################

BACKBONE = "resnet_50"
BATCH_SIZE = 16
TEST_BATCH_SIZE = 5
EPOCHS = 20
lr = 0.0001
transfer_model = resnet_50(pretrained_models_path, pretrained_models)

train_preprocessor = Preprocessor(path = train_dir,
                                  backbone = pretrained_models[BACKBONE],
                                  augment = True)

dev_preprocessor = Preprocessor(path = train_dir,
                                backbone = pretrained_models[BACKBONE],
                                augment = False)

test_preprocessor = Preprocessor(path = test_dir,
                                backbone = pretrained_models[BACKBONE],
                                augment = False)


train_dataloader = DataLoader(train_data, train_preprocessor, BATCH_SIZE,
                              shuffle = True)

dev_dataloader = DataLoader(dev_data, dev_preprocessor, BATCH_SIZE,
                            shuffle = True)

test_dataloader = DataLoader(testdf, test_preprocessor, TEST_BATCH_SIZE,
                             shuffle = False)


model = MyNetwork(base_model = transfer_model,
                  loss_fun = "binary_crossentropy", #multilabel_focal_loss(class_weights=my_class_weights, alpha=0.5, gamma=0),
                  metrics_list = [multilabel_focal_loss(alpha = 0.5, gamma = 0)],
                  train_generator = train_dataloader,
                  dev_generator = dev_dataloader,
                  epochs = EPOCHS,
                  checkpoint_path = MODELOUTPUT_PATH,
                  num_classes = 6)

model.build_model()
model.compile_model(lr)
history = model.learn()

print(history.history.keys())

test_pred = model.predict(test_dataloader)[0:testdf.shape[0]]
# dev_pred = model.predict(dev_dataloader)

test_pred_df = turn_pred_to_dataframe(testdf, test_pred)
# dev_pred_df = turn_pred_to_dataframe(dev_data, dev_pred)

test_pred_df.to_csv("test_pred.csv", index=False)
# dev_pred_df.to_csv("dev_pred.csv", index=False)






































