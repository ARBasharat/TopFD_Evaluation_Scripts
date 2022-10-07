
#!/usr/bin/python3

##Copyright (c) 2014 - 2021, The Trustees of Indiana University.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.

import argparse
import tensorflow as tf
import keras
import numpy as np
# import os
import math
from sklearn.model_selection import train_test_split
from read_model_features import get_features
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt
import generate_eval_plots as eval_plots


def get_training_data_all(features_list):
  X = []
  Y = []
  for i in range(0, len(features_list)):
    if features_list[i].Label == 1:
      X.append([features_list[i].EnvcnnScore, features_list[i].rt_range, features_list[i].charge_range, features_list[i].EvenOddPeakRatios, features_list[i].IntensityCorrelation,
            features_list[i].PercentConsecPeaks, features_list[i].PercentMatchedPeaks, features_list[i].Top3Correlation, features_list[i].MaximaNumber, features_list[i].Abundance,
            features_list[i].NumTheoPeaks, features_list[i].IntensityCosineSimilarity, features_list[i].MzErrorSum, features_list[i].MzErrorSumBase, features_list[i].RepCharge])
      Y.append(1)
    if features_list[i].Label == 0:
      X.append([features_list[i].EnvcnnScore, features_list[i].rt_range, features_list[i].charge_range, features_list[i].EvenOddPeakRatios, features_list[i].IntensityCorrelation,
            features_list[i].PercentConsecPeaks, features_list[i].PercentMatchedPeaks, features_list[i].Top3Correlation, features_list[i].MaximaNumber, features_list[i].Abundance,
            features_list[i].NumTheoPeaks, features_list[i].IntensityCosineSimilarity, features_list[i].MzErrorSum, features_list[i].MzErrorSumBase, features_list[i].RepCharge])
      Y.append(0)
  return X, Y

  
def plot_loss(history):
  plt.Figure()
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.close()
  
def plot_acc(history):
  plt.Figure()
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label='val_accuracy')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.close()
  
def plot_auc(history):
  plt.Figure()
  plt.plot(history.history['auc'], label='auc')
  plt.plot(history.history['val_auc'], label='val_auc')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.ylabel('AUC')
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.close()
  
def test_features_NN_10(features_list, output_model_fname):
  tf.random.set_seed(42)
  X, Y = get_training_data_all(features_list)
  X_train, X_test, y_train, y_test, = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=42)
  print("Total Features:", len(Y), "Positive:", sum(Y), "Negative:", len(Y)-sum(Y))
  print("Train Features:", len(y_train), "Positive:", sum(y_train), "Negative:", len(y_train)-sum(y_train))
  print("Test Features:", len(y_test), "Positive:", sum(y_test), "Negative:", len(y_test)-sum(y_test))
  
  # test_data = np.array([xx[0] for xx in X_test])
  # envcnn_auc = roc_auc_score(y_test, test_data)
  # envcnn_fpr, envcnn_tpr, envcnn_thresholds = roc_curve(y_test, test_data)
  # envcnn_predictions = [round(i) for i in test_data]
  # envcnn_balanced_accuracy = balanced_accuracy_score(y_test, envcnn_predictions, adjusted=False)
  # print('EnvCNN AUC: %.4f' % envcnn_auc)
  # print('EnvCNN Accuracy: %.4f' % envcnn_balanced_accuracy) 
  
  pos = sum(y_train)
  neg = len(y_train)-sum(y_train)
  total = len(y_train)
  weight_for_0 = (1 / neg) * (total / 2.0)
  weight_for_1 = (1 / pos) * (total / 2.0)
  class_weight  = {1:weight_for_1, 0:weight_for_0}
  
  feature_list = {0:"EnvcnnScore", 1:"rt_range", 2:"charge_range", 3:"EvenOddPeakRatios", 4:"IntensityCorrelation", 5:"PercentConsecPeaks",  
                  6:"PercentMatchedPeaks", 7:"Top3Correlation", 8:"MaximaNumber", 9:"Abundance", 10:"NumTheoPeaks", 
                  11:"IntensityCosineSimilarity", 12:"MzErrorSum", 13:"MzErrorSumBase", 14:"RepCharge"}
  idx = [0, 6, 1, 5, 9, 2, 12, 10, 4, 7, 14]
  # return;
  
  # train_data = np.array([(xx[0], xx[6], xx[1], xx[5], math.log(xx[9]), xx[2]/30.0, xx[12]/xx[10], xx[4], xx[7], xx[14]) for xx in X_train])
  # test_data = np.array([(xx[0], xx[6], xx[1], xx[5], math.log(xx[9]), xx[2]/30.0, xx[12]/xx[10], xx[4], xx[7], xx[14]) for xx in X_test])

  # ##############################
  
  # output_model_fname = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\00_train_data\E_11_6_9_14_7_2_3_5_4.h5"
  # train_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3], xx[5], xx[4]) for xx in X_train])
  # test_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3], xx[5], xx[4]) for xx in X_test])
  
  # output_model_fname = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\00_train_data\E_11_6_9_14_7_2_3_5.h5"
  # train_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3], xx[5]) for xx in X_train])
  # test_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3], xx[5]) for xx in X_test])
  
  output_model_fname = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\00_train_data\E_11_6_9_14_7_2_3_5_12.h5"
  train_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3], xx[5], xx[12]/xx[10]) for xx in X_train])
  test_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3], xx[5], xx[12]/xx[10]) for xx in X_test])
 
  ########################################
  # output_model_fname = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\00_train_data\E_6_9_14_7_2_12_11_5_3_4.h5"
  # train_data = np.array([(xx[0], xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[12]/xx[10], xx[1]/60.0, xx[5], xx[3], xx[4]) for xx in X_train])
  # test_data = np.array([(xx[0], xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[12]/xx[10], xx[1]/60.0, xx[5], xx[3], xx[4]) for xx in X_test])
  
  train_labels = np.array(y_train)
  test_labels = np.array(y_test)

  neurons = 200  
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')
  checkpoint = keras.callbacks.ModelCheckpoint(output_model_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #Save Model Checkpoint
  model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(train_data.shape[1],)),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.06),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.06),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(1, activation='sigmoid')])
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1E-5), metrics=['accuracy', 'AUC']) , #validation_split=0.33)
  history = model.fit(train_data, train_labels, verbose=1, epochs=1500, class_weight=class_weight, callbacks=[checkpoint, early_stopping], validation_data=(test_data, test_labels), batch_size=32)
  plot_loss(history)
  plot_acc(history)
  plot_auc(history)
  
  real_scores = model.predict(test_data).ravel()
  predictions = [round(i) for i in real_scores]
  balanced_accuracy = balanced_accuracy_score(y_test, predictions, adjusted=False)
  auc = eval_plots.generate_roc_plots(y_test, real_scores)
  print("Accuracy:", round(balanced_accuracy*100, 2), "and AUC:", round(auc*100, 2))
  
def test_features_NN_direct(features_list, output_model_fname):
  ## Get data
  X, Y = get_training_data_all(features_list)  
  X_train, X_test, y_train, y_test, = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=42)
  print("Total Features:", len(Y), "Positive:", sum(Y), "Negative:", len(Y)-sum(Y))
  print("Train Features:", len(y_train), "Positive:", sum(y_train), "Negative:", len(y_train)-sum(y_train))
  print("Test Features:", len(y_test), "Positive:", sum(y_test), "Negative:", len(y_test)-sum(y_test))

  ## Get class weight
  pos = sum(y_train)
  neg = len(y_train)-sum(y_train)
  total = len(y_train)
  weight_for_0 = (1 / neg) * (total / 2.0)
  weight_for_1 = (1 / pos) * (total / 2.0)
  class_weight  = {1:weight_for_1, 0:weight_for_0}
  print("Test Features:", class_weight )
  
  ## Prepare data
  tf.random.set_seed(42)

  train_data = np.array([(xx[0], xx[1], xx[2]/30.0, xx[4], xx[5], xx[6], xx[7], math.log10(xx[9])) for xx in X_train])
  test_data = np.array([(xx[0], xx[1], xx[2]/30.0, xx[4], xx[5], xx[6], xx[7], math.log10(xx[9])) for xx in X_test])
  
  train_labels = np.array(y_train)
  test_labels = np.array(y_test)

  ## define and train model
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')
  checkpoint = tf.keras.callbacks.ModelCheckpoint(output_model_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #Save Model Checkpoint
  model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(8,)),
  tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(200, kernel_regularizer=tf.keras.regularizers.l1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(1, activation='sigmoid')])
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1E-5), metrics=['accuracy', 'AUC'])
  history = model.fit(train_data, train_labels, verbose=1, epochs=1500, class_weight=class_weight, callbacks=[checkpoint, early_stopping], validation_data=(test_data, test_labels))
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Extract the proteoform features.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-F", "--featureFile", default = r"/data/abbash/MS_Feature_Extraction_refactored/src/train/rep_1_multiCharge_features_labeled.csv", help="csv feature file")
  parser.add_argument("-O", "--outputFile", default = "nn_model_10.h5", help="model name")
  args = parser.parse_args()
  
  # args.featureFile = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_14_v1_results\features\00_train_data\00_train_data.csv"
  # args.featureFile = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_15_v1\00_train_data\00_train_data.csv"
  args.featureFile = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\00_train_data\00_train_data.csv"
  features_list = get_features(args.featureFile)
  # test_features_NN_direct(features_list, args.outputFile)
  test_features_NN_10(features_list, args.outputFile)
