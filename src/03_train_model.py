#!/usr/bin/python3

import os
import argparse
import tensorflow as tf
import keras
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from read_model_features import get_features

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
  
def train_model(features_list, output_model_fname):
  tf.random.set_seed(42)
  X, Y = get_training_data_all(features_list)
  X_train, X_test, y_train, y_test, = train_test_split(X, Y, test_size=0.33, shuffle=True, random_state=42)
  print("Total Features:", len(Y), "Positive:", sum(Y), "Negative:", len(Y)-sum(Y))
  print("Train Features:", len(y_train), "Positive:", sum(y_train), "Negative:", len(y_train)-sum(y_train))
  print("Test Features:", len(y_test), "Positive:", sum(y_test), "Negative:", len(y_test)-sum(y_test))
  
  pos = sum(y_train)
  neg = len(y_train)-sum(y_train)
  total = len(y_train)
  weight_for_0 = (1 / neg) * (total / 2.0)
  weight_for_1 = (1 / pos) * (total / 2.0)
  class_weight  = {1:weight_for_1, 0:weight_for_0}
  
  feature_list = {0:"EnvcnnScore", 1:"rt_range", 2:"charge_range", 3:"EvenOddPeakRatios", 4:"IntensityCorrelation", 5:"PercentConsecPeaks",  
                  6:"PercentMatchedPeaks", 7:"Top3Correlation", 8:"MaximaNumber", 9:"Abundance", 10:"NumTheoPeaks", 
                  11:"IntensityCosineSimilarity", 12:"MzErrorSum", 13:"MzErrorSumBase", 14:"RepCharge"}

  train_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3]) for xx in X_train])
  test_data = np.array([(xx[0], xx[1]/60.0, xx[6], math.log(xx[9]), xx[14], xx[7], xx[2]/30.0, xx[3]) for xx in X_test])
 
  train_labels = np.array(y_train)
  test_labels = np.array(y_test)

  neurons = 200  
  early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1, mode='min')
  checkpoint = keras.callbacks.ModelCheckpoint(output_model_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #Save Model Checkpoint
  model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(train_data.shape[1],)),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(neurons, kernel_regularizer=tf.keras.regularizers.L1(1e-6)),
  tf.keras.layers.LeakyReLU(alpha=0.05),
  tf.keras.layers.Dense(1, activation='sigmoid')])
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1E-5), metrics=['accuracy', 'AUC']) , #validation_split=0.33)
  history = model.fit(train_data, train_labels, verbose=1, epochs=1500, class_weight=class_weight, callbacks=[checkpoint, early_stopping], validation_data=(test_data, test_labels), batch_size=32)
  # plot_loss(history)
  # plot_acc(history)
  # plot_auc(history)
  
  real_scores = model.predict(test_data).ravel()
  predictions = [round(i) for i in real_scores]
  balanced_accuracy = balanced_accuracy_score(y_test, predictions, adjusted=False)
  auc = roc_auc_score(y_test, real_scores)
  print("Accuracy:", round(balanced_accuracy*100, 2), "and AUC:", round(auc*100, 2))
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train ECScore Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-f", "--featureFile", default = r"E:\TopFD_Published_Data\04_Training_Data_ECScore_Model\00_train_data.csv", help="Labeled feature file")
  parser.add_argument("-m", "--outputFile", default = "model.h5", help="model name")
  args = parser.parse_args()

  features_list = get_features(args.featureFile)
  train_model(features_list, args.outputFile)
