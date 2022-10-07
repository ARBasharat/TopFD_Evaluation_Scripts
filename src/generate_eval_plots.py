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

import numpy as np
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def generate_roc_plots(labels, scores):
  auc = roc_auc_score(labels, scores)
  fpr, tpr, thresholds = roc_curve(labels, scores)
  # print('AUC: %.4f' % auc)
  plt.figure()
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr, marker='.')
  plt.ylabel('True positive rate')
  plt.xlabel('False positive rate')
  plt.legend(['Reference', 'EnvCNN', 'MS-Deconv'], loc='lower right')
  plt.title('ROC Curve')
  plt.show()
  return auc
  
  
def generate_rank_plot(multicharge_features_replicate, rank_len = 6000):
  positive = [0] * rank_len
  negative = [0] * rank_len
  for idx in range(0, len(multicharge_features_replicate)):
    for j in range(0, len(multicharge_features_replicate[idx + 1].featureList)):
      if j >= rank_len:
        break
      if multicharge_features_replicate[idx + 1].featureList[j].Label == 1:
        positive[j] = positive[j] + 1
      if multicharge_features_replicate[idx + 1].featureList[j].Label == 0:
        negative[j] = negative[j] + 1
  plt.figure()
  plt.plot(list(range(0, rank_len)), positive)
  plt.plot(list(range(0, rank_len)), negative)
  plt.title('Rank Plot')
  plt.ylabel('Number of PrSMs with label 1')
  plt.xlabel('Rank Number')
  plt.legend(['Positive Features', 'Negative Features'], loc='upper right')
  plt.show()
  plt.close()
