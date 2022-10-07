#!/usr/bin/python3

from math import log
from read_model_features import get_features

import os 
import argparse  

import tensorflow as tf
import numpy as np

from copy import deepcopy
from collections import Counter
from multiChargeFeatureList import MultiChargeFeatureList
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from statistics import mean
from sklearn.metrics import balanced_accuracy_score
  
def output_env_coll_list(output_fname, shortlist_features):
  txt_file = open(output_fname, 'w')
  sep = ","
  txt_file.write("FeatureID" + sep)
  txt_file.write("MinScan" + sep)
  txt_file.write("MaxScan" + sep)
  txt_file.write("MinCharge" + sep)
  txt_file.write("MaxCharge" + sep)
  txt_file.write("MonoMass" + sep)
  txt_file.write("RefinedMonoMass" + sep)
  txt_file.write("RepCharge" + sep)
  txt_file.write("RepMz" + sep)
  txt_file.write("Abundance" + sep)
  txt_file.write("MinElutionTime" + sep)
  txt_file.write("MaxElutionTime" + sep)
  txt_file.write("ApexElutionTime" + sep)
  txt_file.write("ElutionLength" + sep)
  txt_file.write("EnvCNNScore" + sep)
  txt_file.write("PercentMatchedPeaks" + sep)
  txt_file.write("IntensityCorrelation" + sep)
  txt_file.write("Top3Correlation" + sep)
  txt_file.write("EvenOddPeakRatios" + sep)
  txt_file.write("PercentConsecPeaks" + sep)
  txt_file.write("MaximaNumber" + sep)
  txt_file.write("Score" + sep)
  txt_file.write("Label" + "\n")
  for fl_idx in range(0, len(shortlist_features)):
    feature = shortlist_features[fl_idx]
    txt_file.write(str(feature.FeatureID) + sep)  #FeatureID
    txt_file.write(str(feature.MinScan) + sep) #min scan
    txt_file.write(str(feature.MaxScan) + sep)   #max scan
    txt_file.write(str(feature.MinCharge) + sep) #min charge
    txt_file.write(str(feature.MaxCharge) + sep) #max charge
    txt_file.write(str(feature.MonoMass) + sep) #mono_mass
    txt_file.write(str(feature.MonoMass) + sep) #refined mono_mass
    txt_file.write(str(0) + sep) #RepMz
    txt_file.write(str(0) + sep) #RepCharge
    txt_file.write(str(feature.Abundance) + sep) ## Abundance
    txt_file.write(str(feature.MinElutionTime) + sep) ## MinElutionTime
    txt_file.write(str(feature.MaxElutionTime) + sep) ## MaxElutionTime
    txt_file.write(str(feature.ApexElutionTime) + sep) ## ApexElutionTime
    txt_file.write(str(feature.ElutionLength) + sep) ## ElutionLength
    txt_file.write(str(feature.EnvcnnScore) + sep) ## EnvCNNScore
    txt_file.write(str(feature.PercentMatchedPeaks) + sep) ## PercentMatchedPeaks
    txt_file.write(str(feature.IntensityCorrelation) + sep) ## IntensityCorrelation
    txt_file.write(str(feature.Top3Correlation) + sep) ## Top3Correlation
    txt_file.write(str(feature.EvenOddPeakRatios) + sep) ## EvenOddPeakRatios
    txt_file.write(str(feature.PercentConsecPeaks) + sep) ## PercentConsecPeaks
    txt_file.write(str(feature.MaximaNumber) + sep) ## MaximaNumber
    txt_file.write(str(feature.Score) + sep) ## Score
    txt_file.write(str(feature.Label) + "\n") ## Label
  txt_file.close()

def sort_features(multicharge_features):
  multicharge_features.sort(key=lambda x: x.Score, reverse=True)
  
def shortlist_features(multicharge_features, n):
  if n == -1:
    return multicharge_features
  return multicharge_features[0:n]

def get_common_features_first_replicate(features, tolerance, time_tol, tool):
  featureList_all = deepcopy(features)
  
  ## Sort features in replicate 2 onwards by mass for binary search
  featureList = featureList_all[0]
  if tool == 1:
    featureList.sort(key=lambda x: (x is None, x.Score), reverse=True)
  else:
    featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
  for f_idx in range(1, len(featureList_all)): ## Sort rest of features by mass for binary search
    featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  
  ## Get common features based on the RT overlap
  common_features = []
  for feature_idx in range(0, len(featureList)):
    if feature_idx%10000 == 0:
      print("processing feature:", feature_idx)
    feature = featureList[feature_idx]
    if hasattr(feature, 'used'):
      continue
    else:
      tmp_common_features = []
      for featureList_idx_2 in range(1, len(featureList_all)):
        featureList_2 = featureList_all[featureList_idx_2]
        temp_features = _getMatchedFeaturesIndicesMultiCharge(featureList_2, feature, tolerance)
        overlapping_features = _get_overlapping_featuresMultiCharge(temp_features, feature, time_tol)
        if len(overlapping_features) > 0:
          # coverage = [f.coverage for f in overlapping_features]
          # selected_feature = overlapping_features[coverage.index(max(coverage))]
          apex_diff = [abs(feature.ApexElutionTime - f.ApexElutionTime) for f in overlapping_features]
          selected_feature = overlapping_features[apex_diff.index(min(apex_diff))]
          
          index = next((i for i, item in enumerate(featureList_2) if item is not None and item.FeatureID == selected_feature.FeatureID), -1)
          featureList_2[index].used = True
          tmp_common_features.append((featureList_idx_2, index))
      tmp_common_features.insert(0, (0, feature_idx))
      common_features.append(tmp_common_features)
  return common_features

def get_common_features_all_replicate(features, tolerance, time_tol):
  featureList_all = deepcopy(features)
  for f_idx in range(0, len(featureList_all)): ## Sort rest of features by mass for binary search
    featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  
  ## Get common features based on the RT overlap
  common_features = []
  for featureList_idx in range(0, len(featureList_all)):
    print("processing feature List:", featureList_idx)
    featureList = featureList_all[featureList_idx]
    featureList.sort(key=lambda x: (x is None, x.Score), reverse=True)
    for feature_idx in range(0, len(featureList)):
      if feature_idx%10000 == 0:
        print("processing feature:", feature_idx)
      feature = featureList[feature_idx]
      if hasattr(feature, 'used'):
        continue
      else:
        tmp_common_features = []
        for featureList_idx_2 in range(1, len(featureList_all)):
          featureList_2 = featureList_all[featureList_idx_2]
          temp_features = _getMatchedFeaturesIndicesMultiCharge(featureList_2, feature, tolerance)
          overlapping_features = _get_overlapping_featuresMultiCharge(temp_features, feature, time_tol)
          if len(overlapping_features) > 0:
            # coverage = [f.coverage for f in overlapping_features]
            # selected_feature = overlapping_features[coverage.index(max(coverage))]
            
            # scores = [f.Score for f in overlapping_features]
            # selected_feature = overlapping_features[scores.index(max(scores))]
            
            apex_diff = [abs(feature.ApexElutionTime - f.ApexElutionTime) for f in overlapping_features]
            selected_feature = overlapping_features[apex_diff.index(min(apex_diff))]
            
            
            index = next((i for i, item in enumerate(featureList_2) if item is not None and item.FeatureID == selected_feature.FeatureID), -1)
            featureList_2[index].used = True
            tmp_common_features.append((featureList_idx_2, index))
        tmp_common_features.insert(0, (featureList_idx, feature_idx))
        common_features.append(tmp_common_features)
  return common_features

def get_common_features_2rep(multicharge_features_rep_1, multicharge_features_rep_2, tolerance, time_tol):
  featureList = deepcopy(multicharge_features_rep_1)
  featureList.sort(key=lambda x: (x is None, x.Score), reverse=True)
  ## Sort features in replicate 2 by mass for binary search
  featureList_2 = deepcopy(multicharge_features_rep_2)
  featureList_2.sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  
  ## Get common features based on the RT overlap
  common_features = []
  # mass_diff = []
  counter = 0
  for feature_idx in range(0, len(featureList)):
  # for feature_idx in range(0, 18):
    if feature_idx%10000 == 0:
      print("processing feature:", feature_idx)
    feature = featureList[feature_idx]
    if hasattr(feature, 'used'):
      continue
    else:
      tmp_common_features = []
      temp_features = _getMatchedFeaturesIndicesMultiCharge(featureList_2, feature, tolerance)
      overlapping_features = _get_overlapping_featuresMultiCharge(temp_features, feature, time_tol)
      if len(overlapping_features) > 0:
        counter =  counter  + 1
        # mass_diff.append((feature.MonoMass, overlapping_features[0].MonoMass, feature.MinElutionTime, feature.MaxElutionTime, overlapping_features[0].MinElutionTime, overlapping_features[0].MaxElutionTime))
        # print ("feature 1 mass", feature.MonoMass, "feature 2 mass", overlapping_features[0].MonoMass)
        coverage = [f.coverage for f in overlapping_features]
        # overlapping_features.sort(key=lambda x: (x is None, x.coverage), reverse=True)
        # overlapping_features.sort(key=lambda x: (x is None, x.coverage, x.MonoMass), reverse=False)
        selected_feature = overlapping_features[coverage.index(max(coverage))]
        # selected_feature = overlapping_features[0]
        index = next((i for i, item in enumerate(featureList_2) if item is not None and item.FeatureID == selected_feature.FeatureID), -1)
        featureList_2[index].used = True
        tmp_common_features.append((1, index))
        # mass_diff.append((feature.MonoMass, featureList_2[index].MonoMass))
      tmp_common_features.insert(0, (0, feature_idx))
      common_features.append(tmp_common_features)
  print(counter)
  return common_features

def get_common_features_Promex(multicharge_features_rep_1, multicharge_features_rep_2, tolerance, time_tol):
  featureList = deepcopy(multicharge_features_rep_1)
  featureList.sort(key=lambda x: (x is None, x.MonoMass), reverse=True)
  ## Sort features in replicate 2 by mass for binary search
  featureList_2 = deepcopy(multicharge_features_rep_2)
  featureList_2.sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  
  ## Get common features based on the RT overlap
  common_features = []
  # mass_diff = []
  counter = 0
  for feature_idx in range(0, len(featureList)):
  # for feature_idx in range(0, 18):
    if feature_idx%10000 == 0:
      print("processing feature:", feature_idx)
    feature = featureList[feature_idx]
    if hasattr(feature, 'used'):
      continue
    else:
      tmp_common_features = []
      temp_features = _getMatchedFeaturesIndicesMultiCharge(featureList_2, feature, tolerance)
      overlapping_features = _get_overlapping_featuresMultiCharge(temp_features, feature, time_tol)
      if len(overlapping_features) > 0:
        counter =  counter  + 1
        # mass_diff.append((feature.MonoMass, overlapping_features[0].MonoMass, feature.MinElutionTime, feature.MaxElutionTime, overlapping_features[0].MinElutionTime, overlapping_features[0].MaxElutionTime))
        # print ("feature 1 mass", feature.MonoMass, "feature 2 mass", overlapping_features[0].MonoMass)
        coverage = [f.coverage for f in overlapping_features]
        # overlapping_features.sort(key=lambda x: (x is None, x.coverage), reverse=True)
        # overlapping_features.sort(key=lambda x: (x is None, x.coverage, x.MonoMass), reverse=False)
        selected_feature = overlapping_features[coverage.index(max(coverage))]
        # selected_feature = overlapping_features[0]
        index = next((i for i, item in enumerate(featureList_2) if item is not None and item.FeatureID == selected_feature.FeatureID), -1)
        featureList_2[index].used = True
        tmp_common_features.append((1, index))
        # mass_diff.append((feature.MonoMass, featureList_2[index].MonoMass))
      tmp_common_features.insert(0, (0, feature_idx))
      common_features.append(tmp_common_features)
  print(counter)
  return common_features

def _getMatchedFeaturesIndicesMultiCharge(feature_list, feature, tolerance):
  prec_mass = feature.MonoMass
  error_tole = prec_mass * tolerance
  ext_masses = _getExtendMasses(prec_mass)
  min_idx = _binary_search(feature_list, ext_masses[len(ext_masses) - 2] - (2 * error_tole))
  max_idx = _binary_search(feature_list, ext_masses[len(ext_masses) - 1] + (2 * error_tole))
  feature_masses = [feature_list[f_idx] for f_idx in range(min_idx, max_idx) if not hasattr(feature_list[f_idx] , 'used')]
  matched_features_indices = []
  for temp_feature in feature_masses:
    for k in range(0, len(ext_masses)):
      mass_diff = abs(ext_masses[k] - temp_feature.MonoMass)
      if (mass_diff <= error_tole):
        matched_features_indices.append(temp_feature)
        break
  return matched_features_indices

def _get_overlapping_featuresMultiCharge(temp_features, feature, time_tol):
  overlapping = []
  # print(feature.MinElutionTime, feature.MaxElutionTime)
  for feature_idx in range(0, len(temp_features)):
    f = temp_features[feature_idx]
    start_rt, end_rt = _get_overlap(f, feature, time_tol)
    overlapping_rt_range = end_rt - start_rt
    # print(f.MinElutionTime, f.MaxElutionTime, '---', start_rt, end_rt, overlapping_rt_range)
    if overlapping_rt_range > 0:
      min_rt = min(f.MinElutionTime - time_tol, feature.MinElutionTime)
      max_rt = max(f.MaxElutionTime + time_tol, feature.MaxElutionTime)
      overall_rt_range = max_rt - min_rt
      f.coverage = overlapping_rt_range/overall_rt_range
      overlapping.append(f)
  return overlapping

def _getExtendMasses(mass):
  IM = 1.00235
  extend_offsets_ = [0, -IM, IM, 2 * -IM, 2 * IM]  
  result = []
  for i in range(0, len(extend_offsets_)):
    new_mass = mass + extend_offsets_[i]
    result.append(new_mass)
  return result

def _binary_search(feature_list, mass):
  low = 0
  mid = 0
  high = len(feature_list) - 1
  while low <= high:
    mid = (high + low) // 2
    if feature_list[mid].MonoMass < mass:
      low = mid + 1
    elif feature_list[mid].MonoMass > mass:
      high = mid - 1
  return low

def _get_overlap(f, feature, time_tol):
 start_rt = max(feature.MinElutionTime, f.MinElutionTime - time_tol)
 end_rt = min(feature.MaxElutionTime, f.MaxElutionTime + time_tol)
 return (start_rt, end_rt)

def label_features(common, featureList_all, tool):
  for j in range(0, len(featureList_all)):
    if tool == 1:
      featureList_all[j].sort(key=lambda x: (x is None, x.Score), reverse=True)
    else:
      featureList_all[j].sort(key=lambda x: (x is None, x.Abundance), reverse=True)
    for f_idx in range(j + 1, len(featureList_all)):
      featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
    for elem in common:
      if elem[0][0] == j:
        for i in elem:
          featureList_all[i[0]][i[1]].Label = len(elem)

def write_Combined_feature_detailed_10(common_features_all, featureList_all, filename):
  featureList_all[0].sort(key=lambda x: x.Score, reverse=True)
  for f_idx in range(1, len(featureList_all)):
    featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  
  file = open(filename, 'w+')
  for feature_idx in common_features_all:
    if len(feature_idx) == len(featureList_all):
      file.write("FEATURE_Begin \n")
      FeatureID = [featureList_all[idx[0]][idx[1]].FeatureID for idx in feature_idx]
      file.write("FEATURE_Idx: " + str(list(FeatureID)) + "\n")
      Replicate = [idx[0]+1 for idx in feature_idx]
      file.write("Replicate: " + str(list(Replicate)) + "\n")
      MinScan = [featureList_all[idx[0]][idx[1]].MinScan for idx in feature_idx]
      file.write("Start_Scan: " + str(list(MinScan)) + "\n")
      MaxScan = [featureList_all[idx[0]][idx[1]].MaxScan for idx in feature_idx]
      file.write("End_Scan: " + str(list(MaxScan)) + "\n")
      MinElutionTime = [featureList_all[idx[0]][idx[1]].MinElutionTime for idx in feature_idx]
      file.write("Start_Retention_Time: " + str(list(MinElutionTime)) + "\n")
      MaxElutionTime = [featureList_all[idx[0]][idx[1]].MaxElutionTime for idx in feature_idx]
      file.write("End_Retention_Time: " + str(list(MaxElutionTime)) + "\n")
      ElutionLength = [featureList_all[idx[0]][idx[1]].ElutionLength for idx in feature_idx]
      file.write("Total_Retention_Time: " + str(list(ElutionLength)) + "\n")
      MinCharge = [featureList_all[idx[0]][idx[1]].MinCharge for idx in feature_idx]
      file.write("Minimum_Charge: " + str(list(MinCharge)) + "\n")
      MaxCharge = [featureList_all[idx[0]][idx[1]].MaxCharge for idx in feature_idx]
      file.write("Maximum_Charge: " + str(list(MaxCharge)) + "\n")
      MonoMass = [featureList_all[idx[0]][idx[1]].MonoMass for idx in feature_idx]
      file.write("Monoisotopic_Mass: " + str(list(MonoMass)) + "\n")
      Abundance = [featureList_all[idx[0]][idx[1]].Abundance for idx in feature_idx]
      file.write("Abundance: " + str(list(Abundance)) + "\n")
      RepScore = [featureList_all[idx[0]][idx[1]].Score for idx in feature_idx]
      file.write("Score: " + str(list(RepScore)) + "\n")
      file.write("FEATURE_END \n\n")
  file.close()

def plot_distribution_combined(common_features, Label = False):
  labels = [len(i) for i in common_features]
  labels_counter_1 = sorted(Counter(labels).items(), reverse = True)
  print("Label Counter", labels_counter_1)
  
  width = 0.5
  plt.figure()
  plt.bar(Counter(labels).keys(), Counter(labels).values(), width)
  
  labels_counter = Counter(labels)
  for i in range(1, len(Counter(labels).keys()) + 1):
    plt.text(x=i, y=labels_counter[i]+50, s=str(labels_counter[i]), rotation='75', fontdict=dict(color='blue', size=10))
    
  plt.title("Labels")
  plt.xlabel("Feature found in # of replicates")
  plt.show()
  plt.savefig("distribution.png", dpi=500)
  plt.close()

###############
def return_list(multicharge_features_replicate):
  keys = list(multicharge_features_replicate.keys())
  keys.sort()
  featureList_all = []
  for i in keys:
    featureList_all.append(deepcopy(multicharge_features_replicate[i]))
  ## Remove the used keyword
  for fl in featureList_all:
    for f in fl:
      if hasattr(f, "used"):
        delattr(f, 'used')
  return featureList_all

def remove_used(featureList):
  for fl in featureList:
    for f in fl:
      if hasattr(f, "used"):
        delattr(f, 'used')
        
def filter_features(promex_features_replicate, multicharge_features_replicate):
  shortlist_features = []
  feature_keys = sorted(list(promex_features_replicate.keys()))
  for i in feature_keys:
      p = promex_features_replicate[i]
      shortlist_features.append(multicharge_features_replicate[i][0:len(p)])
  remove_used(shortlist_features)
  return shortlist_features

def generate_combine_roc_plots(features, rep_idx = 0, replicate_count_tolerance = 7):
  featureList = features[rep_idx]
  featureList.sort(key=lambda x: x.EnvcnnScore, reverse=True)
  envcnn_scores = []
  labels = []
  for feature in featureList:
    if feature.Label >= replicate_count_tolerance:
      labels.append(1)
    elif feature.Label == 1:
      labels.append(0)
    else:
      continue
    envcnn_scores.append(feature.EnvcnnScore)
  envcnn_auc = roc_auc_score(labels, envcnn_scores)  
  envcnn_fpr, envcnn_tpr, envcnn_thresholds = roc_curve(labels, envcnn_scores)
  envcnn_predictions = [round(i) for i in envcnn_scores]
  envcnn_balanced_accuracy = balanced_accuracy_score(labels, envcnn_predictions, adjusted=False)
  print('EnvCNN AUC: %.4f' % envcnn_auc)
  print('EnvCNN Accuracy: %.4f' % envcnn_balanced_accuracy)
    
  featureList = features[rep_idx]
  featureList.sort(key=lambda x: x.Score, reverse=True)
  scores = []
  labels = []
  for feature in featureList:
    if feature.Label >= replicate_count_tolerance:
      labels.append(1)
    elif feature.Label == 1:
      labels.append(0)
    else:
      continue
    scores.append(feature.Score)
  auc = roc_auc_score(labels, scores)
  fpr, tpr, thresholds = roc_curve(labels, scores)
  predictions = [round(i) for i in scores]
  balanced_accuracy = balanced_accuracy_score(labels, predictions, adjusted=False)
  print('NN AUC: %.4f' % auc)
  print('NN Accuracy: %.4f' % balanced_accuracy)
  
  plt.figure()
  plt.plot(fpr, tpr, marker='.')
  plt.plot(envcnn_fpr, envcnn_tpr, marker='.')
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.ylabel('True positive rate')
  plt.xlabel('False positive rate')
  plt.legend(['ECScore', 'EnvCNN'], loc='lower right')
  plt.title('ROC Curve')
  plt.savefig("SW_620_ROC.png", dpi=1000)
  plt.show()
  plt.close()
  
def generate_rank_plot(features, replicate_count_tolerance = 7):
  rank_len = max([len(f) for f in features])
  positive = [0] * rank_len
  negative = [0] * rank_len
  for idx in range(0, len(features)):
    for j in range(0, len(features[idx])):
      if j >= rank_len:
        break
      if features[idx][j].Label >= replicate_count_tolerance:
        positive[j] = positive[j] + 1
      if features[idx][j].Label == 1:
        negative[j] = negative[j] + 1
  plt.figure()
  plt.plot(list(range(0, rank_len)), positive)
  plt.plot(list(range(0, rank_len)), negative)
  plt.title('Rank Plot')
  plt.ylabel('Number of PrSMs with label 1')
  plt.xlabel('Rank Number')
  plt.legend(['Positive Features', 'Negative Features'], loc='upper right')
  # plt.savefig("ROC_plot_OC.png", dpi=500)
  plt.show()
  plt.close()
  
def generate_mean_rank_plot(features, replicate_count_tolerance = 7, bin_size = 50):
  rank_len = max([len(f) for f in features])

  for idx in range(0, len(features)):
    features[idx].sort(key=lambda x: x.EnvcnnScore, reverse=True)
  EnvCNN_positive = [0] * rank_len
  for idx in range(0, len(features)):
    for j in range(0, len(features[idx])):
      if j >= rank_len:
        break
      if features[idx][j].Label >= replicate_count_tolerance:
        EnvCNN_positive[j] = EnvCNN_positive[j] + 1

  for idx in range(0, len(features)):
    features[idx].sort(key=lambda x: x.Score, reverse=True)          
  LR_positive = [0] * rank_len
  for idx in range(0, len(features)):
    for j in range(0, len(features[idx])):
      if j >= rank_len:
        break
      if features[idx][j].Label >= replicate_count_tolerance:
        LR_positive[j] = LR_positive[j] + 1

  data_1 = []
  data_2 = []
  x = []
  for i in range(0, int(rank_len/bin_size)+1):
    start = bin_size*i
    end = bin_size*(i+1)
    x.append(end)
    if end > rank_len:
      end = rank_len
    data_1.append(mean(EnvCNN_positive[start:end]))
    data_2.append(mean(LR_positive[start:end]))
  
  plt.figure()
  plt.plot(x, data_2)
  plt.plot(x, data_1)
  plt.title('Rank Plot')
  plt.ylabel('Found in # of Replicates')
  plt.xlabel('Rank Number')
  plt.legend(['ECScore', 'EnvCNN'], loc='upper right')
  plt.savefig("SW_620_Rank.png", dpi=1000)
  plt.show()
  plt.close()
  
def compute_rankSum(features, replicate_count_tolerance = 7):
  for idx in range(0, len(features)):
    features[idx].sort(key=lambda x: x.EnvcnnScore, reverse=True)
  ranksum = []
  for i in range(len(features)):
    env_ranksum = 0
    for j in range(len(features[i])):
      if features[i][j].Label >= replicate_count_tolerance:
        env_ranksum = env_ranksum + (j + 1)
    ranksum.append(env_ranksum)
  print("EnvCNN RankSUM value:", sum(ranksum))
  
  for idx in range(0, len(features)):
    features[idx].sort(key=lambda x: x.Score, reverse=True)
  ranksum = []
  for i in range(len(features)):
    env_ranksum = 0
    for j in range(len(features[i])):
      if features[i][j].Label >= replicate_count_tolerance:
        env_ranksum = env_ranksum + (j + 1)
    ranksum.append(env_ranksum)
  print("NN Model RankSUM value:", sum(ranksum))
  
def score_model_NNNN(multicharge_features):
  model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\00_train_data\E_11_6_9_14_7_2_3.h5")
  data = []
  for i in multicharge_features:
    in_feature = [i.EnvcnnScore, i.rt_range/60.0, i.PercentMatchedPeaks, log(i.Abundance), i.RepCharge, i.Top3Correlation, i.charge_range/30.0, i.EvenOddPeakRatios]
    data.append(in_feature)
  test_data = np.array(data)
  scores = model.predict(test_data)
  for i in range(0, len(multicharge_features)):
    multicharge_features[i].Score = scores[i][0] 
    
if __name__ == "__main__":
  print("In main function!!")
  parser = argparse.ArgumentParser(description='Generate data matrix for the mzML data.')
  parser.add_argument("-F", "--feature_dir", default = "", help="Scored feature CSV file (Replicate 1)")
  parser.add_argument("-M", "--model_file", default = "", help="Scored feature CSV file (Replicate 2)")
  parser.add_argument("-e", "--tolerance", default = 10E-6, help="Mass tolerance (ppm)", type = float)
  parser.add_argument("-t", "--timeTolerance", default = 1.0, help="Time tolerance (minutes)", type = float)
  parser.add_argument("-n", "--NumFeatures", default = -1, help="Used to keep top N features", type = int)
  args = parser.parse_args()
  
  args.feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\OC"
  # args.feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\SW_620"
  data_type = args.feature_dir.split('Sep_19_v3\\')[1]
  
  if data_type == "OC":
    replicate_count_tolerance = 8
  else:
    replicate_count_tolerance = 3
  cutoff = 0.5
  
  print("\nLoad Features")  
  feature_files = [f for f in os.listdir(args.feature_dir) if f.endswith(r".feature")]# or f.endswith(r".csv")]
  multicharge_features_replicate = {}
  for file in feature_files:
    if data_type == "OC":  
      file_idx = int(file.split('rep')[1].split('_')[0])
    else:
      file_idx = int(file.split('_0')[1].split('_')[0])
    print("Processing File:", file_idx, file)
    multicharge_features = get_features(os.path.join(args.feature_dir, file))
    score_model_NNNN(multicharge_features)
    multicharge_features = [i for i in multicharge_features if i.Score >=cutoff]
    sort_features(multicharge_features)
    multicharge_features_replicate[file_idx] = multicharge_features
  
  
  ######## GET AUC
  features = return_list(multicharge_features_replicate)
  common_features_r1 = get_common_features_first_replicate(features, args.tolerance, args.timeTolerance, 0)
  plot_distribution_combined(common_features_r1, True)
  label_features(common_features_r1, features, 0)
  
  generate_combine_roc_plots(features, 0, replicate_count_tolerance)
  generate_mean_rank_plot(features, replicate_count_tolerance, bin_size = 100)
  compute_rankSum(features, replicate_count_tolerance)
  
  positive_features = [f.Score for f in features[0] if f.Label >= replicate_count_tolerance]
  negative_features = [f.Score for f in features[0] if f.Label == 1]
  plt.Figure()
  plt.hist(positive_features, bins=10, range=(0, 1), alpha= 0.6)
  plt.hist(negative_features, bins=10, range=(0, 1), alpha= 0.6)
  plt.legend(['+ve', '-ve'])
  plt.title("Score distributio")
  plt.show()
  plt.close()
  
  print("Number of +ve features:", len(positive_features))
  print("Number of -ve features:", len(negative_features))
  
