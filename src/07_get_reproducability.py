#!/usr/bin/python3

import os 
import argparse  
import statistics
import seaborn as sns
import tensorflow as tf
import numpy as np
from math import log, log10
from copy import deepcopy
from collections import Counter
from matplotlib import pyplot as plt
from read_model_features import get_features
from multiChargeFeatureList import MultiChargeFeatureList
  
def output_env_coll_list(output_fname, shortlist_features):
  txt_file = open(output_fname, 'w')
  sep = ","
  txt_file.write("FeatureID" + sep)
  txt_file.write("MinScan" + sep)
  txt_file.write("MaxScan" + sep)
  txt_file.write("MinCharge" + sep)
  txt_file.write("MaxCharge" + sep)
  txt_file.write("MonoMass" + sep)
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
  txt_file.write("IntensityCosineSimilarity" + sep)
  txt_file.write("NumTheoPeaks" + sep)
  txt_file.write("MzErrorSum" + sep)
  txt_file.write("MzErrorSumBase" + sep)  
  txt_file.write("Score" + sep)
  txt_file.write("Label" + "\n")
  for fl_idx in range(0, len(shortlist_features)):
    feature = shortlist_features[fl_idx]
    txt_file.write(str(feature.FeatureID) + sep)  #FeatureID
    txt_file.write(str(feature.MinScan) + sep) #MinScan
    txt_file.write(str(feature.MaxScan) + sep)   #MaxScan
    txt_file.write(str(feature.MinCharge) + sep) #MinCharge
    txt_file.write(str(feature.MaxCharge) + sep) #MaxCharge
    txt_file.write(str(feature.MonoMass) + sep) #MonoMass
    txt_file.write(str(feature.RepCharge) + sep) #RepCharge
    txt_file.write(str(feature.RepMz) + sep) #RepMz
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
    txt_file.write(str(feature.IntensityCosineSimilarity) + sep) ## IntensityCosineSimilarity
    txt_file.write(str(feature.NumTheoPeaks) + sep) ## NumTheoPeaks
    txt_file.write(str(feature.MzErrorSum) + sep) ## MzErrorSum
    txt_file.write(str(feature.MzErrorSumBase) + sep) ## MzErrorSumBase
    txt_file.write(str(feature.Score) + sep) ## Score
    txt_file.write(str(feature.Label) + "\n") ## Label
  txt_file.close()

def sort_features(multicharge_features):
  multicharge_features.sort(key=lambda x: x.Score, reverse=True)

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
          apex_diff = [abs(feature.ApexElutionTime - f.ApexElutionTime) for f in overlapping_features]
          selected_feature = overlapping_features[apex_diff.index(min(apex_diff))]
          index = next((i for i, item in enumerate(featureList_2) if item is not None and item.FeatureID == selected_feature.FeatureID), -1)
          featureList_2[index].used = True
          tmp_common_features.append((featureList_idx_2, index))
      tmp_common_features.insert(0, (0, feature_idx))
      common_features.append(tmp_common_features)
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
  for feature_idx in range(0, len(temp_features)):
    f = temp_features[feature_idx]
    start_rt, end_rt = _get_overlap(f, feature, time_tol)
    overlapping_rt_range = end_rt - start_rt
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

def print_distribution_combined(common_features):
  labels = [len(i) for i in common_features]
  labels_counter_1 = sorted(Counter(labels).items(), reverse = True)
  print("Label Counter", labels_counter_1)
 

def return_list(multicharge_features_replicate):
  keys = list(multicharge_features_replicate.keys())
  keys.sort()
  featureList_all = []
  for i in keys:
    featureList_all.append(deepcopy(multicharge_features_replicate[i]))
  ## Remove the used keyword
  for fl in featureList_all:
    featureList_all
    for f in fl:
      if hasattr(f, "used"):
        delattr(f, 'used')
  return featureList_all

def filter_features_tools(topfd_features_replicate, promex_features_replicate, flashdeconv_features_replicate, xtract_features_replicate):
  feature_keys = sorted(list(topfd_features_replicate.keys()))
  for i in feature_keys:
    t = topfd_features_replicate[i]
    p = promex_features_replicate[i]
    f = flashdeconv_features_replicate[i]
    x = xtract_features_replicate[i]
    print("Before - Rep", i, "lengths are", len(t), len(p), len(f),len(x))
    length = sorted([len(t), len(p), len(f),len(x)])
    selected_length = length[0]
    topfd_features_replicate[i] = topfd_features_replicate[i][0:selected_length]
    promex_features_replicate[i] = promex_features_replicate[i][0:selected_length]
    flashdeconv_features_replicate[i] = flashdeconv_features_replicate[i][0:selected_length]
    xtract_features_replicate[i] = xtract_features_replicate[i][0:selected_length]
    ######## Print new lengths
    t = topfd_features_replicate[i]
    p = promex_features_replicate[i]
    f = flashdeconv_features_replicate[i]
    x = xtract_features_replicate[i]
    print("Rep", i, "lengths are", len(t), len(p), len(f),len(x))


def get_intensity_correlation(common_features, featureList, tool):
  if tool == 1:
    featureList[0].sort(key=lambda x: (x is None, x.Score), reverse=True)
  else:
    featureList[0].sort(key=lambda x: (x is None, x.Abundance), reverse=True)
  for f_idx in range(1, len(featureList)):
    featureList[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  common_10 = [i for i in common_features if len(i) == len(featureList)]
  intens = []
  for elem in common_10:
    f = [featureList[i[0]][i[1]] for i in elem]
    intens.append([log10(i.Abundance) for i in f])
  intens_arr = np.array(intens)
  intens_correlation = np.corrcoef(intens_arr, rowvar=False)
  return intens_correlation

def score_features(multicharge_features, model_file):
  model = tf.keras.models.load_model(model_file)
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
  parser = argparse.ArgumentParser(description='Get feature and quantitaive reproducability.')
  parser.add_argument("-f", "--dataDir", default = r"E:\TopFD_Published_Data\05_Post_Artifacts_Removal\01_Ovarian_Cancer", help="Directory containing features")
  parser.add_argument("-m", "--model_file", default = r"E:\TopFD_Published_Data\04_Training_Data_ECScore_Model\model\ecscore.h5", help="ECScore model file")
  parser.add_argument("-e", "--tolerance", default = 10E-6, help="Mass tolerance (ppm)", type = float)
  parser.add_argument("-t", "--timeTolerance", default = 1.0, help="Time tolerance (minutes)", type = float)
  args = parser.parse_args()  
 
  score_cutoff = 0.5
  topfd_feature_dir = os.path.join(args.dataDir, "TopFD")
  promex_feature_dir = os.path.join(args.dataDir, "ProMex")
  flashdeconv_feature_dir = os.path.join(args.dataDir, "FlashDeconv")
  xtract_feature_dir = os.path.join(args.dataDir, "Xtract")
  data_type = args.dataDir.split('\\')[-1]
 
  feature_files = [f for f in os.listdir(topfd_feature_dir) if f.endswith(r".csv")]
  topfd_features_replicate = {}
  for file in feature_files:
    file_idx = int(file.split('_')[1])
    print("Processing File:", file_idx, file)
    topfd_features = get_features(os.path.join(topfd_feature_dir, file))
    score_features(topfd_features, args.model_file)
    topfd_features = [i for i in topfd_features if i.Score >= score_cutoff]
    sort_features(topfd_features)
    topfd_features_replicate[file_idx] = topfd_features

  ###### ProMex
  print("\nLoad Features")  
  feature_files = [f for f in os.listdir(promex_feature_dir) if f.endswith(r".csv")]
  promex_features_replicate = {}
  for file in feature_files:
    file_idx = int(file.split('_')[1])
    print("Processing File:", file_idx, file)
    promex_features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(promex_feature_dir, file))
    sort_features(promex_features.featureList)
    promex_features_replicate[file_idx] = promex_features.featureList  
  
  ###### FlashDeconv
  print("\nLoad Features")  
  feature_files = [f for f in os.listdir(flashdeconv_feature_dir) if f.endswith(r".csv")]
  flashdeconv_features_replicate = {}
  for file in feature_files:
    file_idx = int(file.split('_')[1])
    print("Processing File:", file_idx, file)
    flashdeconv_features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(flashdeconv_feature_dir, file))
    flashdeconv_features = [i for i in flashdeconv_features.featureList]
    sort_features(flashdeconv_features)
    flashdeconv_features_replicate[file_idx] = flashdeconv_features
    
  ###### Xtract
  print("\nLoad Features")  
  feature_files = [f for f in os.listdir(xtract_feature_dir) if f.endswith(r".csv")]
  xtract_features_replicate = {}
  for file in feature_files:
    file_idx = int(file.split('_')[1])
    print("Processing File:", file_idx, file)
    xtract_features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(xtract_feature_dir, file))
    xtract_features.featureList.sort(key=lambda x: x.FeatureID, reverse=False)
    xtract_features_replicate[file_idx] = xtract_features.featureList
  
  # ########### Shortlist features
  print(len(topfd_features_replicate[1]), len(promex_features_replicate[1]), len(flashdeconv_features_replicate[1]), len(xtract_features_replicate[1]))
  filter_features_tools(topfd_features_replicate, promex_features_replicate, flashdeconv_features_replicate, xtract_features_replicate)  
  
  print("TopFD")
  topfd_features = return_list(topfd_features_replicate)
  common_features_topfd = get_common_features_first_replicate(topfd_features, args.tolerance, args.timeTolerance, 0)
  print_distribution_combined(common_features_topfd)
  label_features(common_features_topfd, topfd_features, 0)

  print("ProMex")  
  promex_features = return_list(promex_features_replicate)
  common_features_promex = get_common_features_first_replicate(promex_features, args.tolerance, args.timeTolerance, 0)
  print_distribution_combined(common_features_promex)
  label_features(common_features_promex, promex_features, 0)
  
  print("FlashDeconv")
  flashdeconv_features = return_list(flashdeconv_features_replicate)
  common_features_flashdeconv = get_common_features_first_replicate(flashdeconv_features, args.tolerance, args.timeTolerance, 0)
  print_distribution_combined(common_features_flashdeconv)
  label_features(common_features_flashdeconv, flashdeconv_features, 0)

  print("Xtract")
  xtract_features = return_list(xtract_features_replicate)
  common_features_xtract = get_common_features_first_replicate(xtract_features, args.tolerance, args.timeTolerance, 0)
  print_distribution_combined(common_features_xtract)
  label_features(common_features_xtract, xtract_features, 0)
  
  topfd_sum = sum([len(i) for i in common_features_topfd])
  promex_sum = sum([len(i) for i in common_features_promex])
  flashdeconnv_sum = sum([len(i) for i in common_features_flashdeconv])
  xtract_sum = sum([len(i) for i in common_features_xtract])
  print("TopFD Sum:", topfd_sum, ", ProMex Sum:", promex_sum, ", FlashDeconnv Sum:", flashdeconnv_sum, ", and Xtract Sum:", xtract_sum, )
   
  ############## Quantative reproducability
  intens_correlation_topfd = get_intensity_correlation(common_features_topfd, topfd_features, 0)
  intens_correlation_promex = get_intensity_correlation(common_features_promex, promex_features, 0)
  intens_correlation_flshdeconv = get_intensity_correlation(common_features_flashdeconv, flashdeconv_features, 0)
  intens_correlation_xtract = get_intensity_correlation(common_features_xtract, xtract_features, 0)
  vmin = min(np.min(intens_correlation_topfd), np.min(intens_correlation_promex), np.min(intens_correlation_flshdeconv), np.min(intens_correlation_xtract))
  vmax = max(np.max(intens_correlation_topfd), np.max(intens_correlation_promex), np.max(intens_correlation_flshdeconv), np.max(intens_correlation_xtract))
  
  topfd_min = np.min(intens_correlation_topfd)
  promex_min = np.min(intens_correlation_promex)
  flshdeconv_min = np.min(intens_correlation_flshdeconv)
  xtract_min = np.min(intens_correlation_xtract)
  
  anot = False
  colormap = "YlGnBu"
  fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
  cbar_ax = fig.add_axes([.75, .3, .03, .4])
  sns.heatmap(intens_correlation_topfd, linewidth=.5, linecolor="black", annot=anot, fmt = '.2f', square=True, ax=axes[0,0], vmin=vmin, vmax=vmax, cmap=colormap, cbar_ax=cbar_ax, xticklabels=range(1, intens_correlation_topfd.shape[0]+1), yticklabels=range(1, intens_correlation_topfd.shape[0]+1))
  sns.heatmap(intens_correlation_promex, linewidth=.5, linecolor="black", annot=anot, fmt = '.2f', square=True, ax=axes[0, 1], vmin=vmin, vmax=vmax, cmap=colormap, cbar_ax=cbar_ax, xticklabels=range(1, intens_correlation_promex.shape[0]+1), yticklabels=range(1, intens_correlation_promex.shape[0]+1))
  sns.heatmap(intens_correlation_flshdeconv, linewidth=.5, linecolor="black", annot=anot, fmt = '.2f', square=True, ax=axes[1, 0], vmin=vmin, vmax=vmax, cmap=colormap, cbar_ax=cbar_ax, xticklabels=range(1, intens_correlation_flshdeconv.shape[0]+1), yticklabels=range(1, intens_correlation_flshdeconv.shape[0]+1))
  sns.heatmap(intens_correlation_xtract, linewidth=.5, linecolor="black", annot=anot, fmt = '.2f', square=True, ax=axes[1, 1], vmin=vmin, vmax=vmax, cmap=colormap, cbar_ax=cbar_ax, xticklabels=range(1, intens_correlation_xtract.shape[0]+1), yticklabels=range(1, intens_correlation_xtract.shape[0]+1))
  axes[0,0].title.set_text('TopFD')
  axes[0,1].title.set_text('ProMex')
  axes[1,0].title.set_text('FlashDeconv')
  axes[1,1].title.set_text('Xtract')
  axes[0,0].tick_params(bottom=False)
  axes[0,1].tick_params(left=False, bottom=False)
  axes[1,1].tick_params(left=False)
  fig.tight_layout(rect=[0, 0, .8, 1])
  plt.savefig("HeatMap_" + data_type + ".jpg", dpi=1500)
  plt.show()

  