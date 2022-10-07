#!/usr/bin/python3

import os 
import argparse  
import tensorflow as tf
import numpy as np
from math import log, log10
from copy import deepcopy
from collections import Counter
from multiChargeFeatureList import MultiChargeFeatureList
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from read_model_features import get_features
  
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

def get_common_features_first_replicate(features, tolerance, time_tol):
  featureList_all = deepcopy(features)
  ## Sort features in replicate 2 onwards by mass for binary search
  featureList = featureList_all[0]
  featureList.sort(key=lambda x: (x is None, x.Score), reverse=True)
  for f_idx in range(1, len(featureList_all)): ## Sort rest of features by mass for binary search
    featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  ## Get common features based on the RT overlap
  multi_selected =[]
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
        if len(overlapping_features) > 1:
          fffff = [feature]
          fffff.extend(overlapping_features)
          multi_selected.append((featureList_idx_2, fffff))
        if len(overlapping_features) > 0:
          # coverage = [f.coverage for f in overlapping_features]
          # selected_feature = overlapping_features[coverage.index(max(coverage))]
          
          # inte_diff = [abs(feature.Abundance - f.Abundance) for f in overlapping_features]
          # selected_feature = overlapping_features[inte_diff.index(min(inte_diff))]
          
          # scores = [f.Score for f in overlapping_features]
          # selected_feature = overlapping_features[scores.index(max(scores))]
          
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
  extend_offsets_ = [0, -IM, IM, 2 * -IM, 2 * IM]#, 3 * -IM, 3 * IM, 4 * -IM, 4 * IM]    
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

def label_features(common, featureList_all):
  for j in range(0, len(featureList_all)):
    featureList_all[j].sort(key=lambda x: (x is None, x.Score), reverse=True)
    for f_idx in range(j + 1, len(featureList_all)):
      featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
    for elem in common:
      if elem[0][0] == j:
        for i in elem:
          featureList_all[i[0]][i[1]].Label = len(elem)

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

def filter_features_tools(topfd_features_replicate, flashdeconv_features_replicate):
  feature_keys = sorted(list(topfd_features_replicate.keys()))
  for i in feature_keys:
    t = topfd_features_replicate[i]
    f = flashdeconv_features_replicate[i]
    print("Before - Rep", i, "lengths are", len(t), len(f))
    selected_length = min(len(t), len(f))
    # selected_length = len(p)
    topfd_features_replicate[i] = topfd_features_replicate[i][0:selected_length]
    flashdeconv_features_replicate[i] = flashdeconv_features_replicate[i][0:selected_length]
    ######## Print new lengths
    t = topfd_features_replicate[i]
    f = flashdeconv_features_replicate[i]
    print("Rep", i, "lengths are", len(t), len(f))
    
def score_model_NN(multicharge_features):
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\SW_480_all_feature_nofilter_regen_2\EC_RT_PercentMatchedPeaks_Abundance_PercentConsecPeaks_Top3Correlation_RC_C_MzErrorSum_PCC.h5")
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg\SW_480\moving_avg_model.h5")
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg\SW_480\model.h5")
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\SW_480_all_feature_nofilter_regen_2\model.h5")
  
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\SW_480_all_feature_nofilter_regen_2\model_n.h5")
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg\SW_480\model_n.h5")
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg\SW_480_old_splitting_n\model_n.h5")
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg\SW_480_old\model_n.h5")
  
  model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg_2\SW_480\model_nn.h5")
  
  data = []
  for i in multicharge_features:
    in_feature = [i.EnvcnnScore, i.rt_range, i.PercentMatchedPeaks, log(i.Abundance), i.PercentConsecPeaks, i.Top3Correlation, i.RepCharge, i.charge_range/30, i.MzErrorSum/i.NumTheoPeaks, i.IntensityCorrelation]
    data.append(in_feature)
  test_data = np.array(data)
  scores = model.predict(test_data)
  for i in range(0, len(multicharge_features)):
    multicharge_features[i].Score = scores[i][0]

def score_model_NN2(multicharge_features):
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg_2\SW_480\model_6_1_14_2_5_4_9_3_12_7_leaky06_16.h5")
  # model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\01_Results_moving_avg_2\SW_480\model_6_1_14_2_5_4_9_3_12_7.h5")
  model = tf.keras.models.load_model(r"C:\Users\Abdul\Documents\GitHub\proteomics_cpp_topfd_new_imp\scripts\feature_eval\nn_model_10.h5")
  data = []
  for i in multicharge_features:
    in_feature = [i.EnvcnnScore, i.PercentMatchedPeaks, i.rt_range, i.PercentConsecPeaks, log(i.Abundance), i.charge_range/30, i.MzErrorSum/i.NumTheoPeaks, i.IntensityCorrelation,  i.Top3Correlation,  i.RepCharge]
    # in_feature = [i.EnvcnnScore, i.PercentMatchedPeaks, i.rt_range, i.RepCharge, i.charge_range/30, i.PercentConsecPeaks, i.IntensityCorrelation, log(i.Abundance), i.EvenOddPeakRatios, i.MzErrorSum/i.NumTheoPeaks,  i.Top3Correlation]
    data.append(in_feature)
  test_data = np.array(data)
  scores = model.predict(test_data)
  for i in range(0, len(multicharge_features)):
    multicharge_features[i].Score = scores[i][0]
    
def get_intensity_correlation(common_features, featureList):
  featureList[0].sort(key=lambda x: (x is None, x.Score), reverse=True)
  for f_idx in range(1, len(featureList)):
    featureList[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  
  common_10 = [i for i in common_features if len(i) == len(featureList)]
  intens = []
  for elem in common_10:
    f = [featureList[i[0]][i[1]] for i in elem]
    # intens.append([i.Abundance for i in f])
    intens.append([log10(i.Abundance) for i in f])
 
  intens_arr = np.array(intens)
  intens_correlation = np.corrcoef(intens_arr, rowvar=False)
  return intens_correlation

def plot_heat_map(data, tool):
  import seaborn as sns
  import matplotlib.pylab as plt
  plt.style.use("default")
  plt.figure()
  sns.heatmap(data, linewidth=1, linecolor="black", annot=True, square=True, xticklabels=range(1, data.shape[0]+1), yticklabels=range(1, data.shape[0]+1))
  plt.savefig("HeatMap_" + tool + ".jpg", dpi=1000)
  plt.show()
  

if __name__ == "__main__":
  print("In main function!!")
  parser = argparse.ArgumentParser(description='Generate data matrix for the mzML data.')
  parser.add_argument("-F", "--feature_dir", default = "", help="Scored feature CSV file (Replicate 1)")
  parser.add_argument("-M", "--model_file", default = "", help="Scored feature CSV file (Replicate 2)")
  parser.add_argument("-e", "--tolerance", default = 10E-6, help="Mass tolerance (ppm)", type = float)
  parser.add_argument("-t", "--timeTolerance", default = 1.0, help="Time tolerance (minutes)", type = float)
  parser.add_argument("-n", "--NumFeatures", default = -1, help="Used to keep top N features", type = int)
  args = parser.parse_args()
  
  # args.feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_14_v1_results\features\OC"
  args.feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_14_v1_results\features\SW_620"
  
  data_type = args.feature_dir.split('features\\')[1]
  feature_files = [f for f in os.listdir(args.feature_dir) if f.endswith(r".feature")]
  topfd_features_replicate = {}
  for file in feature_files:
    if data_type == 'OC':
      file_idx = int(file.split('rep')[1].split('_')[0])
    if data_type == 'SW_480' or data_type == 'SW_620':
      file_idx = int(file.split('RPLC_0')[1].split('_')[0])
    if data_type == 'WHIM_2' or data_type == 'WHIM_16':
      file_idx = int(file.split('techrep')[1].split('_')[0])
    print("Processing File:", file_idx, file)
    topfd_features = get_features(os.path.join(args.feature_dir, file))
    score_model_NN2(topfd_features)
    sort_features(topfd_features)
    topfd_features_replicate[file_idx] = topfd_features
                                                     
  
  #################################################################
  # args.flashdeconv_dir = r"C:\Users\Abdul\Documents\CRC_data\SW_480\FlashDeconv"
  args.flashdeconv_dir = r"C:\Users\Abdul\Documents\CRC_data\SW_620\FlashDeconv"
  print("\nLoad Features")  
  feature_files = [f for f in os.listdir(args.flashdeconv_dir) if f.endswith(r".tsv")]
  flashdeconv_features_replicate = {}
  for file in feature_files:
    file_idx = int(file.split('_')[1])
    print("Processing File:", file_idx, file)
    # flashdeconv_features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(args.flashdeconv_dir, file))
    flashdeconv_features = MultiChargeFeatureList.get_features_FD_tsv(os.path.join(args.flashdeconv_dir, file))
    flashdeconv_features = [i for i in flashdeconv_features.featureList if i.Score >= 0.85]
    sort_features(flashdeconv_features)
    flashdeconv_features_replicate[file_idx] = flashdeconv_features
  #################################################################  
  
  
  ########## Shortlist features
  filter_features_tools(topfd_features_replicate, flashdeconv_features_replicate)
  
  print("TopFD")
  topfd_features = return_list(topfd_features_replicate)
  common_features_topfd = get_common_features_first_replicate(topfd_features, args.tolerance, args.timeTolerance)
  plot_distribution_combined(common_features_topfd, True)
  label_features(common_features_topfd, topfd_features)

  print("FlashDeconv")
  flashdeconv_features = return_list(flashdeconv_features_replicate)
  common_features_flashdeconv = get_common_features_first_replicate(flashdeconv_features, args.tolerance, args.timeTolerance)
  plot_distribution_combined(common_features_flashdeconv, True)
  label_features(common_features_flashdeconv, flashdeconv_features)

  
  topfd_sum = sum([len(i) for i in common_features_topfd])
  flashdeconnv_sum = sum([len(i) for i in common_features_flashdeconv])
  print("TopFD Sum:", topfd_sum, ", FlashDeconnv Sum:", flashdeconnv_sum)
   
  ###########
  ###########
  ###########      
  intens_correlation = get_intensity_correlation(common_features_topfd, topfd_features)
  plot_heat_map(intens_correlation, "TopFD")
  
  
  intens_correlation = get_intensity_correlation(common_features_flashdeconv, flashdeconv_features)
  plot_heat_map(intens_correlation, "FlashDeconv")
  

# ############# get features idea for intensity comparison
# topfd_features[0].sort(key=lambda x: (x is None, x.Score), reverse=True)
# for f_idx in range(1, len(topfd_features)):
#   topfd_features[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)

# common_10 = [i for i in common_features_topfd if len(i) == len(topfd_features)]
# intens = []
# for elem_idx in range(0, len(common_10)):
#   elem = common_10[elem_idx]
#   f = [topfd_features[i[0]][i[1]] for i in elem]
#   intens.append([i.Abundance for i in f])
#   # intens.append([log10(i.Abundance) for i in f])
 
# intens_arr = np.array(intens)
# intens_correlation = np.corrcoef(intens_arr, rowvar=False)

### get charge range distribution of features found in all replicates
topfd_charge_range = [f.charge_range + 1 for f in topfd_features[0] if f.Label == len(topfd_features)] 
flashdeconv_charge_range = [(f.MaxCharge - f.MinCharge + 1) for f in flashdeconv_features[0] if f.Label == len(topfd_features)] 

from collections import Counter
print(Counter(topfd_charge_range))
print(Counter(flashdeconv_charge_range))

topfd_charge_range_1 = sum([1 for f in topfd_features[0] if f.MinCharge == 1 and (f.MaxCharge - f.MinCharge + 1) == 1 and f.Label == len(topfd_features)])
flashdeconv_charge_range_1 = sum([1 for f in flashdeconv_features[0] if f.MinCharge == 1 and (f.MaxCharge - f.MinCharge + 1) == 1 and f.Label == len(topfd_features)])
print("Charge 1 Feratures:", topfd_charge_range_1, flashdeconv_charge_range_1)

# # ############### Get features common between all tools. 
# topfd_features_all_rep1 = [f for f in topfd_features[0] if f.Label >= 7]
# promex_features_all_rep1 = [f for f in promex_features[0] if f.Label >= 7]
# flashdeconv_features_all_rep1 = [f for f in flashdeconv_features[0] if f.Label >= 7]
# xtract_features_all_rep1 = [f for f in xtract_features[0] if f.Label >= 7]
# print(len(topfd_features_all_rep1), len(promex_features_all_rep1), len(flashdeconv_features_all_rep1), len(xtract_features_all_rep1))

# common_features_12 = get_common_features_first_replicate([topfd_features_all_rep1, promex_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_12, True)

# common_features_13 = get_common_features_first_replicate([topfd_features_all_rep1, xtract_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_13, True)

# common_features_14 = get_common_features_first_replicate([topfd_features_all_rep1, flashdeconv_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_14, True)
  
# common_features_23 = get_common_features_first_replicate([promex_features_all_rep1, xtract_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_23, True)

# common_features_24 = get_common_features_first_replicate([promex_features_all_rep1, flashdeconv_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_24, True)

# common_features_34 = get_common_features_first_replicate([xtract_features_all_rep1, flashdeconv_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_34, True)

# common_features_234 = get_common_features_first_replicate([promex_features_all_rep1, xtract_features_all_rep1, flashdeconv_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_234, True)

# common_features_134 = get_common_features_first_replicate([topfd_features_all_rep1, xtract_features_all_rep1, flashdeconv_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_134, True)

# common_features_124 = get_common_features_first_replicate([topfd_features_all_rep1, promex_features_all_rep1, flashdeconv_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_124, True)

# common_features_123 = get_common_features_first_replicate([topfd_features_all_rep1, promex_features_all_rep1, xtract_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_123, True)

# common_features_1234 = get_common_features_first_replicate([topfd_features_all_rep1, promex_features_all_rep1, flashdeconv_features_all_rep1, xtract_features_all_rep1], args.tolerance, args.timeTolerance)
# plot_distribution_combined(common_features_1234, True)

