#!/usr/bin/python3

import os 
import argparse  
from copy import deepcopy
from collections import Counter
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
  multicharge_features.featureList.sort(key=lambda x: x.Score, reverse=True)
  
def shortlist_features(multicharge_features, n):
  if n == -1:
    return multicharge_features.featureList
  return multicharge_features.featureList[0:n]

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

def get_common_features_first_replicate(features, tolerance, time_tol):
  featureList_all = deepcopy(features)  
  ## Sort features in replicate 2 onwards by mass for binary search
  featureList = featureList_all[0]
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

def get_common_features_all_replicates(features, tolerance, time_tol):
  featureList_all = deepcopy(features)
  for f_idx in range(0, len(featureList_all)): ## Sort rest  of features by mass for binary search
    featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
  ## Get common features based on the RT overlap
  common_features = []
  for featureList_idx in range(0, len(featureList_all)):
    print("processing feature List:", featureList_idx)
    featureList = featureList_all[featureList_idx]
    featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
    for feature_idx in range(0, len(featureList)):
      if feature_idx%10000 == 0:
        print("processing feature:", feature_idx)
      feature = featureList[feature_idx]
      if hasattr(feature, 'used'):
        continue
      else:
        tmp_common_features = []
        for featureList_idx_2 in range(featureList_idx + 1, len(featureList_all)):
          featureList_2 = featureList_all[featureList_idx_2]
          temp_features = _getMatchedFeaturesIndicesMultiCharge(featureList_2, feature, tolerance)
          overlapping_features = _get_overlapping_featuresMultiCharge(temp_features, feature, time_tol)
          if len(overlapping_features) > 0:
            apex_diff = [abs(feature.ApexElutionTime - f.ApexElutionTime) for f in overlapping_features]
            selected_feature = overlapping_features[apex_diff.index(min(apex_diff))]
            index = next((i for i, item in enumerate(featureList_2) if item is not None and item.FeatureID == selected_feature.FeatureID), -1)
            featureList_2[index].used = True
            tmp_common_features.append((1, index))
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

def label_features(common, featureList_all):
  for j in range(0, len(featureList_all)):
    featureList_all[j].sort(key=lambda x: (x is None, x.Abundance), reverse=True)
    for f_idx in range(j + 1, len(featureList_all)):
      featureList_all[f_idx].sort(key=lambda x: (x is None, x.MonoMass), reverse=False)
    for elem in common:
      if elem[0][0] == j:
        for i in elem:
          featureList_all[i[0]][i[1]].Label = len(elem)

def write_Combined_feature_detailed_10(common_features_all, featureList_all, filename):
  featureList_all[0].sort(key=lambda x: (x is None, x.Abundance), reverse=True)
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
      RepMz = [featureList_all[idx[0]][idx[1]].RepMz for idx in feature_idx]
      file.write("Representative_MZ: " + str(list(RepMz)) + "\n")
      RepCharge = [featureList_all[idx[0]][idx[1]].RepCharge for idx in feature_idx]
      file.write("Representative_Charge: " + str(list(RepCharge)) + "\n")
      RepScore = [featureList_all[idx[0]][idx[1]].Score for idx in feature_idx]
      file.write("Score: " + str(list(RepScore)) + "\n")
      file.write("FEATURE_END \n\n")
  file.close()

def print_distribution_combined(common_features):
  labels = [len(i) for i in common_features]
  labels_counter_1 = sorted(Counter(labels).items(), reverse = True)
  print("Label Counter", labels_counter_1)
  
if __name__ == "__main__":
  print("Label Features!!!!")
  parser = argparse.ArgumentParser(description='Generate data matrix for the mzML data.')
  parser.add_argument("-f", "--feature_dir", default = r"E:\TopFD_Published_Data\03_Extracted_Features\01_Ovarian_Cancer\TopFD", help="Directory containing the feature files")
  parser.add_argument("-e", "--tolerance", default = 10E-6, help="Mass tolerance (ppm)", type = float)
  parser.add_argument("-t", "--timeTolerance", default = 1.0, help="Time tolerance (minutes)", type = float)
  args = parser.parse_args()
  
  feature_files = [f for f in os.listdir(args.feature_dir) if f.endswith(r".feature") or f.endswith(r".csv")]
  data_type = args.feature_dir.split('\\')[-2]

  multicharge_features_replicate = {}
  for file in feature_files:
    if data_type == '01_Ovarian_Cancer' or data_type == 'OC':
      file_idx = int(file.split('rep')[1].split('_')[0])
    if data_type == '03_SW480_Colorectal_Cancer' or data_type == '02_SW620_Colorectal_Cancer' or data_type == 'SW480' or data_type == 'SW620':
      file_idx = int(file.split('RPLC_0')[1].split('_')[0])
    if data_type == '04_WHIM2_Breast_Cancer' or data_type == '05_WHIM16_Breast_Cancer' or data_type == 'WHIM2' or data_type == 'WHIM16':
      file_idx = int(file.split('techrep')[1].split('_')[0])
    print("Processing File:", file_idx, file)
    multicharge_features = get_features(os.path.join(args.feature_dir, file))
    multicharge_features_replicate[file_idx] = multicharge_features
  
  features = return_list(multicharge_features_replicate)
  common_features_single = get_common_features_first_replicate(features, args.tolerance, args.timeTolerance)
  print_distribution_combined(common_features_single)  
  label_features(common_features_single, features)
  
  if data_type == '01_Ovarian_Cancer' or data_type == 'OC':
    replicate_count_tol_for_labeling = 8
  if data_type == '03_SW480_Colorectal_Cancer' or data_type == '02_SW620_Colorectal_Cancer' or data_type == 'SW480' or data_type == 'SW620':
    replicate_count_tol_for_labeling = 3
  if data_type == '04_WHIM2_Breast_Cancer' or data_type == '05_WHIM16_Breast_Cancer' or data_type == 'WHIM2' or data_type == 'WHIM16':
    replicate_count_tol_for_labeling = 5
    
  rep1_features = []
  for f in features[0]:
    if f.Label == 1:
      f.Label = 0
      rep1_features.append(f)
  for f in features[0]:
    if f.Label >= replicate_count_tol_for_labeling:
      f.Label = 1
      rep1_features.append(f)
    
  output_env_coll_list(os.path.join(args.feature_dir, data_type + "_rep_" + str(1) + "_labeled.csv"), rep1_features)
