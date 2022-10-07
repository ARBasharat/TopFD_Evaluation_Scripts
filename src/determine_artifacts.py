
from read_model_features import get_features
import os
import numpy as np
import tensorflow as tf
from math import log
from copy import deepcopy
from matplotlib import pyplot as plt
from multiChargeFeatureList import MultiChargeFeatureList

def read_features(feature_dir, file_type, idx_reader):
  files = [f for f in os.listdir(feature_dir) if f.endswith(r".csv") or f.endswith(r".ms1ft") or f.endswith(r".tsv") or f.endswith(r".xlsx")]
  features_replicate = {}
  for file in files:
    if idx_reader == 1:
      file_idx = int(file.split('p')[1].split('_')[0])
    if idx_reader == 2:
      file_idx = int(file.split('_')[1].split('_')[0])
    if idx_reader == 3:
      file_idx = int(file.split('_0')[1].split('.')[0])
    if idx_reader == 4:
      file_idx = int(file.split('_')[1].split('.')[0])
    
    # print("Processing File:", file_idx, "-", file)
    if file_type == "ProMex":
      features = MultiChargeFeatureList.get_features_ms1ft(os.path.join(feature_dir , file))
    if file_type == "FlashDeconv":
      features = MultiChargeFeatureList.get_features_FD_tsv(os.path.join(feature_dir , file))
    if file_type == "Xtract":
      features = MultiChargeFeatureList.get_features_xtract_xls(os.path.join(feature_dir , file))
    if file_type == "TopFD":
      features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(feature_dir , file))
      
    features_replicate[file_idx] = features
  return features_replicate

def create_directory(base_direc, target_directory_name):
  target_directory = os.path.join(base_direc, target_directory_name)
  if not os.path.exists(target_directory ):
    os.mkdir(target_directory )
  return target_directory

def get_mz(mass, charge):
  proton = 1.00727
  return (mass + (charge * proton)) / charge
  
def get_mass(mz, charge):
  proton = 1.00727
  return (mz * charge) - (charge * proton)

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

def _getExtendMasses(mass, isotopes):
  IM = 1.00235
  result = []
  for shift in range(-isotopes, isotopes+1):
    result.append(mass + shift*IM)
  return result

##############################################
# def _getLowHarmonicMasses(feature):
#   IM = 1.00235
#   low_harmonic_masses = []
#   for charge in range(feature.MinCharge, feature.MaxCharge + 1):
#     mz = get_mz(feature.MonoMass, charge)
#     masses = []
#     for c_idx in range(min_charge, max_charge + 1):
#       c = charge * c_idx
#       if c <= max_charge:
#         masses.append(get_mass(mz, c))
#     for mass in masses:
#       for shift in range(-isotopes, isotopes+1):
#         low_harmonic_masses.append(mass + shift*IM)
#   return sorted(low_harmonic_masses)

# def _getHighHarmonicMasses(feature):
#   IM = 1.00235
#   high_harmonic_masses = []
#   for charge in range(feature.MinCharge, feature.MaxCharge + 1):
#     mz = get_mz(feature.MonoMass, charge)
#     masses = []
#     for c_idx in range(min_charge, max_charge + 1):
#       c = int(charge / c_idx)      
#       if c <= 2:
#         # print(c, env_utils.get_mass(mz, c))
#         masses.append(get_mass(mz, c))
#     for mass in masses:
#       for shift in range(-isotopes, isotopes+1):
#         high_harmonic_masses.append(mass + shift*IM)
#   return sorted(high_harmonic_masses)

# def _getIsotopologueMasses(feature):
#   IM = 1.00235
#   return [feature.MonoMass + IM, feature.MonoMass -IM]

##############################################
def _getLowHarmonicMasses(feature):
  IM = 1.00235
  masses = []
  for c_idx in range(min_charge, max_charge + 1):
    masses.append(feature.MonoMass*c_idx)
  
  low_harmonic_masses = []
  for mass in masses:
    for shift in range(-isotopes, isotopes+1):
      low_harmonic_masses.append(mass + shift*IM)
  return sorted(low_harmonic_masses)

def _getHighHarmonicMasses(feature):
  IM = 1.00235
  masses = []
  for c_idx in range(min_charge, max_charge + 1):
    masses.append(feature.MonoMass/c_idx)
    
  high_harmonic_masses = []
  for mass in masses:
    for shift in range(-isotopes, isotopes+1):
      high_harmonic_masses.append(mass + shift*IM)
  return sorted(high_harmonic_masses)

def _getIsotopologueMasses(feature):
  IM = 1.00235
  return [feature.MonoMass + IM, feature.MonoMass -IM]

###################################################
def get_overlap(f, feature):
  start_rt = max(f.MinElutionTime, feature.MinElutionTime)
  end_rt = min(f.MaxElutionTime, feature.MaxElutionTime)
  return (start_rt, end_rt)

def _check_rt_overlap(f, feature, time_tol):
  start_rt, end_rt = get_overlap(f, feature)
  overlapping_rt_range = end_rt - start_rt
  status = False
  if overlapping_rt_range > 0:
    feature_range = feature.MaxElutionTime - feature.MinElutionTime
    percent_overlap = overlapping_rt_range/feature_range 
    if percent_overlap > time_tol: ## 80% coverage
      status = True
  return status

def check_in_list(mass, mass_list, error_tol):
  for m in mass_list:
    if abs(m-mass) < error_tol:
      return True
  return False

def determine_artifacts(features):
  # import copy
  # features = copy.deepcopy(multicharge_features_replicate[1]) ###########///////////////////////
  low_harmonics = []
  high_harmonics = []
  isotopologues = []
  for init_idx in range(0, len(features)):
    # if init_idx%5000 == 0:
    #   print("Processing", init_idx, "of", len(features))
    feature = features[init_idx]
    # if feature.status != "Valid": continue
    error_tol = error_tole * feature.MonoMass
    shortlisted_features = []
    for f_idx in range(0, len(features)):
      f = features[f_idx]
      if feature.status == "Valid":
        if _check_rt_overlap(f, feature, 0.8) and f.Abundance >= feature.Abundance:
          shortlisted_features.append(f_idx)
    
    if len(shortlisted_features) > 0:
      low_harmonic_masses = _getLowHarmonicMasses(feature)
      high_harmonic_masses = _getHighHarmonicMasses(feature)
      isotoplogue_masses = _getIsotopologueMasses(feature)
      feature_cluseter = [feature]
      for f_idx in shortlisted_features:
        f = features[f_idx]
        mass = f.MonoMass
        if check_in_list(mass, low_harmonic_masses, error_tol):
          # if feature.Score <= f.Score:
          if feature.Abundance < f.Abundance:
            feature_cluseter.append(f)
            # feature.status = "low_harmonics"
            # low_harmonics.append((feature.FeatureID, feature.MonoMass, feature.Score, feature.Abundance, (feature.MinCharge, feature.MaxCharge), (round(feature.MinElutionTime, 2), round(feature.MaxElutionTime, 2)), feature.Label, " -------- \n", f.FeatureID, f.MonoMass, f.Score,  f.Abundance, (f.MinCharge, f.MaxCharge), (round(f.MinElutionTime, 2), round(f.MaxElutionTime, 2)), f.Label))
          # break
        elif check_in_list(mass, high_harmonic_masses, error_tol):
          # if feature.Score <= f.Score:
          if feature.Abundance < f.Abundance:
            feature_cluseter.append(f)
            # feature.status = "high_harmonics"
            # high_harmonics.append((feature.FeatureID, feature.MonoMass, feature.Score, feature.Abundance, (feature.MinCharge, feature.MaxCharge), (round(feature.MinElutionTime, 2), round(feature.MaxElutionTime, 2)), feature.Label, " -------- \n", f.FeatureID, f.MonoMass, f.Score,  f.Abundance, (f.MinCharge, f.MaxCharge), (round(f.MinElutionTime, 2), round(f.MaxElutionTime, 2)), f.Label))
          # break
        elif check_in_list(mass, isotoplogue_masses, error_tol):
          # if feature.Score <= f.Score:
          if feature.Abundance < f.Abundance:
            feature_cluseter.append(f)
            # feature.status = "isotopologues"
            # isotopologues.append((feature.FeatureID, feature.MonoMass, feature.Score, feature.Abundance, (feature.MinCharge, feature.MaxCharge), (round(feature.MinElutionTime, 2), round(feature.MaxElutionTime, 2)), feature.Label, " -------- \n", f.FeatureID, f.MonoMass, f.Score,  f.Abundance, (f.MinCharge, f.MaxCharge), (round(f.MinElutionTime, 2), round(f.MaxElutionTime, 2)), f.Label))
          # break
    
      max_score = max([i.Abundance for i in feature_cluseter])
      if len(feature_cluseter) > 1:
        for i in feature_cluseter:
          if i.Abundance == max_score:
            continue
          i.status = "artifact"
    
  valid_sum = sum([1 for f in features if f.status == "Valid"])
  artifact_sum = sum([1 for f in features if f.status == "artifact"])
  # artifact_ids = [f.FeatureID for f in features if f.status == "artifact"]

  # low_harmonics_sum = sum([1 for f in features if f.status == "low_harmonics"])
  # high_harmonics_sum = sum([1 for f in features if f.status == "high_harmonics"])
  # isotopologues_sum = sum([1 for f in features if f.status == "isotopologues"])
  print("Number of Valid features:", valid_sum, "and Artifacts:", str(artifact_sum))
  print("Percentage of Valid features:", str(round(valid_sum/len(features)*100, 2)) + "%", "and Artifacts:", str(round(artifact_sum/len(features)*100, 2)) + "%")
  
  # print("Number of Valid features:", valid_sum, "and Artifacts:", str(low_harmonics_sum) + ",", str(high_harmonics_sum) + ",", str(isotopologues_sum))
  # print("Percentage of Valid features:", str(round(valid_sum/len(features)*100, 2)) + "%", "and Artifacts:", str(round(low_harmonics_sum/len(features)*100, 2)) + "%" , str(round(high_harmonics_sum/len(features)*100, 2)) + "%", str(round(isotopologues_sum/len(features)*100, 2)) + "%")
  return low_harmonics, high_harmonics, isotopologues

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
    
################### Read Features
error_tole = 10E-6
time_tol = 0.8
min_charge = 2
max_charge = 30
isotopes = 10

################ Output Directory
topfd_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\OC"
promex_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\ProMex\OC"
flashdeconv_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\FlashDeconv\OC"
xtract_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\Xtract\OC"

# topfd_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\SW_620"
# promex_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\ProMex\SW_620"
# flashdeconv_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\FlashDeconv\SW_620"
# xtract_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\Xtract\SW_620"

output_dir = create_directory(os.path.dirname(topfd_feature_dir), "Artifact_removal_new")
data_type = topfd_feature_dir.split('Sep_19_v3\\')[1]
if data_type == "OC":
  output_dir = create_directory(output_dir, "OC")
else:
  output_dir = create_directory(output_dir, "SW_620")

output_dir = create_directory(output_dir, "TopFD")
###################

print("\nTopFD")  
cutoff = 0.5
feature_files = [f for f in os.listdir(topfd_feature_dir) if f.endswith(r".feature")]
multicharge_features_replicate = {}
for file in feature_files:
  if data_type == "OC":  
    file_idx = int(file.split('rep')[1].split('_')[0])
  else:
    file_idx = int(file.split('_0')[1].split('_')[0])
  print("Processing File:", file_idx, file)
  multicharge_features = get_features(os.path.join(topfd_feature_dir, file))
  score_model_NNNN(multicharge_features)
  for i in multicharge_features: i.status = "Valid"  
  multicharge_features.sort(key=lambda x: x.Score, reverse=False)
  multicharge_features = [i for i in multicharge_features if i.Score >= cutoff]
  multicharge_features_replicate[file_idx] = multicharge_features
#####
topfd_features_status = []
for rep_idx in range(1, len(multicharge_features_replicate) + 1):
  print("Processing Replicate", rep_idx)
  topfd_feature_info = multicharge_features_replicate[rep_idx]
  determine_artifacts(topfd_feature_info)
  topfd_features_status.append(topfd_feature_info)
  shortlisted_features = [i for i in topfd_feature_info if i.status == "Valid"]
  output_fname = os.path.join(output_dir, "rep_"+ str(rep_idx) + "_multiCharge_features.csv") 
  output_env_coll_list(output_fname, shortlisted_features)
print("\n")
  
##############################################################################
print("ProMex")
promex_features_replicate = read_features(promex_feature_dir, "ProMex", 3)
# promex_features_replicate = read_features(promex_feature_dir, "ProMex", 1)
output_dir = create_directory(output_dir, "ProMex")
promex_features_status = []
for rep_idx in range(1, len(promex_features_replicate) + 1):
  print("Processing Replicate", rep_idx)
  promex_feature_info = deepcopy(promex_features_replicate[rep_idx])
  for i in promex_feature_info:
     i.status = "Valid"
  promex_feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=False)
  determine_artifacts(promex_feature_info)
  promex_features_status.append(promex_feature_info)
  shortlisted_features = [i for i in promex_feature_info if i.status == "Valid"]
  MultiChargeFeatureList.print_features_2(shortlisted_features, rep_idx, output_dir)
print("\n")

# ##############################################################################
print("flashdeconv")
flashdeconv_features_replicate = read_features(flashdeconv_feature_dir, "FlashDeconv", 3)
# flashdeconv_features_replicate = read_features(flashdeconv_feature_dir, "FlashDeconv", 1)
output_dir = create_directory(output_dir, "FlashDeconv")
flashdeconv_features_status = []
for rep_idx in range(1, len(flashdeconv_features_replicate) + 1):
  print("Processing Replicate", rep_idx)
  flashdeconv_feature_info = deepcopy(flashdeconv_features_replicate[rep_idx])
  for i in flashdeconv_feature_info:
     i.status = "Valid"
  flashdeconv_feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=False)
  determine_artifacts(flashdeconv_feature_info)
  flashdeconv_features_status.append(flashdeconv_feature_info)
  shortlisted_features = [i for i in flashdeconv_feature_info if i.status == "Valid"]
  MultiChargeFeatureList.print_features_2(shortlisted_features, rep_idx, output_dir)
print("\n")

# ##############################################################################
print("xtract")
xtract_features_replicate = read_features(xtract_feature_dir, "Xtract", 4)
output_dir = create_directory(output_dir, "Xtract")
xtract_features_status = []
for rep_idx in range(1, len(xtract_features_replicate) + 1):
  print("Processing Replicate", rep_idx)
  xtract_feature_info = deepcopy(xtract_features_replicate[rep_idx])
  for i in xtract_feature_info:
     i.status = "Valid"
  xtract_feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=False)
  determine_artifacts(xtract_feature_info)
  xtract_features_status.append(xtract_feature_info)
  shortlisted_features = [i for i in xtract_feature_info if i.status == "Valid"]
  MultiChargeFeatureList.print_features_2(shortlisted_features, rep_idx, output_dir)
print("\n")

# ##############################################################################
promex_features_status[0].featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
flashdeconv_features_status[0].featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
xtract_features_status[0].featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
topfd_features_status[0].sort(key=lambda x: (x is None, x.Abundance), reverse=True)

promex_status = [1 if i.status == "Valid" else 0 for i in promex_features_status[0]]
flashdeconv_status = [1 if i.status == "Valid" else 0 for i in flashdeconv_features_status[0]]
xtract_status = [1 if i.status == "Valid" else 0 for i in xtract_features_status[0]]
topfd_status = [1 if i.status == "Valid" else 0 for i in topfd_features_status[0]]
print(sum(topfd_status), sum(promex_status), sum(flashdeconv_status), sum(xtract_status))
print(sum(topfd_status)/len(topfd_status), sum(promex_status)/len(promex_status), sum(flashdeconv_status)/len(flashdeconv_status), sum(xtract_status)/len(xtract_status))

bin_size = 100
x_p1 = range(1, len(promex_status), bin_size)
p1 = [100*sum(promex_status[0:i])/len(promex_status[0:i]) for i in range(1, len(promex_status), bin_size)]
x_p2 = range(1, len(flashdeconv_status), bin_size)
p2 = [100*sum(flashdeconv_status[0:i])/len(flashdeconv_status[0:i]) for i in range(1, len(flashdeconv_status), bin_size)]
x_p3 = range(1, len(xtract_status), bin_size)
p3 = [100*sum(xtract_status[0:i])/len(xtract_status[0:i]) for i in range(1, len(xtract_status), bin_size)]
x_p4 = range(1, len(topfd_status), bin_size)
p4 = [100*sum(topfd_status[0:i])/len(topfd_status[0:i]) for i in range(1, len(topfd_status), bin_size)]

plt.Figure()
plt.plot(x_p1, p1)
plt.plot(x_p2, p2)
plt.plot(x_p3, p3)
plt.plot(x_p4, p4)
plt.legend(['ProMex', 'FlashDeconv', 'Xtract', 'TopFD'])
plt.show()
plt.close()
