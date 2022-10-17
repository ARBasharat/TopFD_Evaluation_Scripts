#!/usr/bin/python3

import os
import argparse
import pymzml
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
from multiChargeFeatureList import MultiChargeFeatureList

def read_features(feature_dir):
  files = [f for f in os.listdir(feature_dir) if f.endswith(r".csv")]
  features_replicate = {}
  for file in files:
    file_idx = int(file.split('_')[1].split('_')[0])
    features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(feature_dir , file))
    features_replicate[file_idx] = features
  return features_replicate

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Compare TIC with feature Intensity', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-f", "--dataDir", default = r"E:\TopFD_Published_Data\05_Post_Artifacts_Removal\02_SW620_Colorectal_Cancer", help="Directory containing features")
  parser.add_argument("-m", "--mzmlFile", default = r"E:\TopFD_Published_Data\02_mzML_Files\02_SW620_Colorecatal_Cancer\20220207_UreaExtracted_SW620_C4_RPLC_01.mzML", help="mzML file")
  args = parser.parse_args()
  
  rep_idx = 1
  topfd_feature_dir = os.path.join(args.dataDir, "TopFD")
  promex_feature_dir = os.path.join(args.dataDir, "ProMex")
  flashdeconv_feature_dir = os.path.join(args.dataDir, "FlashDeconv")
  xtract_feature_dir = os.path.join(args.dataDir, "Xtract")
  data_type = args.dataDir.split('\\')[-1]
  
  run = pymzml.run.Reader(args.mzmlFile, MS1_Precision = 5e-6, MSn_Precision = 20e-6)
  total_ion_current = []
  retention_times =[]
  for s in run:
    if s.ms_level == 1:
      total_ion_current.append(s.TIC)
      retention_times.append(s.scan_time_in_minutes())
 
  tic_bins = [0] * (int(retention_times[-1]) + 1)
  for idx in range(0, len(retention_times)):
    bin_idx = int(retention_times[idx])
    tic_bins[bin_idx] = tic_bins[bin_idx] + total_ion_current[idx]
  
  ##############################################################################
  promex_features_replicate = read_features(promex_feature_dir)
  features = promex_features_replicate[rep_idx]
  feature_info = deepcopy(features)
  feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
  
  promex_inte_bins = [0] * (int(retention_times[-1]) + 1)
  for f_idx in range(0, len(feature_info)):
    f = feature_info[f_idx]
    apex_rt = int(f.ApexElutionTime)
    if apex_rt < len(tic_bins):
      promex_inte_bins[apex_rt] = promex_inte_bins[apex_rt] + f.Abundance
  
  ##############################################################################
  flashdeconv_features_replicate = read_features(flashdeconv_feature_dir)
  features = flashdeconv_features_replicate[rep_idx]
  feature_info = deepcopy(features)
  feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
  
  flashdeconv_inte_bins = [0] * (int(retention_times[-1]) + 1)
  for f_idx in range(0, len(feature_info)):
    f = feature_info[f_idx]
    apex_rt = int(f.ApexElutionTime)
    if apex_rt < len(tic_bins):
      flashdeconv_inte_bins[apex_rt] = flashdeconv_inte_bins[apex_rt] + f.Abundance
  
  ##############################################################################
  xtract_features_replicate = read_features(xtract_feature_dir)
  features = xtract_features_replicate[rep_idx]
  feature_info = deepcopy(features)
  feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
  
  xtract_inte_bins = [0] * (int(retention_times[-1]) + 1)
  for f_idx in range(0, len(feature_info)):
    f = feature_info[f_idx]
    apex_rt = int(f.ApexElutionTime)
    if apex_rt < len(tic_bins):
      xtract_inte_bins[apex_rt] = xtract_inte_bins[apex_rt] + f.Abundance
  
  # ##############################################################################
  topfd_features_replicate = read_features(topfd_feature_dir)
  features = topfd_features_replicate[rep_idx]
  feature_info = deepcopy(features)
  feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)
  
  topfd_inte_bins = [0] * (int(retention_times[-1]) + 1)
  for f_idx in range(0, len(feature_info)):
    f = feature_info[f_idx]
    apex_rt = int(f.ApexElutionTime)
    if apex_rt < len(tic_bins):
      topfd_inte_bins[apex_rt] = topfd_inte_bins[apex_rt] + f.Abundance
  
  print("\n\nCosine Similarity")
  max_inte = max(tic_bins)
  tic_bins_n = [i/max_inte for i in tic_bins]
  topfd_inte_bins_n = [i/max_inte for i in topfd_inte_bins]
  print("TopFD:", 1- cosine(tic_bins_n, topfd_inte_bins_n))
  tic_bins_n = [i/max_inte for i in tic_bins]
  promex_inte_bins_n = [i/max_inte for i in promex_inte_bins]
  print("ProMex:", 1- cosine(tic_bins_n, promex_inte_bins_n))
  tic_bins_n = [i/max_inte for i in tic_bins]
  flashdeconv_inte_bins_n = [i/max_inte for i in flashdeconv_inte_bins]
  print("FlashDeconv:", 1- cosine(tic_bins_n, flashdeconv_inte_bins_n))
  tic_bins_n = [i/max_inte for i in tic_bins]
  xtract_inte_bins_n = [i/max_inte for i in xtract_inte_bins]
  print("Xtract:", 1- cosine(tic_bins_n, xtract_inte_bins_n))
  
  
  max_inte = max(tic_bins)
  fig, axes = plt.subplots(2, 2, sharey=True, sharex=True)
  axes[0,0].plot(tic_bins_n) ## 
  axes[0,0].plot(topfd_inte_bins_n)
  axes[0,0].legend(['TIC', 'TopFD'])
  axes[0,0].set_ylabel('Relative Intensity')
  axes[0,0].xaxis.set_tick_params(labelbottom=True)
  
  axes[0,1].plot(tic_bins_n) ##
  axes[0,1].plot(promex_inte_bins_n)
  axes[0,1].legend(['TIC', 'ProMex'])
  axes[0,1].xaxis.set_tick_params(labelbottom=True)
  axes[0,1].yaxis.set_tick_params(labelbottom=True)
  
  axes[1,0].plot(tic_bins_n) ##
  axes[1,0].plot(flashdeconv_inte_bins_n)
  axes[1,0].legend(['TIC', 'FlashDeconv'])
  axes[1,0].set_xlabel('Retention Time (minutes)')
  axes[1,0].set_ylabel('Relative Intensity')
  
  axes[1,1].plot(tic_bins_n) ##
  axes[1,1].plot(xtract_inte_bins_n)
  axes[1,1].legend(['TIC', 'Xtract'])
  axes[1,1].set_xlabel('Retention Time (minutes)')
  axes[1,1].yaxis.set_tick_params(labelbottom=True)

  plt.tight_layout(pad=2.0)
  if data_type == '01_Ovarian_Cancer' or data_type == 'OC':
    plt.savefig("TIC_OC.jpg", dpi=1500)
  else:
    plt.savefig("TIC_SW_620.jpg", dpi=1500)
  plt.show()
  plt.close()
