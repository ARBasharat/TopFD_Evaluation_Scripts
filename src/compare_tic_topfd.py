
import os
import pymzml
from matplotlib import pyplot as plt
from multiChargeFeatureList import MultiChargeFeatureList
from read_model_features import get_features

def read_features(feature_dir):
  files = [f for f in os.listdir(feature_dir) if f.endswith(r".csv")]
  features_replicate = {}
  for file in files:
    file_idx = int(file.split('_')[1].split('_')[0])
    # print("Processing File:", file_idx, "-", file)
    features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(feature_dir , file))
    features_replicate[file_idx] = features
  return features_replicate

def get_xic(selected_lines):
  xic = []
  rt = []
  for line in selected_lines:
    if("Retention_Times" in line):
      data = line.split(': ')[1]
      if '[]' in data:
        return xic, rt  
      data = data[1:-2]
      vals = data.split(', ')
      rt = [round(float(i), 4) for i in vals]
    if("Envelope_XIC" in line):
      data = line.split(': ')[1][1:-2]
      vals = data.split(', ')
      xic = [float(i) for i in vals]
  return xic, rt  

def get_end_index(lines, begin_idx):
  idx = begin_idx
  while (idx < len(lines) and "FEATURE_END" not in lines[idx]):
    idx = idx + 1
  return idx

def get_features_xic(topfd_feature_file, retention_times, topfd_features):
  topfd_features_ids = [f.FeatureID for f in topfd_features]
  with open(topfd_feature_file) as f:
    lines = f.readlines()

  feature_id = 0  
  feature_xic = [0] * len(retention_times)
  begin_idx = 0
  while (begin_idx < len(lines)):
    end_idx = get_end_index(lines, begin_idx)
    selected_lines = lines[begin_idx:end_idx +1]
    xic, rt = get_xic(selected_lines)
    if feature_id in topfd_features_ids:
      for i in range(0, len(rt)):
        if rt[i] > max(retention_times):
          continue
        rt_idx = retention_times.index(rt[i])
        feature_xic[rt_idx] = feature_xic[rt_idx] + xic[i]
    feature_id  = feature_id  + 1
    begin_idx = end_idx + 1
    if begin_idx >= len(lines):
      break
  return feature_xic

def get_tic(mzmlFile):
  run = pymzml.run.Reader(mzmlFile, MS1_Precision = 5e-6, MSn_Precision = 20e-6)
  total_ion_current = []
  retention_times =[]
  for s in run:
    if s.ms_level == 1:
      if s.scan_time_in_minutes() > 92.0:
        continue
      total_ion_current.append(s.TIC)
      retention_times.append(round(s.scan_time_in_minutes(), 4))
  
  # tic_bins = [0] * (int(retention_times[-1]) + 1)
  # for idx in range(0, len(retention_times)):
  #   bin_idx = int(retention_times[idx])
  #   print(retention_times[idx], bin_idx)
  #   tic_bins[bin_idx] = tic_bins[bin_idx] + total_ion_current[idx]
  return total_ion_current, retention_times

rep_idx = 1

# ###### OC
# mzmlFile = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\OC\CPTAC_Intact_rep" + str(rep_idx) + "_15Jan15_Bane_C2-14-08-02RZ.mzml"
# ## topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\OC_all_feature_filter_regen_2_newModel\pseudo_files\OT_rep" + str(rep_idx) + "_pseudo.txt"
# topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\OC_Fresh2\pseudo_files\OT_rep" + str(rep_idx) + "_pseudo.txt"
# topfd_feature_csv = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\OC\TopFD_lowScore\rep_" + str(rep_idx) + "_multiCharge_features.csv"

# ##### OC
# mzmlFile = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\OC\CPTAC_Intact_rep" + str(rep_idx) + "_15Jan15_Bane_C2-14-08-02RZ.mzml"
# topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\TopFD\OC\pseudo_files\OT_rep" + str(rep_idx) + "_pseudo.txt"
# topfd_feature_csv = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\TopFD\OC\rep_" + str(rep_idx) + "_multiCharge_features.csv"

###### SW_620
mzmlFile = r"C:\Users\Abdul\Documents\CRC_data\SW_620\mzml\20220207_UreaExtracted_SW620_C4_RPLC_01.mzML"
topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\TopFD\SW_620\pseudo_files\SW620_1_pseudo.txt"
topfd_feature_csv = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\00_All_results\TopFD\SW_620\rep_1_multiCharge_features.csv"

# ###### WHIM_2
# mzmlFile = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\WHIM_2\RP4H_P32_WHIM2_biorep1_techrep1.mzML"
# # topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\WHIM_2_all_feature_filter_regen_2_newModel\pseudo_files\WHIM2_rep" + str(rep_idx) + "_pseudo.txt"
# topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\WHIM_2_Fresh2\pseudo_files\WHIM2_rep" + str(rep_idx) + "_pseudo.txt"
# topfd_feature_csv = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\WHIM_2\TopFD_lowScore\rep_" + str(rep_idx) + "_multiCharge_features.csv"

# # ###### WHIM_16
# mzmlFile = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\WHIM_16\RP4H_P33_WHIM16_biorep1_techrep1.mzML"
# # topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\WHIM_16_all_feature_filter_regen_2_newModel\pseudo_files\WHIM16_rep" + str(rep_idx) + "_pseudo.txt"
# topfd_feature_file = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\feature_data_nn_score\WHIM_16_Fresh2\pseudo_files\WHIM16_rep" + str(rep_idx) + "_pseudo.txt"
# topfd_feature_csv = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\WHIM_16\TopFD_lowScore\rep_" + str(rep_idx) + "_multiCharge_features.csv"

topfd_features = get_features(topfd_feature_csv)
topfd_features.sort(key=lambda x: (x.FeatureID), reverse=False)
total_ion_current, retention_times = get_tic(mzmlFile)
feature_xic = get_features_xic(topfd_feature_file, retention_times, topfd_features)

max_inte = max(total_ion_current)
total_ion_current_n = [i/max_inte for i in total_ion_current]
feature_xic_n = [i/max_inte for i in feature_xic]

tic_bins = [0] * (int(retention_times[-1]) + 1)
for idx in range(0, len(retention_times)):
  bin_idx = int(retention_times[idx])
  # print(retention_times[idx], bin_idx)
  tic_bins[bin_idx] = tic_bins[bin_idx] + total_ion_current[idx]
  
topfd_bins = [0] * (int(retention_times[-1]) + 1)
for idx in range(0, len(retention_times)):
  bin_idx = int(retention_times[idx])
  # print(retention_times[idx], bin_idx)
  topfd_bins[bin_idx] = topfd_bins[bin_idx] + feature_xic[idx]

plt.Figure()
plt.plot(tic_bins)
plt.plot(topfd_bins)
plt.xlabel("Retention Time (minute)")
plt.ylabel("Relative Intensity")
plt.legend(['Total Ion Current', 'TopFD'])
plt.savefig("OC_tic_t.png", dpi=1000)
# plt.savefig("WHIM_2_tic_t.png", dpi=1000)
# plt.savefig("WHIM_16_tic_t.png", dpi=1000)
# plt.savefig("SW_620_tic_t.png", dpi=1000)
plt.show()
plt.close()

plt.Figure()
plt.plot(total_ion_current)
plt.plot(feature_xic)
plt.xlabel("Retention Time (minute)")
plt.ylabel("Relative Intensity")
plt.legend(['Total Ion Current', 'TopFD'])
# plt.savefig("OC_tic_t.png", dpi=1000)
# plt.savefig("WHIM_2_tic_t.png", dpi=1000)
# plt.savefig("WHIM_16_tic_t.png", dpi=1000)
plt.savefig("SW_620_tic_t.png", dpi=1000)
plt.show()
plt.close()

from scipy.stats import pearsonr
print("TopFD:", pearsonr(total_ion_current_n, feature_xic_n)[0])

max_inte = max(tic_bins)
total_ion_current_n = [i/max_inte for i in tic_bins]
feature_xic_n = [i/max_inte for i in topfd_bins]
print("TopFD:", pearsonr(total_ion_current_n, feature_xic_n)[0])
