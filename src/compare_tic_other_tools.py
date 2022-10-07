import os
import pymzml
from copy import deepcopy
from matplotlib import pyplot as plt
from multiChargeFeatureList import MultiChargeFeatureList
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

def read_features(feature_dir):
  files = [f for f in os.listdir(feature_dir) if f.endswith(r".csv")]
  features_replicate = {}
  for file in files:
    file_idx = int(file.split('_')[1].split('_')[0])
    # print("Processing File:", file_idx, "-", file)
    features = MultiChargeFeatureList.get_multiCharge_features_evaluation(os.path.join(feature_dir , file))
    features_replicate[file_idx] = features
  return features_replicate

rep_idx = 1

# # ##### OC
mzmlFile = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results\OC\CPTAC_Intact_rep1_15Jan15_Bane_C2-14-08-02RZ.mzml"
topfd_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\OC\TopFD"
promex_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\OC\ProMex"
flashdeconv_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\OC\FlashDeconv"
xtract_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\OC\Xtract"

# promex_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\OC\ProMex"
# flashdeconv_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\OC\FlashDeconv"
# xtract_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\OC\Xtract"
# topfd_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal\OC\TopFD"

# #### SW_620
# mzmlFile = r"C:\Users\Abdul\Documents\CRC_data\SW_620\mzml\20220207_UreaExtracted_SW620_C4_RPLC_01.mzML"
# topfd_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\SW_620\TopFD"
# promex_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\SW_620\ProMex"
# flashdeconv_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\SW_620\FlashDeconv"
# xtract_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal_new\SW_620\Xtract"


# promex_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\SW_620\ProMex"
# flashdeconv_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\SW_620\FlashDeconv"
# xtract_feature_dir = r"C:\Users\Abdul\Documents\GitHub\MS_Feature_Extraction_refactored\00_Results_v2\SW_620\Xtract"
# topfd_feature_dir = r"C:\Users\Abdul\Documents\topfd_cpp_results\Sep_19_v3\Artifact_removal\SW_620\TopFD"


run = pymzml.run.Reader(mzmlFile, MS1_Precision = 5e-6, MSn_Precision = 20e-6)
total_ion_current = []
retention_times =[]
for s in run:
  if s.ms_level == 1:
    # if s.scan_time_in_minutes() > 92.0:
    #   continue
    total_ion_current.append(s.TIC)
    retention_times.append(s.scan_time_in_minutes())

tic_bins = [0] * (int(retention_times[-1]) + 1)
for idx in range(0, len(retention_times)):
  bin_idx = int(retention_times[idx])
  # print(retention_times[idx], bin_idx)
  tic_bins[bin_idx] = tic_bins[bin_idx] + total_ion_current[idx]

##############################################################################
promex_features_replicate = read_features(promex_feature_dir)
features = promex_features_replicate[rep_idx]
feature_info = deepcopy(features)
feature_info.featureList.sort(key=lambda x: (x is None, x.Abundance), reverse=True)

promex_inte_bins = [0] * (int(retention_times[-1]) + 1)
for f_idx in range(0, len(feature_info)):
  # print("Process", f_idx, "out of", len(feature_info))
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
  # print("Process", f_idx, "out of", len(feature_info))
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
  # print("Process", f_idx, "out of", len(feature_info))
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
  # print("Process", f_idx, "out of", len(feature_info))
  f = feature_info[f_idx]
  apex_rt = int(f.ApexElutionTime)
  if apex_rt < len(tic_bins):
    topfd_inte_bins[apex_rt] = topfd_inte_bins[apex_rt] + f.Abundance


# max_inte = max(max(tic_bins), max(promex_inte_bins), max(flashdeconv_inte_bins), max(xtract_inte_bins), max(topfd_inte_bins))
# max_inte = max(max(promex_inte_bins), max(flashdeconv_inte_bins), max(xtract_inte_bins), max(topfd_inte_bins))

print("\n\nCosine Similarity")
max_inte = max(tic_bins)
## TopFD
# max_inte = max(topfd_inte_bins)
tic_bins_n = [i/max_inte for i in tic_bins]
topfd_inte_bins_n = [i/max_inte for i in topfd_inte_bins]
print("TopFD:", 1- cosine(tic_bins_n, topfd_inte_bins_n))
## ProMex
# max_inte = max(promex_inte_bins)
tic_bins_n = [i/max_inte for i in tic_bins]
promex_inte_bins_n = [i/max_inte for i in promex_inte_bins]
print("ProMex:", 1- cosine(tic_bins_n, promex_inte_bins_n))
## FlashDeconv
# max_inte = max(flashdeconv_inte_bins)
tic_bins_n = [i/max_inte for i in tic_bins]
flashdeconv_inte_bins_n = [i/max_inte for i in flashdeconv_inte_bins]
print("FlashDeconv:", 1- cosine(tic_bins_n, flashdeconv_inte_bins_n))
## Xtract
# max_inte = max(xtract_inte_bins)
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
# axes[0,0].tick_params(bottom=False)
# axes[0,1].tick_params(left=False, bottom=False)
# axes[1,1].tick_params(left=False)

plt.tight_layout(pad=2.0)
# plt.savefig("TIC_SW_620.jpg", dpi=1500)
plt.savefig("TIC_OC.jpg", dpi=1500)
plt.show()





# print("Pearson Correlation")
# print("ProMex:", pearsonr(tic_bins_n, promex_inte_bins_n)[0])
# print("FlashDeconv:", pearsonr(tic_bins_n, flashdeconv_inte_bins_n)[0])
# print("Xtract:", pearsonr(tic_bins_n, xtract_inte_bins_n)[0])
# print("TopFD:", pearsonr(tic_bins_n, topfd_inte_bins_n)[0])



# fig, axes = plt.subplots(2, 2)#, sharex=True, sharey=True)
# ## TopFD
# max_inte = max(topfd_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# topfd_inte_bins_n = [i/max_inte for i in topfd_inte_bins]
# print("TopFD:", 1- cosine(tic_bins_n, topfd_inte_bins_n))
# axes[0,0].plot(tic_bins_n) ## 
# axes[0,0].plot(topfd_inte_bins_n)
# axes[0,0].legend(['TIC', 'TopFD'])
# axes[0,0].set_ylabel('Relative Intensity')
# ## ProMex
# max_inte = max(promex_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# promex_inte_bins_n = [i/max_inte for i in promex_inte_bins]
# print("ProMex:", 1- cosine(tic_bins_n, promex_inte_bins_n))
# axes[0,1].plot(tic_bins_n) ##
# axes[0,1].plot(promex_inte_bins_n)
# axes[0,1].legend(['TIC', 'ProMex'])
# ## FlashDeconv
# max_inte = max(flashdeconv_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# flashdeconv_inte_bins_n = [i/max_inte for i in flashdeconv_inte_bins]
# print("FlashDeconv:", 1- cosine(tic_bins_n, flashdeconv_inte_bins_n))
# axes[1,0].plot(tic_bins_n) ##
# axes[1,0].plot(flashdeconv_inte_bins_n)
# axes[1,0].legend(['TIC', 'FlashDeconv'])
# axes[1,0].set_xlabel('Retention Time (minutes)')
# axes[1,0].set_ylabel('Relative Intensity')
# ## Xtract
# max_inte = max(xtract_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# xtract_inte_bins_n = [i/max_inte for i in xtract_inte_bins]
# print("Xtract:", 1- cosine(tic_bins_n, xtract_inte_bins_n))
# axes[1,1].plot(tic_bins_n) ##
# axes[1,1].plot(xtract_inte_bins_n)
# axes[1,1].legend(['TIC', 'Xtract'])
# axes[1,1].set_xlabel('Retention Time (minutes)')

# axes[0,0].tick_params(bottom=False)
# axes[0,1].tick_params(left=False, bottom=False)
# axes[1,1].tick_params(left=False)

# # fig.supxlabel('Retention Time (minutes)')
# # fig.supylabel('Relative Intensity')

# # fig.tight_layout(rect=[0, 0, .8, 1])
# # plt.savefig("TIC_SW_620.jpg", dpi=1500)
# plt.savefig("TIC_OC.jpg", dpi=1500)
# plt.show()


# fig, axes = plt.subplots(2, 2)
# ## TopFD
# max_inte = max(topfd_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# topfd_inte_bins_n = [i/max_inte for i in topfd_inte_bins]
# print("TopFD:", 1- cosine(tic_bins_n, topfd_inte_bins_n))
# axes[0,0].plot(tic_bins_n) ## 
# axes[0,0].plot(topfd_inte_bins_n)
# axes[0,0].legend(['TIC', 'TopFD'])
# axes[0,0].set_ylabel('Relative Intensity')
# ## ProMex
# max_inte = max(promex_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# promex_inte_bins_n = [i/max_inte for i in promex_inte_bins]
# print("ProMex:", 1- cosine(tic_bins_n, promex_inte_bins_n))
# axes[0,1].plot(tic_bins_n) ##
# axes[0,1].plot(promex_inte_bins_n)
# axes[0,1].legend(['TIC', 'ProMex'])
# ## FlashDeconv
# max_inte = max(flashdeconv_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# flashdeconv_inte_bins_n = [i/max_inte for i in flashdeconv_inte_bins]
# print("FlashDeconv:", 1- cosine(tic_bins_n, flashdeconv_inte_bins_n))
# axes[1,0].plot(tic_bins_n) ##
# axes[1,0].plot(flashdeconv_inte_bins_n)
# axes[1,0].legend(['TIC', 'FlashDeconv'])
# axes[1,0].set_xlabel('Retention Time (minutes)')
# axes[1,0].set_ylabel('Relative Intensity')
# ## Xtract
# max_inte = max(xtract_inte_bins)
# tic_bins_n = [i/max_inte for i in tic_bins]
# xtract_inte_bins_n = [i/max_inte for i in xtract_inte_bins]
# print("Xtract:", 1- cosine(tic_bins_n, xtract_inte_bins_n))
# axes[1,1].plot(tic_bins_n) ##
# axes[1,1].plot(xtract_inte_bins_n)
# axes[1,1].legend(['TIC', 'Xtract'])
# axes[1,1].set_xlabel('Retention Time (minutes)')

# plt.tight_layout()
# plt.savefig("TIC_SW_620.jpg", dpi=1500)
# # plt.savefig("TIC_OC.jpg", dpi=1500)
# plt.show()


##############################################################################
##############################################################################
# tic_bins_n = [i/max(tic_bins) for i in tic_bins]
# promex_inte_bins_n = [i/max(promex_inte_bins) for i in promex_inte_bins]
# flashdeconv_inte_bins_n = [i/max(flashdeconv_inte_bins) for i in flashdeconv_inte_bins]
# xtract_inte_bins_n = [i/max(xtract_inte_bins) for i in xtract_inte_bins]
# # topfd_inte_bins_n = [i/max(topfd_inte_bins) for i in topfd_inte_bins]

# plt.Figure()
# plt.plot(tic_bins)
# plt.plot(promex_inte_bins)
# plt.xlabel("Retention Time (minute)")
# plt.ylabel("Relative Intensity")
# plt.legend(['Total Ion Current', 'ProMex'])
# # plt.savefig("OC_tic_p.png", dpi=1000)
# # plt.savefig("SW_620_tic_1p.png", dpi=1000)
# plt.show()
# plt.close()

# plt.Figure()
# plt.plot(tic_bins)
# plt.plot(flashdeconv_inte_bins)
# plt.xlabel("Retention Time (minute)")
# plt.ylabel("Relative Intensity")
# plt.legend(['Total Ion Current', 'FlashDeconv'])
# # plt.savefig("OC_tic_f.png", dpi=1000)
# plt.savefig("SW_620_tic_1f.png", dpi=1000)
# plt.show()
# plt.close()

# plt.Figure()
# plt.plot(tic_bins)
# plt.plot(xtract_inte_bins)
# plt.xlabel("Retention Time (minute)")
# plt.ylabel("Relative Intensity")
# plt.legend(['Total Ion Current', 'Xtract'])
# # plt.savefig("OC_tic_x.png", dpi=1000)
# # plt.savefig("SW_620_tic_1x.png", dpi=1000)
# plt.show()
# plt.close()

# plt.Figure()
# plt.plot(tic_bins)
# plt.plot(topfd_inte_bins)
# plt.xlabel("Retention Time (minute)")
# plt.ylabel("Relative Intensity")
# plt.legend(['Total Ion Current', 'TopFD'])
# # plt.savefig("OC_tic_t.png", dpi=1000)
# plt.savefig("SW_620_tic_1t.png", dpi=1000)
# plt.show()
# plt.close()

# plt.Figure()
# plt.plot(tic_bins_n)
# plt.plot(topfd_inte_bins_n)
# plt.xlabel("Retention Time (minute)")
# plt.ylabel("Relative Intensity")
# plt.legend(['TIC', 'TopFD'])
# plt.show()
# plt.close()

# plt.Figure()
# plt.plot(tic_bins_n)
# plt.plot(promex_inte_bins_n)
# plt.plot(flashdeconv_inte_bins_n)
# plt.plot(xtract_inte_bins_n)
# plt.plot(topfd_inte_bins_n)
# plt.legend(['TIC', 'ProMex', 'FlashDeconv', 'Xtract', 'TopFD'])
# plt.show()
# plt.close()


######### OC
# Cosine Similarity
# ProMex: 0.7674572173564117
# FlashDeconv: 0.7741025740201228
# Xtract: 0.6209620014230507
# TopFD: 0.8079723552623128

############# SW_620
# Cosine Similarity
# ProMex: 0.2358266182500971
# FlashDeconv: 0.7663169812756652
# Xtract: 0.6718540249026032
# TopFD: 0.8137235132595965
