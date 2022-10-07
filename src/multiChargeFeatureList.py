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

#!/usr/bin/python3
import os
import pandas as pd
from multiChargeFeature import MultiChargeFeature

class MultiChargeFeatureList():
  def __init__(self, featureList, featureList_singleCharge):
    self.featureList = featureList
    self.featureList_singleCharge = featureList_singleCharge
  
  @classmethod
  def get_multiCharge_features_evaluation(cls, feature_file):
    ## Called from labelling features
    feature_info = pd.read_csv(feature_file, sep=',')
    featureList = []
    for feature_idx in range(0, len(feature_info)):
      featurePd = feature_info.iloc[feature_idx]
      multiChargeFeature = MultiChargeFeature(int(featurePd["FeatureID"]), int(featurePd["MinScan"]), 
                 int(featurePd["MaxScan"]), int(featurePd["MinCharge"]), int(featurePd["MaxCharge"]), 
                 float(featurePd["MonoMass"]), 
                 # int(featurePd["ShiftNum"]), 
                 # float(featurePd["RepMz"]), 
                  -1,
                  0,
                 float(featurePd["Abundance"]), float(featurePd["MinElutionTime"]), float(featurePd["MaxElutionTime"]), 
                 float(featurePd["ApexElutionTime"]), float(featurePd["ElutionLength"]), 
                 #float(featurePd["EnvCNNScore"]), 
                 #float(featurePd["PercentMatchedPeaks"]), 
                 #float(featurePd["IntensityCorrelation"]), 
                 #float(featurePd["Top3Correlation"])
                 0,
                 0,
                 0,
                 0
                 )
      multiChargeFeature.ChargeRange = int(featurePd["MaxCharge"]-featurePd["MinCharge"]+1)
      if "Label" in featurePd:
        multiChargeFeature.Label = int(featurePd["Label"])
      multiChargeFeature.Score = float(featurePd["Score"])
      # multiChargeFeature.Score = 1
      featureList.append(multiChargeFeature)
    featureList.sort(key=lambda x: (x.Score), reverse=True)
    return cls(featureList, None)
  
  @classmethod
  def get_features_FD_tsv(cls, feature_file):
    feature_info = pd.read_csv(feature_file, sep='\t')
    featureList = []
    for feature_idx in range(0, len(feature_info)):
      featurePd = feature_info.iloc[feature_idx]
      multiChargeFeature = MultiChargeFeature(featurePd["FeatureIndex"], -1, -1, 
                  featurePd["MinCharge"], featurePd["MaxCharge"], featurePd["MonoisotopicMass"], -1, 
                  -1, featurePd["SumIntensity"], featurePd["StartRetentionTime"]/60, featurePd["EndRetentionTime"]/60, 
                  float(featurePd["ApexRetentionTime"])/60, featurePd["RetentionTimeDuration"]/60, -1, -1, -1, -1)
      multiChargeFeature.ChargeRange = featurePd["MaxCharge"]-featurePd["MinCharge"]+1
      multiChargeFeature.Score = featurePd["IsotopeCosineScore"]
      featureList.append(multiChargeFeature)
    return cls(featureList, None)
  
  @classmethod
  def get_features_ms1ft(cls, feature_file):
    feature_info = pd.read_csv(feature_file, sep='\t')
    featureList = []
    for feature_idx in range(0, len(feature_info)):
      featurePd = feature_info.iloc[feature_idx]
      if featurePd["MinCharge"] > 30 and featurePd["MaxCharge"] > 30:
        continue
      multiChargeFeature = MultiChargeFeature(featurePd["FeatureID"], featurePd["MinScan"], featurePd["MaxScan"], 
                  featurePd["MinCharge"], featurePd["MaxCharge"], featurePd["MonoMass"], featurePd["RepCharge"], 
                  featurePd["RepMz"], featurePd["Abundance"], featurePd["MinElutionTime"], featurePd["MaxElutionTime"], 
                  -1, featurePd["ElutionLength"], featurePd["LikelihoodRatio"], -1, -1, -1)
      min_scan = multiChargeFeature.MinScan
      max_scan = multiChargeFeature.MaxScan
      apex_scan = int(featurePd["ApexScanNum"])
      rt_bin = (multiChargeFeature.MaxElutionTime - multiChargeFeature.MinElutionTime)/(max_scan - min_scan + 1)
      multiChargeFeature.ApexElutionTime = multiChargeFeature.MinElutionTime + ((apex_scan - min_scan + 1) * rt_bin)
      multiChargeFeature.ChargeRange = featurePd["MaxCharge"]-featurePd["MinCharge"]+1
      featureList.append(multiChargeFeature)
    return cls(featureList, None)
  
  @classmethod
  def get_features_xtract_xls(cls, feature_file):
    df = pd.read_excel(feature_file)
    featureList = []
    FeatureID = 0
    for i in range(0, len(df)):
      if not pd.isna(df.iloc[i]["Monoisotopic Mass"]):
        MinScan = int(df.iloc[i]["Scan Range"].split(' - ')[0])
        MaxScan = int(df.iloc[i]["Scan Range"].split(' - ')[1])
        MinCharge = int(df.iloc[i]["Charge State Distribution"].split(' - ')[0])
        MaxCharge = int(df.iloc[i]["Charge State Distribution"].split(' - ')[1])
        MonoMass = float(df.iloc[i]["Monoisotopic Mass"])
        RepCharge = -1
        RepMz = -1
        Abundance = float(df.iloc[i]["Sum Intensity"])
        MinElutionTime = float(df.iloc[i]["Start Time (min)"])
        MaxElutionTime = float(df.iloc[i]["Stop Time (min)"])
        ApexElutionTime = float(df.iloc[i]["Apex RT"])
        ElutionLength = MaxElutionTime - MinElutionTime
        Score = -1
        multiChargeFeature = MultiChargeFeature(FeatureID, MinScan, MaxScan, MinCharge, MaxCharge, MonoMass,
                      RepCharge, RepMz, Abundance, MinElutionTime, MaxElutionTime, ApexElutionTime, ElutionLength, Score, -1, -1, -1)
        multiChargeFeature.ChargeRange = int(df.iloc[i]["Number of Charge States"])
        featureList.append(multiChargeFeature)
        FeatureID = FeatureID + 1
    return cls(featureList, None)
  
  def __len__(self):
    return len(self.featureList)
  
  def __getitem__(self, index):
    return self.featureList[index]
  
  def __setitem__(self, index, newvalue):
    self.featureList[index] = newvalue
  
  @staticmethod
  def print_features(featureList, filename):
    for idx in range(0, len(featureList)):
      feature = featureList[idx]
      multiChargeFeature = feature.to_dict()
      df = pd.DataFrame(multiChargeFeature, index=[idx])
      if idx == 0:
        df.to_csv(filename, index = False, header=True)
      else:
        df.to_csv(filename, mode='a', index = False, header=False)
        
        
  @staticmethod
  def print_features_2(featureList, rep_num, output_dir):
    for idx in range(0, len(featureList)):
      feature = featureList[idx]
      multiChargeFeature = feature.to_dict()
      df = pd.DataFrame(multiChargeFeature, index=[idx])
      if idx == 0:
        df.to_csv(output_dir + os.sep + "rep_" + str(rep_num) + "_multiCharge_features.csv", index = False, header=True)
      else:
        df.to_csv(output_dir + os.sep + "rep_" + str(rep_num) + "_multiCharge_features.csv", mode='a', index = False, header=False)
        
