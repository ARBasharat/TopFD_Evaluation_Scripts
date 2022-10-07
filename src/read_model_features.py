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
import pandas as pd

class MultiChargeFeature():
  def __init__(self, FeatureID, MinScan, MaxScan, MinCharge, MaxCharge, MonoMass,
                Abundance, MinElutionTime, MaxElutionTime, ApexElutionTime, ElutionLength, EnvcnnScore, Score, Label):
    self.FeatureID = FeatureID
    self.MinScan = MinScan
    self.MaxScan = MaxScan
    self.MinCharge = MinCharge
    self.MaxCharge = MaxCharge
    self.MonoMass = MonoMass
    self.Abundance = Abundance
    self.MinElutionTime = MinElutionTime
    self.MaxElutionTime = MaxElutionTime
    self.ApexElutionTime = ApexElutionTime
    self.ElutionLength = ElutionLength
    self.EnvcnnScore = EnvcnnScore
    self.Score = Score
    self.Label = Label

def get_features(feature_file):
  # feature_info = pd.read_csv(feature_file, sep=',')
  feature_info = pd.read_csv(feature_file, sep='\t|,')
  featureList = []
  for feature_idx in range(0, len(feature_info)):
    featurePd = feature_info.iloc[feature_idx]
    multiChargeFeature = MultiChargeFeature(int(featurePd["FeatureID"]), 
                                            int(featurePd["MinScan"]), 
                                            int(featurePd["MaxScan"]),
                                            int(featurePd["MinCharge"]),
                                            int(featurePd["MaxCharge"]),
                                            float(featurePd["MonoMass"]), 
                                            float(featurePd["Abundance"]),
                                            float(featurePd["MinElutionTime"]),
                                            float(featurePd["MaxElutionTime"]),
                                            float(featurePd["ApexElutionTime"]),
                                            float(featurePd["ElutionLength"]),
                                            float(featurePd["EnvCNNScore"]),
                                            float(featurePd["Score"]),
                                            int(featurePd["Label"]))
    
    multiChargeFeature.rt_range = float(featurePd["MaxElutionTime"]) - float(featurePd["MinElutionTime"])
    multiChargeFeature.charge_range = int(featurePd["MaxCharge"]) - int(featurePd["MinCharge"])
    
    if "PercentMatchedPeaks" in featurePd:
      multiChargeFeature.PercentMatchedPeaks = float(featurePd["PercentMatchedPeaks"])
    else:
      multiChargeFeature.PercentMatchedPeaks = -1
      
    if "IntensityCorrelation" in featurePd:
      multiChargeFeature.IntensityCorrelation = float(featurePd["IntensityCorrelation"])
    else:
      multiChargeFeature.IntensityCorrelation = -1
      
    if "Top3Correlation" in featurePd:
      multiChargeFeature.Top3Correlation = float(featurePd["Top3Correlation"])
    else:
      multiChargeFeature.Top3Correlation = -1
      
    if "EvenOddPeakRatios" in featurePd:
      multiChargeFeature.EvenOddPeakRatios = float(featurePd["EvenOddPeakRatios"])
    else:
      multiChargeFeature.EvenOddPeakRatios = -1
      
    if "PercentConsecPeaks" in featurePd:
      multiChargeFeature.PercentConsecPeaks = float(featurePd["PercentConsecPeaks"])
    else:
      multiChargeFeature.PercentConsecPeaks = -1
    
    if "MaximaNumber" in featurePd:
      multiChargeFeature.MaximaNumber = int(featurePd["MaximaNumber"])
    else:
      multiChargeFeature.MaximaNumber = -1
      
    if "IntensityCosineSimilarity" in featurePd:
      multiChargeFeature.IntensityCosineSimilarity = float(featurePd["IntensityCosineSimilarity"])
    else:
      multiChargeFeature.IntensityCosineSimilarity = -1
      
    if "NumTheoPeaks" in featurePd:
      multiChargeFeature.NumTheoPeaks = float(featurePd["NumTheoPeaks"])
    else:
      multiChargeFeature.NumTheoPeaks = -1
        
    if "MzErrorSum" in featurePd:
      multiChargeFeature.MzErrorSum = float(featurePd["MzErrorSum"])
    else:
      multiChargeFeature.MzErrorSum = -1
    
    if "MzErrorSumBase" in featurePd:
      multiChargeFeature.MzErrorSumBase = float(featurePd["MzErrorSumBase"])
    else:
      multiChargeFeature.MzErrorSumBase = -1     
      
    if "RepCharge" in featurePd:
      multiChargeFeature.RepCharge = float(featurePd["RepCharge"])
    else:
      multiChargeFeature.RepCharge = -1   
      
    if "RepMz" in featurePd:
      multiChargeFeature.RepMz = float(featurePd["RepMz"])
    else:
      multiChargeFeature.RepMz = -1   
    
    featureList.append(multiChargeFeature)
  return featureList

