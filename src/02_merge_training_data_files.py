#!/usr/bin/python3

import os 
import argparse  
from copy import deepcopy
from collections import Counter
from matplotlib import pyplot as plt
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

if __name__ == "__main__":
  print("Merge Feature files!!!!")
  parser = argparse.ArgumentParser(description='Merge feature files to train model.')
  parser.add_argument("-f", "--feature_dir", default = r"E:\TopFD_Published_Data\04_Training_Data_ECScore_Model\Labeled_Features", help="Directory containing the feature files")
  args = parser.parse_args()
  
  feature_files = [f for f in os.listdir(args.feature_dir) if f.endswith(r".feature") or f.endswith(r".csv")]
  
  multicharge_features_replicate = []
  for file in feature_files:
    print("Processing File:", file)
    multicharge_features = get_features(os.path.join(args.feature_dir, file))
    multicharge_features_replicate.append(multicharge_features)
    
  features = []
  for idx in range(0, len(multicharge_features_replicate)):
    features.extend(multicharge_features_replicate[idx])
    
  for idx in range(0, len(features)):
    features[idx].FeatureID = idx
    
  output_env_coll_list(os.path.join(os.path.dirname(args.feature_dir), "00_train_data.csv"), features)
