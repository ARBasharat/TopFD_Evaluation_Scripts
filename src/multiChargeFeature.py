#!/usr/bin/python3

import statistics

class MultiChargeFeature():
  def __init__(self, FeatureID, MinScan, MaxScan, MinCharge, MaxCharge, MonoMass,
                RepCharge, RepMz, Abundance, MinElutionTime, MaxElutionTime, 
                ApexElutionTime, ElutionLength, EnvcnnScore, PercentMatchedPeaks, IntensityCorrelation, Top3Correlation):
    self.FeatureID = FeatureID
    self.MinScan = MinScan
    self.MaxScan = MaxScan
    self.MinCharge = MinCharge
    self.MaxCharge = MaxCharge
    self.MonoMass = MonoMass
    self.RepCharge = RepCharge
    self.RepMz = RepMz
    self.Abundance = Abundance
    self.MinElutionTime = MinElutionTime
    self.MaxElutionTime = MaxElutionTime
    self.ApexElutionTime = ApexElutionTime
    self.ElutionLength = ElutionLength
    self.EnvcnnScore = EnvcnnScore
    self.PercentMatchedPeaks = PercentMatchedPeaks
    self.IntensityCorrelation = IntensityCorrelation
    self.Top3Correlation = Top3Correlation
    self.Score = 0
    self.Label = 0
    
  def to_dict(self):
    return {
      'FeatureID': self.FeatureID,
      'MinScan': self.MinScan,
      'MaxScan': self.MaxScan,
      'MinCharge': self.MinCharge,
      'MaxCharge': self.MaxCharge,
      'MonoMass': self.MonoMass,
      'RepCharge': self.RepCharge,
      'RepMz': self.RepMz,
      'Abundance': self.Abundance,
      'MinElutionTime': self.MinElutionTime,
      'MaxElutionTime': self.MaxElutionTime,
      'ApexElutionTime': self.ApexElutionTime,
      'ElutionLength': self.ElutionLength,
      'EnvCNNScore': self.EnvcnnScore,
      'PercentMatchedPeaks': self.PercentMatchedPeaks,
      'IntensityCorrelation' : self.IntensityCorrelation,
      'Top3Correlation' : self.Top3Correlation,
      'Score': self.Score,
      'Label': self.Label
  }
  
  @staticmethod
  def get_multiCharge_feature_dict(multicharge_feature_list, idx):
    FeatureID = idx
    MinScan = min([j.start_scan for j in multicharge_feature_list])
    MaxScan = max([j.end_scan for j in multicharge_feature_list])
    MinCharge = min([j.charge for j in multicharge_feature_list])
    MaxCharge =  max([j.charge for j in multicharge_feature_list])
    MonoMass = min([j.prec_mass for j in multicharge_feature_list])
    intens = [sum(j.envelope_xic) for j in multicharge_feature_list]
    RepFeature = intens.index(max(intens))
    RepCharge = multicharge_feature_list[RepFeature].charge
    RepMz = multicharge_feature_list[RepFeature].mz
    Abundance = sum([j.intensity for j in multicharge_feature_list])
    score = max([j.score for j in multicharge_feature_list])
    MinElutionTime = min([j.start_rt for j in multicharge_feature_list])
    MaxElutionTime = max([j.end_rt for j in multicharge_feature_list])
    ApexElutionTime = min([j.apex_rt for j in multicharge_feature_list])
    ## Get mean of all charge states!
    PercentMatchedPeaks = statistics.mean([j.percent_matched_peaks for j in multicharge_feature_list])
    IntensityCorrelation = statistics.mean([j.intensity_correlation for j in multicharge_feature_list])
    Top3Correlation = statistics.mean([j.top_3_scan_corr for j in multicharge_feature_list])
    ElutionLength = MaxElutionTime - MinElutionTime
    feature = MultiChargeFeature(FeatureID, MinScan, MaxScan, MinCharge, MaxCharge, MonoMass,
               RepCharge, RepMz, Abundance, MinElutionTime, MaxElutionTime, ApexElutionTime, 
               ElutionLength, score, PercentMatchedPeaks, IntensityCorrelation, Top3Correlation)
    return feature.to_dict()
