# TopFD_Evaluation_Scripts 

You can downaload the data using: https://www.toppic.org/software/toppic/topfd_supplemental.html


## Step 1: Label the proteoform features for training the model.
Label the SW480 and BC data to prepare them for training the ECScore model. 
```
pythton3 01_label_training_data.py -f TopFD_Published_Data\03_Extracted_Features\03_SW480_Colorectal_Cancer\TopFD
pythton3 01_label_training_data.py -f TopFD_Published_Data\03_Extracted_Features\04_WHIM2_Breast_Cancer\TopFD
pythton3 01_label_training_data.py -f TopFD_Published_Data\03_Extracted_Features\05_WHIM16_Breast_Cancer\TopFD
```

## Step 2: Merge the training data
> Copy the labeled feature files (*03_SW480_Colorectal_Cancer_rep_1_labeled.csv, 04_WHIM2_Breast_Cancer_rep_1_labeled.csv, and 05_WHIM16_Breast_Cancer_rep_1_labeled.csv*) to a new directory (*Example: TopFD_Published_Data\04_Training_Data_ECScore_Model\Labeled_Features*). 
```
pythton3 02_merge_training_data_files.py -f TopFD_Published_Data\04_Training_Data_ECScore_Model\Labeled_Features
```
This will generate the training data file **00_train_data.csv**.

## Step 3: Train the model
Train the ECScore model
```
pythton3 03_train_model.py -f TopFD_Published_Data\04_Training_Data_ECScore_Model\00_train_data.csv
```

## Step 4: Get model performance
The script will take feature directory as input and will determine ECScore model's performance
```
python3 04_get_model_performance.py -f TopFD_Published_Data\03_Extracted_Features\01_Ovarian_Cancer\TopFD -m TopFD_Published_Data\04_Training_Data_ECScore_Model\model\ecscore.h5
python3 04_get_model_performance.py -f TopFD_Published_Data\03_Extracted_Features\02_SW620_Colorectal_Cancer\TopFD -m TopFD_Published_Data\04_Training_Data_ECScore_Model\model\ecscore.h5
```

## Step 5: Prepare test data - Remove Artifacts
Remove mass artifacts from the OC and SW620 data sets for TopFD, ProMex, FlashDeconv and Xtract
```
python3 05_remove_artifacts.py -f TopFD_Published_Data\03_Extracted_Features\01_Ovarian_Cancer -m TopFD_Published_Data\04_Training_Data_ECScore_Model\model\ecscore.h5
python3 05_remove_artifacts.py -f TopFD_Published_Data\03_Extracted_Features\02_SW620_Colorectal_Cancer -m TopFD_Published_Data\04_Training_Data_ECScore_Model\model\ecscore.h5
```
This script will create a new directory containing valid feature files. 

## Step 6: Compare total ion current and Feature Intensity
The script will take valid features as input and will compare the performance of all tools.
```
python3 06_compare_tic.py -f TopFD_Published_Data\05_Post_Artifacts_Removal\01_Ovarian_Cancer -m TopFD_Published_Data\02_mzML_Files\01_Ovarian_Cancer\CPTAC_Intact_rep1_15Jan15_Bane_C2-14-08-02RZ.mzML
python3 06_compare_tic.py -f TopFD_Published_Data\05_Post_Artifacts_Removal\02_SW620_Colorectal_Cancer -m TopFD_Published_Data\02_mzML_Files\02_SW620_Colorecatal_Cancer\20220207_UreaExtracted_SW620_C4_RPLC_01.mzML
```

## Step 7: Get feature and intensity reproducability
The script will take valid features as input and will compare the performance of all tools for feature and quantitaive reproducability.
```
python3 07_get_reproducability.py -f TopFD_Published_Data\05_Post_Artifacts_Removal\01_Ovarian_Cancer -m TopFD_Published_Data\02_mzML_Files\01_Ovarian_Cancer\CPTAC_Intact_rep1_15Jan15_Bane_C2-14-08-02RZ.mzML
python3 07_get_reproducability.py -f TopFD_Published_Data\05_Post_Artifacts_Removal\02_SW620_Colorectal_Cancer -m TopFD_Published_Data\02_mzML_Files\02_SW620_Colorecatal_Cancer\20220207_UreaExtracted_SW620_C4_RPLC_01.mzML
```
This will output occourance of feature in the replicates and will generate pairwise log-abundance correlation plot.

