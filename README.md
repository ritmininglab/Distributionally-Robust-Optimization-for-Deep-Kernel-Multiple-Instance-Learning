# Distributionally-Robust-Optimization-for-Deep-Kernel-Multiple-Instance-Learning
This is the code for the paper "Distributionally Robust Optimization for Deep Kernel Multiple Instance Learning". 
Processed dataset required for the training will be provided separately in the google-drive because of the size issue.
We have evaluation on five different datasets in the paper: (1) SanghaiTech, (2) UCF-Crime, (3) Avenue, (4) SanghaiTech Outlier, and (5) UCF-Crime Multimoal.

## SanghaiTech Dataset:
To train the model for SanghaiTech Dataset execute the following command:
python train_mil_sanghaitech.py split_no rep_no eta

Trains the model and Saves (1) all losses, (2) testing AUCS,  and (3) validation  AUCS under the directory logs/SanghaiTech in each 10 number of iterations. Also, stores the resulting best model under the directory trained_models/SanghaiTech

## UCF-Crime Dataset:
To train the model for UCF-Crime Dataset, execute the following command:

python train_mil_ucfcrime.py rep_no eta

Trains the model and Saves (1) all losses, and (2) testing AUCS under the directory logs/UCF_Crime in each 10 number of iterations. Also, stores the resulting best model under the directory trained_models/UCF_Crime

## Avenue Dataset:
To train the model for Avenue Dataset, execute the following command:

python train_mil_avenue.py cv_no rep_no eta


Trains the model and Saves (1) total loss, and (2) testing AUCS under the directory logs/Avenue in each 10 number of iterations. Also, stores the resulting best model under the directory trained_models/Avenue


## SanghaiTech Outlier Dataset:
To train the model for SanghaiTech Outlier Dataset, execute the following command:

python train_mil_sanghaitech_outlier.py split_no rep_no eta

Trains the model and Saves (1) all losses, (2) testing AUCS,  and (3) validation  AUCS under the directory logs/SanghaiTech_Outlier in each 10 number of iterations. Also, stores the resulting best model under the directory trained_models/SanghaiTech_Outlier

## UCF-Crime Multimodal Dataset:
To train the model for UCF-Crime Dataset, execute the following command:

python train_mil_ucfcrime_multimodal.py rep_no eta

Trains the model and Saves (1) total loss, and (2) testing AUCS under the directory logs/UCF_Crime_Multimodal in each 10 number of iterations. Also, stores the resulting best model under the directory trained_models/UCF_Crime_Multimodal


Where; 
#### (1) rep_no: Replication number we considered during training. If we want to run the model for different random initialization, we can call with different rep_no
#### (2) eta: Hyperparameter considered in the training for DRO framework (10^-8-1.0)
#### (3) split_no: validation-test pair for sanghaitech and sanghaitech outlier dataset (1, 20)
####  (4) cv_no: cross validation number considered in the Avenue dataset. We have 5 pairs of training-testing set (1-5)
