import pandas as pd
import numpy as np

# # # # # # # CALCULATE THE ACCURACY OF PREDICTION # # # # # # # #

# IMPORTING THE PREDICTION RESULT FROM EXCEL
Well_Pred_Pos2 = pd.read_excel('New_Poseidon_2.xlsx')
Well_Pred_Pos2 = Well_Pred_Pos2.drop(['DEPTH','RS','RD','DTSM','DTCO','DCAV'],axis=1)

# REMOVE ROWS CONTAINING -999.25
Well_Pred_Pos2 = Well_Pred_Pos2.drop(np.arange(9615,9722,1)) #remove rows with no specified target
Well_Pred_Pos2 = Well_Pred_Pos2[Well_Pred_Pos2.GR != -999.25]
Well_Pred_Pos2 = Well_Pred_Pos2[Well_Pred_Pos2.TNP != -999.25]
Well_Pred_Pos2 = Well_Pred_Pos2[Well_Pred_Pos2.RHOB != -999.25]

# # # CHANGE CHERT TO Z-CHERT, ETC --> CHANGE ITS CATEGORY CODES TO THE LAST NUMBER SO THAT THE ACCURACY CALCULATION VALID
# # # THE LITHOLOGY THAT ITS NAME SHOULD BE CHANGED IS THE ONE THAT DOESN'T HAVE A PAIR IN PREDICTION RESULT OR ORIGINAL RESULT

facies = Well_Pred_Pos2["Target Original 5"]
facies = facies.replace('Calcilutite', 'Z-Calcilutite')
# facies = facies.replace('Chert', 'Z-Chert')
# facies = facies.replace('Calcarenite', 'Z-Calcarenite')

facies_pred = Well_Pred_Pos2["Train 9-5-10"]
facies_pred = facies_pred.replace('Calcilutite', 'Z-Calcilutite')

# FACIES FREQUENCY OF BOTH ACTUAL DATA AND PREDICTED DATA
Facies_Freq1 = np.unique(facies, return_counts=True)
# print(Facies_Freq1)
Facies_Freq2 = np.unique(facies_pred, return_counts=True)
# print(Facies_Freq2)

facies = facies.astype('category')
F1 = facies.cat.codes
facies_pred = facies_pred.astype('category')
F2 = facies_pred.cat.codes

# # # # # # # # # # # # # # COUNTING ACCURACY # # # # # # # # # # #
from sklearn.metrics import accuracy_score
Acc = accuracy_score(F1,F2)
# print(Acc)

# # # # # # ACCURACY OF MODE OF 10 LOOP PREDICTION # # # # # # #

from scipy import stats as s

Well_Pred_Posei2 = pd.read_excel('New_Poseidon_2.xlsx')
Well_Pred_Posei2 = Well_Pred_Posei2.drop(['RS','RD','DTSM','DTCO','DCAV'],axis=1)

# REMOVE ROWS CONTAINING -999.25
Well_Pred_Posei2 = Well_Pred_Posei2.drop(np.arange(9615,9722,1)) #remove rows with no specified target
Well_Pred_Posei2 = Well_Pred_Posei2[Well_Pred_Posei2.GR != -999.25]
Well_Pred_Posei2 = Well_Pred_Posei2[Well_Pred_Posei2.TNP != -999.25]
Well_Pred_Posei2 = Well_Pred_Posei2[Well_Pred_Posei2.RHOB != -999.25]

Dataset = Well_Pred_Posei2.drop(['TNP','GR','RHOB','DEPTH'],axis=1)
Dataset2 = Dataset.iloc[:,np.arange(11,21,1)]
# print(Dataset2)
Dataset2 = np.asarray(Dataset2, dtype=list)

l = len(Dataset2)

Facies = []

for i in range(l):
    Final_Facies = s.mode(Dataset2[i,:])
    Final_Facies = np.array(Final_Facies)
    Final_Facies = Final_Facies[0,:]
    Facies.append(Final_Facies)

Facies = np.hstack(Facies).reshape(2423,1)
Dataset = Dataset.reset_index()
Dataset[["Final Facies"]] = pd.DataFrame(Facies)

# # # CHANGE CHERT TO Z-CHERT, ETC --> CHANGE ITS CATEGORY CODES TO THE LAST NUMBER SO THAT THE ACCURACY CALCULATION VALID
# # # THE LITHOLOGY THAT ITS NAME SHOULD BE CHANGED IS THE ONE THAT DOESN'T HAVE A PAIR IN PREDICTION RESULT OR ORIGINAL RESULT
facies_ = Dataset["Target Original 5"]
# facies_ = facies_.replace('Chert', 'Z-Chert')
# facies_ = facies_.replace('Calcarenite', 'Z-Calcarenite')

facies_pred = Dataset["Final Facies"]
facies_pred = facies_pred.replace('Calcilutite','Z-Calcilutite')

# FACIES FREQUENCY OF BOTH ACTUAL DATA AND PREDICTED DATA
Facies_Freq3 = np.unique(facies_, return_counts=True)
print(Facies_Freq3)
Facies_Freq4 = np.unique(facies_pred, return_counts=True)
print(Facies_Freq4)

facies__ = facies_.astype('category')
F3 = facies__.cat.codes
facies__pred = facies_pred.astype('category')
F4 = facies__pred.cat.codes

# # # # # # # # # # COUNTING ACCURACY # # # # # # # # # # #
Acc = accuracy_score(F3,F4)
print(Acc)