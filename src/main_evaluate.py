from os import listdir

from confusion_matrix import *
from tree import *
from utils import *

if __name__ ==  "__main__":
    
    #On importe y_test
    y_test = pd.read_csv(folder_predic + "y_test.csv", header=None)

    #On importe chaque prédictions, et on calcule les métriques et la matrice de confusion
    for file in [f for f in listdir(folder_predic) if f != "y_test.csv"]:
        print("------ Predictions associés au fichier {} ------".format(file))

        predictions = pd.read_csv(folder_predic + file, header=None)

        #Pour chaque classe, on affiche les métriques
        for classe in sorted(y_test.iloc[:,0].unique().tolist()):
            print("\nClasse {} :".format(classe))
            print("\tExactitude : \t{}".format(accuracy(y_test, predictions, classe)))
            print("\tPrécision : \t{}".format(precision(y_test, predictions, classe)))
            print("\tRappel : \t{}".format(recall(y_test, predictions, classe)))
            print("\tF1_Score : \t{}".format(f1_score(y_test, predictions, classe)))
        
        print("\nMatrice de confusion :")
        Confusion_matrix(y_test, predictions).print("\t")

        print("\n\n")