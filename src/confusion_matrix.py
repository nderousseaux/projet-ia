import pandas as pd

class Confusion_matrix():
    """Modèlise une matrice de confusion
    Possède un tableau à deux dimensions (dataframe pandas) représentant la matrice de confusion.
    La première dimension représente le label prédi et la seconde représente le vrai label
    """

    def __init__(self,data, predictions):
        """Initialise la matrice de confusion à partir des vraies classes et des prédictions
        """
        self.data = data
        self.predictions = predictions

        classes = sorted(self.data.iloc[:,0].unique().tolist())
        self.matrix = pd.DataFrame(columns=classes, index=classes)

        #i : prédictions, j: vraies classes
        for i in classes:
            for j in classes:
                #On calcule le nombre de j qui à été attribué à i
                self.nb_attribuate(i,j)

        
    
    def nb_attribuate(self, i, j):
        """Calcule le nombre d'instance de la classe J, qui à été attribué à la classe I
        """
        res = 0
        for index, sample in self.data.iterrows():
            if sample[0] == j and self.predictions.loc[index][0] == i:
                res += 1
        self.matrix.loc[i,j] = res
                
    def print(self, space):
        """Affiche la matrice de confusion
        """
        print("{}X:Predictions / Y:True labels".format(space))

        print(self.matrix.rename(index=lambda n: "            " + str(n)))
        