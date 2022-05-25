import numpy as np

from utils import *

class Tree():
    """ Modèlise un arbre de décision binaire
    Possède :
        - Une partition de donnée
        - Une profondeur
        - Une attribut
        - Une valeur de split pour cet attribut
        - Le gain d'information acqui avec ce split
        - Les partitions déduites de ce split
        - Une prédiction (combien d'individu de chaque classe est arrivé dans ce noeud)
        - Une branche de droite et de gauche (si n'est pas une feuille)
    """
     
    def __init__(self, data, deep=0, max_deep=8):
        """ Construit un arbre de décision binaire de profondeur indiqué
        """
        self.gain = 0
        self.data = data
        
        #On détermine la meilleure règle du noeud, et ses partitions associés
        self.determine_rules()

        #Si on n'est pas au fond de l'arbre
        if deep <= max_deep and self.gain != 0:
            self.left_node = Tree(self.partitions[0], deep+1, max_deep)
            self.right_node = Tree(self.partitions[1], deep+1, max_deep)

    def determine_rules(self):
        """ Détermine la meilleure règle du jeu de donnée (attribut, et valeur de coupe)
        """
        #Pour chaque attribut
        for attribut_l in self.data.iloc[:, :-1]:
            #On calcule la meilleure valeur de split pour cet attribut
            gain_l, split_value_l, partitions_l = self.determine_split(attribut_l)
            
            # Et on garde le meilleur attribut
            if gain_l >= self.gain:
                self.attribute = attribut_l
                self.gain = gain_l
                self.split_value = split_value_l
                self.partitions = partitions_l

    def determine_split(self, attribute):
        """ Détermine la meilleure valeur de split pour un attribut donné
            Renvoie les partitions associée à cette coupe
        """
        gain = 0
        partitions = [None, None]

        #Pour chaque quartile, on calcule le gain d'information
        for quartile in np.arange(0.25, 1, 0.25):
            split_value_l = self.data.quantile(quartile)[attribute]

            p1 = self.data.loc[self.data[attribute] < split_value_l]
            p2 = self.data.loc[self.data[attribute] >= split_value_l]

            #Calcul du gain inhérent à ce split 
            gain_l = Tree.determine_gain([p1,p2])

            # On garde la meilleure valeur de split
            if gain_l >= gain:
                gain = gain_l
                split_value = split_value_l
                partitions[0] = p1
                partitions[1] = p2

        return gain, split_value, partitions

    def prediction(self, percentage=False):
        """ Renvoie la prédiction du noeud
        """
        return self.data["Class"].value_counts(normalize=percentage)

    def is_leaf(self):
        """ Vrai si l'arbre est une feuille
        """
        if not hasattr(self, "right_node") and not hasattr(self, 'left_node'):
            return True
        return False

    def test_data(self, sample, percentage=False):
        """Si percentage : Renvoie les probabilité d'appartenance à chaque classe de l'échantillon
        Sinon: Renvoie la classe la plus probable 
        """

        if self.is_leaf():
            if percentage: #Renvoie la distribution de probabilité
                return self.prediction(True)
            else: #Renvoie la classe la plus probable
                return self.prediction().sort_values(ascending=False).index[0]

        else:
            if sample[self.attribute] < self.split_value:
                return self.left_node.test_data(sample, percentage=percentage)
            else:
                return self.right_node.test_data(sample, percentage=percentage)

    def compute_predictions(self, test_data):
        """ Calcule une liste de prédiction à partir de données de test
        """

        return test_data.apply(self.test_data, axis=1)

    def compute_performance(self, test_data):
        """ Retourne le pourcentage de données correctement classées
        """
        nb_match = 0
        predictions = self.compute_predictions(test_data)

        for index, sample in test_data.iterrows():
            if sample["Class"] == predictions.loc[index]:
                nb_match +=1

        return nb_match/len(test_data)

    @staticmethod
    def determine_gain(partitions):
        """Calcule le gain résultant de la découpe des partions
        """

        full_df = pd.concat([partitions[1],partitions[0]])

        res = Tree.entropy(full_df)
        for partition in partitions:
            res -= (len(partition)/len(full_df)) * Tree.entropy(partition)
        return res

    @staticmethod
    def entropy(data):
        """Calcule l'entropie résultante d'un jeu de donnée
        """
        res = 0

        for nb in data["Class"].value_counts():
            proba = nb/len(data)
            res += proba * math.log(proba, 2)
        
        return -res
