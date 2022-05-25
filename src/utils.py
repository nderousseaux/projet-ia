from sklearn.model_selection import train_test_split
import math
import numpy as np
import pandas as pd

folder = "src/csv/"
folder_predic = folder + "predictions/"
file = folder + "data.csv"
random_seed = 42
np.random.seed(seed=random_seed)

def prepare_data(file, seed=random_seed):
    """Retourne les données d'entrainement et de test.
    """

    data = pd.read_csv(file)

    #On sépare les attributs des labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    #On normalise les données
    for c in X.columns:
        X[c] = X[c]  / X[c].max()
    
    
    #On sépare les données d'entrainement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    #On ajoute le one_hot
    y_train = pd.concat([y_train, pd.get_dummies(y_train)], axis=1)
    y_test = pd.concat([y_test, pd.get_dummies(y_test)], axis=1)
    
    return X_train, X_test, y_train, y_test

def nb_true_positive(data, predictions, classe):
    """Calcule le nombre de vrai positifs parmis les prédictions pour une classe précise
    """
    res = 0
    for index, sample in data.iterrows():
        if sample[0] == classe and predictions.loc[index][0] == classe:
            res += 1
    return res

def nb_false_positive(data, predictions, classe):
    """Calcule le nombre de faux positifs parmis les prédictions pour une classe précise
    """
    res = 0
    for index, sample in data.iterrows():
        if sample[0] != classe and predictions.loc[index][0] == classe:
            res += 1
    return res

def nb_true_negative(data, predictions, classe):
    """Calcule le nombre de vrai négatifs parmis les prédictions pour une classe précise
    """
    res = 0
    for index, sample in data.iterrows():
        if sample[0] != classe and predictions.loc[index][0] != classe:
            res += 1
    return res

def nb_false_negative(data, predictions, classe):
    """Calcule le nombre de faux négatifs parmis les prédictions pour une classe précise
    """
    res = 0
    for index, sample in data.iterrows():
        if sample[0] == classe and predictions.loc[index][0] != classe:
            res += 1
    return res

def accuracy(data, predictions, classe):
    """Calcule l'exactitude de prediction, pour une classe précise
    """
    #(Vrai positifs + Vrai négatifs) / Total
    return (nb_true_positive(data, predictions, classe)+nb_true_negative(data, predictions, classe)) / len(data)

def precision(data, predictions, classe):
    """Calcule la précision de prediction, pour une classe précise
    """
    true_positive = nb_true_positive(data, predictions, classe)
    
    #Vrai positifs / (Vrai positifs + faux positif)
    return true_positive/(true_positive + nb_false_positive(data, predictions, classe))

def recall(data, predictions, classe):
    """Calcule le recall de prediction, pour une classe précise
    """
    true_positive = nb_true_positive(data, predictions, classe)

    #Vrai positifs / (Vrai positifs + faux négatifs)
    return true_positive / (true_positive + nb_false_negative(data, predictions, classe))

def f1_score(data, predictions, classe):
    """Calcule le f1-score de prediction, pour une classe précise
    """
    recall_l = recall(data, predictions, classe)
    precision_l = precision(data, predictions, classe)

    #2 * ((recall * precision) / (recall + precision))
    return 2 * (
        (recall_l * precision_l) /
        (recall_l + precision_l)
    )