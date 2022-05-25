from neural_net import *
from tree import *
from utils import *


def neural(X_train, X_test, y_train, y_test):
    """Crée quatres réseaux de neurones et enregistre leurs prédictions :
    - tanh: 10,8,6
    - tanh: 6,4
    - relu: 10,8,6
    - relu: 10,8,4
    """
    
    neural = Neural_net(X_train, y_train.iloc[:,1:], [10,8,4], Neural_net.tanh)
    predictions = neural.compute_predictions(X_test, y_test.iloc[:,1:]) 
    predictions.to_csv(folder_predic + "predict_neural_10_8_6_tanh.csv", index=False, header=False)

    neural = Neural_net(X_train, y_train.iloc[:,1:], [6,4], Neural_net.tanh)
    predictions = neural.compute_predictions(X_test, y_test.iloc[:,1:]) 
    predictions.to_csv(folder_predic + "predict_neural_6_4_tanh.csv", index=False, header=False)

    neural = Neural_net(X_train, y_train.iloc[:,1:], [10,8,6], Neural_net.relu)
    predictions = neural.compute_predictions(X_test, y_test.iloc[:,1:]) 
    predictions.to_csv(folder_predic + "predict_neural_10_8_6_relu.csv", index=False, header=False)

    neural = Neural_net(X_train, y_train.iloc[:,1:], [10,8,4], Neural_net.relu)
    predictions = neural.compute_predictions(X_test, y_test.iloc[:,1:]) 
    predictions.to_csv(folder_predic + "predict_neural_10_8_4_relu.csv", index=False, header=False)

def tree(X_train, X_test, y_train, y_test):
    """Crée deux arbres (de profondeur 5 et 7) et enregistre leurs prédictions
    """
    #Les attributs et les labels sont fusionnés pour l'ocasion.
    #Le contraire aurait entrainé trop de comparaison entre les deux dataframes
    #faisant drastiquement baisser la perfomance du programme.
    #On retire l'encodage onehot, inutile pour l'arbre
    df_train = pd.concat([X_train, y_train.iloc[:,0]], axis=1)
    df_test = pd.concat([X_test, y_test.iloc[:,0]], axis=1)

    #On enregistre les predictions des arbres de taille 5 et 7
    tree = Tree(df_train,0,5)
    predictions = tree.compute_predictions(df_test)
    predictions.to_csv(folder_predic + "predict_tree_5.csv", index=False, header=False)

    tree = Tree(df_train,0,7)
    predictions = tree.compute_predictions(df_test)
    predictions.to_csv(folder_predic + "predict_tree_7.csv", index=False, header=False)


if __name__ ==  "__main__":
    #On récupère les données
    X_train, X_test, y_train, y_test = prepare_data(file)

    #On enregistre le y_test
    y_test.iloc[:, 0].to_csv(folder_predic + "y_test.csv", index=False, header=False)

    tree(X_train, X_test, y_train, y_test)

    neural(X_train, X_test, y_train, y_test)
