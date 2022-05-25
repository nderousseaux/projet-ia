import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from utils import *

class Neural_net():
    """Modèlise un réseau de neurone
    Possède :
        - La liste de tailles de ses différentes couches
        - Une fonction d'activation
        - Un taux d'apprentissage
        - Une patience
        - Des partitions et des labels d'entrainement et de validation
        - Des matrices d'activation (représente à quel point chaque neurone est activé)
        - Des matrices d'erreur (représente l'erreur propre à chaque neurone)
        - Des matrices de poids (représente le poids de la connection avec un neurone de la couche suivante)
        - Des matrices de biais (représente le biais de chaque neurone)
        - Des matrices de signal (représente le signal d'entrée de chaque neurone)
        - La liste des noms des classes (utile pour les prédictions)
    """
    
    def __init__(self, X, y, hidden_layer_sizes, activation, learning_rate=0.01, patience=3, train=True, seed=random_seed):
        """ Construit un réseau de neurone de dimensions indiquées
        """
        self.layer_sizes =  [X.shape[-1]] + hidden_layer_sizes + [y.shape[-1]] 
        self.activation = activation
        self.learning_rate = learning_rate
        self.columns_name = y.columns
        self.patience = patience

        #On met de coté le jeu de validation
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X, y, test_size=0.15, random_state=seed)

        #On initialise les matrices
        self.init_matrix()

        #On entraine le modèle
        if train:
            self.train()

    def init_matrix(self):
        """Initialise les matrices du réseau de neurone
        """
        self.A = [None] * (self.nb_layers()-1)# Matrices d'activation (Une pour chaque couche sauf la première)
        self.df = [None] * (self.nb_layers()-1) # Matrices d'erreur (Une pour chaque couche sauf la première)
        self.Z = [None] * (self.nb_layers()-1) #Matrices de signal d'entrée de chaque neurone (Une pour chaque couche sauf la première)
        self.W = [] #Matrices de poids (une pour chaque couche, sauf la dernière)
        self.B = [] #Matrices de biais (une pour chaque couche, sauf la première)

        for i in range(0,self.nb_layers()-1):
            #On initialise W : (x:Neurones couche suivante, y:neurones couche courrante)
            self.W.append(
                np.random.uniform(
                    low=-1.0, high=1.0,
                    size=(
                        self.layer_sizes[i+1],
                        self.layer_sizes[i]
                    )
                )
            )
            #On initialise B (x:Neurones de la couche courrante)
            self.B.append(
                    np.zeros(self.layer_sizes[i+1])
            )

    def train(self):
        """Entraine le modèle jusqu'au sur-apprentissage (early stopping)
        """
        best_erreur = 1
        cur_patience = 0

        #Malgré l'early stopping, on laisse une limite au nombre d'époques (la fonction d'erreur peut osciller)
        for i in range(200):
            #On réalise une époque d'entrainement
            self.one_epoch()

            #Une fois sur deux, on teste avec les données de validation
            if i%2 == 0:
                pred, erreur = self.compute_predictions(self.X_validate, self.y_validate, erreur=True)

                if erreur < best_erreur: #Si on trouve une meilleure erreur, on remet la patience à 0
                    best_erreur = erreur
                    cur_patience = 0
                elif cur_patience < self.patience: #Sinon on incrémente le nombre de passage sans amélioration
                    cur_patience +=1
                else: #Si la limite est dépassée, on arrête l'entrainement
                    break
                    
            #On mélange les données pour que la prochaine époque ne soit pas la même
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
            
    def one_epoch(self):
        """Réalise une époque d'entrainement (Propagation avant, rétro-propagation) sur les données en entrée
        Renvoie l'erreur moyenne
        """
        #Pour chaque donnée, on fait une passe-avant/passe-arrière
        for X_sample, y_sample in zip(self.X_train.values, self.y_train.values):
            self.prop_forward(X_sample, y_sample)
            self.prop_backward(X_sample,y_sample)

    def prop_forward(self, data, label):
        """Réalise une passe avant avec la donnée en paramètre
        Retourne l'erreur et l'activation de la dernière couche
        """
        #Au début, la première couche est activée par les entrées
        next_layer = data 
        
        # Pour chaque couche, sauf la dernière
        for i in range(self.nb_layers()-1):

            #On calcule le signal d'entrée de chaque neurone
            self.Z[i] = np.dot(self.W[i], next_layer) + self.B[i]

            #On calcule l'activation des neurone (et leur erreur au passage)
            if i != self.nb_layers()-2: #On active avec la fonction définie
                self.A[i],self.df[i] = self.activation(self.Z[i])
                next_layer = self.A[i]
            else: #Si c'est la dernière couche, on active avec softmax
                self.A[i], self.df[i] = self.softmax(self.Z[i])
                            
        return (
            self.cross_entropy_cost(self.A[-1],label),
            self.A[-1]
        )

    def prop_backward(self, data, label):
        """Réalise une passe arrière avec la donnée en paramètre
        """
        # On initialise les matrices d'erreur
        delta = [None] * (self.nb_layers()-1)
        dW = [None] * (self.nb_layers()-1)
        db = [None] * (self.nb_layers()-1)

        #On initialise la dernière couche avec les résultats du modèle
        delta[-1] = np.dot((self.A[-1] - label),self.df[-1])
        dW[-1] = np.transpose(delta[-1] * self.A[-2][...,None])
        db[-1] = delta[-1]

        #On recule dans les couches et on propage l'erreur
        for i in range(self.nb_layers()-3 , -1, -1):
            delta[i] = np.multiply(np.dot(self.W[i+1].T, delta[i+1]), self.df[i])
            if i == 0: #Si on est sur la première couche, on compare à l'erreur
                dW[i] = np.transpose(delta[i] * data[...,None])
            else:
                dW[i] = np.transpose(delta[i] * self.A[i-1][...,None])
            db[i] = delta[i]

        # On met à jour les poids et les biais
        for i in range(self.nb_layers()-1):
            self.W[i] = self.W[i] - self.learning_rate*dW[i]
            self.B[i] = self.B[i] - self.learning_rate*db[i]

    def nb_layers(self):
        """Renvoie le nombre de couches du réseau
        """
        return len(self.layer_sizes)

    def test_data(self, data, label, percentage=False):
        """Renvoie l'erreur et les prédictions
        Si percentage : Renvoie les probabilité d'appartenance à chaque classe de l'échantillon
        Sinon: Renvoie la classe la plus probable 
        """
        erreur, prediction = self.prop_forward(data, label)
        if percentage: #Renvoie la distribution de probabilité
            pred = pd.DataFrame(columns=self.columns_name)
            pred.loc[0, :] = prediction
        else: #Renvoie la classe la plus probable
            pred = self.columns_name[np.argmax(prediction)]

        return erreur, pred

    def compute_predictions(self, X, y, erreur=False):
        """ Calcule une liste de prédiction à partir des données en paramètre
        Peut renvoyer la moyenne de l'erreur
        """
        predictions = []
        erreurs = []
        for data, label in zip(X.values, y.values):
            err, pred = self.test_data(data, label)
            predictions.append(pred)
            erreurs.append(err)
            

        if erreur:
            return (
                pd.DataFrame(predictions),
                np.mean(erreurs)
            )
        else: 
            return pd.DataFrame(predictions)

    @staticmethod
    def tanh(Z):
        """Fonction d'activation tangeante
        Renvoie les entrées activées et la dérivée d'erreur
        """
        A = np.empty(Z.shape)
        A = 2.0/(1 + np.exp(-2.0*Z)) - 1 # A = np.tanh(Z)
        df = 1-A**2
        return A,df

    @staticmethod
    def relu(Z):
        """Fonction d'activation relu
        Renvoie les entrées activées et la dérivée d'erreur
        """
        A = np.empty(Z.shape)
        A = np.maximum(0,Z)
        df = (Z > 0).astype(int)
        return A,df
    
    @staticmethod
    def cross_entropy_cost(y_hat, y):
        """Calcule l'erreur associée à une prédiction
        """
        n  = y_hat.shape[0]
        ce = -np.sum(y*np.log(y_hat+1e-9))/n
        return ce   

    @staticmethod
    def softmax(z):
        """Calcule le softmax et le gradient d'erreur de softmax
        """
        shiftz = z - np.max(z)
        exps = np.exp(shiftz)
        softmax = exps / np.sum(exps)
        gradient = -np.outer(softmax, softmax) + np.diag(softmax.flatten())
        return softmax, gradient