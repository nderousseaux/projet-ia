from unittest import TestCase
import numpy as np

from utils import *
from neural_net import *


class Test_neural_net(TestCase):
    """Classe de tests de certaines fonctionnalités du réseau de neurone
    - Passe avant avec une instance
    - Rétropropagation et mise à jour après une passe avant
    Attention : Les résultats ne sont corrects qu'avec une random seed de 10
    """
    random_seed=10
    X = np.full(14, 0.5)
    y = np.array([0,0,0,1])
    expected_err = 0.29258652199741814
    expected_pred = [
        0.209256212599809,
        0.34038540633210523,
        0.1400988364881137,
        0.310259544579972
    ]
    expected_b0 = [
        -0.00037535049718140373,
        0.00030200819270271884,
        2.8148838973498085e-05,
        0.0018372780160471024,
        -0.0003758681377821066,
        -0.0010949920810086793
    ]
    expected_w00 = [
        0.5424536112849012,
         -0.9586837765297878,
         0.26710879460396014,
         0.497420089828633,
         -0.0031736506434098533,
         -0.5505943841868954,
         -0.6040619457293428,
         0.5208737491493267,
         -0.6619660021235199,
         -0.8235080469005702,
         0.3705319614870038,
         0.9065990171412823,
         -0.9922911425927619,
         0.024196851522962526
    ]

    def init(self):
        """Initialise le réseau de neurone et réinitialise l'aléatoire
        """
        np.random.seed(seed=self.random_seed)

        #On récupère les données
        X_train, X_test, y_train, y_test = prepare_data(file, seed=self.random_seed)

        #On initialise un réseau de neurone, sans l'entrainer
        self.neural = Neural_net(X_train, y_train.iloc[:,1:], [6,4], Neural_net.tanh, train=False, seed=self.random_seed)

    def test_prop_forward(self):
        """Teste la passe avant d'une instance
        Avec une valeur X,y, teste si les valeurs de prédiction et d'erreur sont cohérentes
        """
        self.init()
        
        err, pred = self.neural.prop_forward(self.X, self.y)

        self.assertEqual(err, self.expected_err)
        self.assertTrue(np.array_equal(pred, self.expected_pred))

    def test_prop_backward(self):
        """Teste une passe avant suivi d'une passe arrière
        Teste les valeurs des matrices de poids et de biais de rang 0
        """
        self.init()
        self.neural.prop_forward(self.X, self.y)
        self.neural.prop_backward(self.X,self.y)


        self.assertTrue(np.array_equal(self.neural.B[0], self.expected_b0))
        self.assertTrue(np.array_equal(self.neural.W[0][0], self.expected_w00))