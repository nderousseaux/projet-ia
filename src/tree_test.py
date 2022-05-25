from unittest import TestCase
import pandas as pd

from utils import *
from tree import *


class Test_tree(TestCase):
    """Classe de tests de certaines fonctionnalités de l'arbre de décision
    - Calcul de l'entropie d'une partition
    - Calcul du gain d'un partitionnement
    - Détermination d'un meilleur partitionnement
    """

    partitionA = pd.DataFrame(data={
        'col1': [1, 2, 4, 2, 1], 
        'col2': [3, 4, 1, 3, 2],
        'Class': [1, 2, 1, 3, 1]
    })
    partitionB = pd.DataFrame(data={
        'col1': [2, 3, 1, 3, 4], 
        'col2': [1, 1, 2, 1, 2],
        'Class': [3, 3, 1, 1, 2]
    })
    expected_entropy_A = 1.3709505944546687
    expected_entropy_B = 1.5219280948873621
    expected_gain = 0.039035952556318976
    expected_split_col1_parA = [[1, 3, 1], [1, 2, 1], [1, 2, 1]]
    expected_split_col1_parB = [[2, 4, 2], [4, 1, 1], [2, 3, 3], [2, 1, 3], [3, 1, 3], [3, 1, 1], [4, 2, 2]]
    expected_split_col2_parA = [[4, 1, 1], [2, 1, 3], [3, 1, 3], [3, 1, 1]]
    expected_split_col2_parB = [[1, 3, 1], [2, 4, 2], [2, 3, 3], [1, 2, 1], [1, 2, 1], [4, 2, 2]]

    def test_entropy(self):
        """Teste l'entropie résultante d'une partition
        """
        entropyA = Tree.entropy(self.partitionA)
        entropyB = Tree.entropy(self.partitionB)
        self.assertEqual(entropyA, self.expected_entropy_A)
        self.assertEqual(entropyB, self.expected_entropy_B)


    def test_gain(self):
        """Teste le gain d'information résultant d'un partitionnement
        """
        gain = Tree.determine_gain([self.partitionA, self.partitionB])
        self.assertEqual(gain, self.expected_gain)

    def test_partitionnement(self):
        """Teste le calcul du meilleur partitionnement pour un attribut donné
        """

        #On crée l'arbre
        tree = Tree(pd.concat([self.partitionA, self.partitionB]), 0, 1)

        gain, split_value, partitions = tree.determine_split("col1")
        self.assertEqual(partitions[0].values.tolist(), self.expected_split_col1_parA)
        self.assertEqual(partitions[1].values.tolist(), self.expected_split_col1_parB)

        gain, split_value, partitions = tree.determine_split("col2")
        self.assertEqual(partitions[0].values.tolist(), self.expected_split_col2_parA)
        self.assertEqual(partitions[1].values.tolist(), self.expected_split_col2_parB)
        pass
