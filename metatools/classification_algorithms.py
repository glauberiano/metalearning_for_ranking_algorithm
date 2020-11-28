import sys
#sys.path.append(0, '..')

import time
import os
import pandas as pd

#from datasets.datasets import DataInfo

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class Dataprep:
    """"Dataprep.
    
    Classe com métodos de tratamento de variáveis.
    """

    @staticmethod
    def onehotencoder(dataset, column_name):
        """One Hot Encoder
        Codifica variáveis categóricas em um vetor numérico binário.
        Parâmetros:
        ----------
        dataset : <pandas.Dataframe>
            Base de dados
        column_name : <str>
            Coluna que deve ser transformada.
        
        Retorna:
        -------
        dataset_enc : <pandas.DataFrame>
            Base de dados `dataset` com coluna `column_name` transformada.
        """

        enc = OneHotEncoder()
        enc.fit(dataset[column_name].values.reshape(-1, 1))
        encoded_values = enc.transform(dataset[column_name].values.reshape(-1, 1)).toarray()
        encoded_columns = enc.get_feature_names([column_name])
        df_encoded = pd.DataFrame(data= encoded_values,
                                  columns= encoded_columns)

        dataset_enc = dataset.copy()
        dataset_enc = dataset_enc.drop(columns=[column_name])
        dataset_enc = pd.concat([df_encoded, dataset_enc], axis=1)

        return dataset_enc

    @staticmethod
    def labelencoder(dataset, column_name):
        """Label Encoder
        Codifica variáveis resposta categóricas em um vetor numérico entre [0 e n_classes-1].
        Parâmetros:
        ----------
        dataset : <pandas.Dataframe>
            Base de dados
        column_name : <str>
            Coluna que deve ser transformada.
        
        Retorna:
        -------
        dataset_enc : <pandas.DataFrame>
            Base de dados `dataset` com coluna `column_name` transformada.
        """

        le = LabelEncoder()
        le.fit(dataset[column_name].values)
        dataset[column_name] = le.transform(dataset[column_name].values)
        return dataset

class Classifiers:
    """Classificador
    
    Essa classe é responsável por verificar o desempenho preditivo que o algoritmo de classificaçao a(j) tem na base de dados d(i).
    
    Parâmetros
    ----------
    - dataset_info : tupla, (<dataset_name> :: str, pandas.DataFrame(), <target_column> :: str), not_null
        Informações sobre a base de dados devem ser encapsulados em um tupla e enviados nesta variável. A tupla deve conter o apelido da base de dados, seus valores em formato pandas.DataFrame() e o nome da variável resposta em formato string.
    
    - algorithm : str, not_null
        Nome de um dos algoritmos definidos em Classifiers.classifiers_key()
    Atributos
    ---------
        - self.dataset_name: apelido de uma base de dados.
        - self.dataset: pandas.DataFrame() com os valores da base.
        - self.target_name: nome da coluna com a variável resposta.
        - self.algorithm: nome do modelo que irá avaliar a base `self.dataset`.
        - self.accuracies: lista com a acurácia obtida pelo modelo em cada Fold da base de dados.
        - self.times: lista com o tempo de execução do modelo em cada Fold da base de dados.
    """

    def __init__(self, dataset_info, algorithm):
        self.dataset_name = dataset_info[0]
        self.dataset = dataset_info[1].copy()
        self.target_name = dataset_info[2]
        self.algorithm = Classifiers.classifiers_key(key=algorithm)

    def run_model(self, hot_enconding=True):
        """ Calcula a acurácia e tempo de execução que um algoritmo obtém em cada Fold de uma base de dados.
        
        Parâmetros:
        ------------
        - hot_encoding: bool, default=True.
            Se `hot_encoding=True`, transforma as variáveis categóricas da base em numericas. Utiliza `Dataprep.onehotencoder()` para variáveis independentes e o `Dataprep.labelencoder()` para variaveis dependentes.
        Retorna:
        --------
        - Define os atributos de classe `self.accuracies` e `self.times`.
        
        """

        if hot_enconding:
            columns = self.dataset.columns

            for col in columns:
                if (self.dataset[col].dtype == object):
                    if (col == self.target_name):
                        self.dataset = Dataprep.labelencoder(dataset= self.dataset, column_name= col)
                    else:
                        self.dataset = Dataprep.onehotencoder(dataset= self.dataset, column_name= col)

        X = self.dataset.loc[:, ~self.dataset.columns.isin([self.target_name])].copy()
        y = self.dataset[self.target_name]

        self.accuracies = list()
        self.times = list()

        from sklearn.model_selection import KFold
        crossfold = KFold(n_splits=10, shuffle=True, random_state=42)

        for train_index, test_index in crossfold.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            ini = time.time()
            self.algorithm.fit(X=X_train, y=y_train)
            pred = self.algorithm.predict(X_test)
            self.times.append((time.time() - ini) + 0.001)

            accuracy = 0
            for i, indice in enumerate(X_test.index):
                if y_test[indice] == pred[i]:
                    accuracy = accuracy+1
            self.accuracies.append(accuracy/len(X_test))


    @staticmethod
    def classifiers_key(key=None):
        classifiers_dict = {
            'DecisionTree' : DecisionTreeClassifier()
            ,'RandomForest' : RandomForestClassifier()
            ,'NaiveBayes' : GaussianNB()
            ,'Knn1' : KNeighborsClassifier(n_neighbors=1)
            ,'Knn5' : KNeighborsClassifier(n_neighbors=5)
            ,'Knn10' : KNeighborsClassifier(n_neighbors=10)
            ,'LDA' : LinearDiscriminantAnalysis()
        }

        if key == None:
            return classifiers_dict
        else:
            return classifiers_dict[key]