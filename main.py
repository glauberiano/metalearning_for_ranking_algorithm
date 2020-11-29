import os
import pandas as pd
import pickle
import argparse
from argparse import ArgumentParser

from datasets import download
from metatools import metafeatures
from metatools.metalearning import RankingAlgorithm, StatisticalTest
from metatools.classification_algorithms import Classifiers, Dataprep
from utils import Utils, Datasets

def create_performance_dict(models_list, path_databases='datasets/bases_tratadas/', save=False):
    """ Calcula o desempenho e tempo de execução de cada algoritmos em `models_list` para cada base de dados no diretório `path_databases`. A validação dos modelos é feita utilizando `KFold=10`. O desempenho e tempo de execução de cada Fold é salvo no dicionário `dataset_performances`.
    
    Parâmetros:
    ----------
        - models_list : list(), not-null
            lista de modelos SkLearn configurados em `Classifiers.classifiers_key()`.
        - path_databases: IO.path
            caminho para o diretório com bases de dados csv.
        - save: bool, default=False
            salve `dataset_performances` em `classification_algorithms/models/performance_dict.p` (arquivo Pickle).
    Retorna:
    ------
        - dataset_performances: dict()
            Dicionário com o desempenho e tempo que cada algoritmo obteve em cada Fold de validação.
        
    """
    PATH = path_databases
    bases = os.listdir(PATH)
    dict_of_attributes = Datasets.dict_of_attributes()
    dataset_performances = dict()

    print("\nCriando dicionário de desempenho para as bases em \"{}\"".format(PATH))
    for base in bases:
        print("Avaliando base {}...".format(base))
        df_ = pd.read_csv(PATH + base, index_col=[0])
        dataset_info = (base[:-4], df_, dict_of_attributes[base[:-4]][1])
        dataset_performances[base[:-4]] = dict()

        for model_ in models_list:
            clf = Classifiers(dataset_info= dataset_info, algorithm= model_)
            clf.run_model()
            dataset_performances[base[:-4]][model_] = (clf.accuracies,clf.times)
    print("Avaliação concluída.")    

    if save:
        if not os.path.exists('metatools/models/'):
            os.makedirs('metatools/models/')
        
        print("\nSalvando dicionário de performances em {}".format('metalearning/models/performance_dict.p'))
        with open('metatools/models/performance_dict.p', 'wb') as fp:
            pickle.dump(dataset_performances, fp, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset_performances

if __name__ == "__main__":
    ## Parâmetros iniciais: para agilizar a coleta de dados ##
    load_performance_dict = False
    load_dataset = True
    save = True
    
    if not os.path.exists('results/pickle/'):
        os.makedirs('results/pickle/')
        
    models_list = Classifiers.classifiers_key().keys()    

    ## Baixando bases de dados do UCL
    if load_dataset:
        download.run()
        print("Download concuído.")

    # gerando o dicionário de performances
    if load_performance_dict:
        performance_dict = pickle.load(open('metatools/models/performance_dict.p','rb'))
    else:
        performance_dict = create_performance_dict(models_list=models_list, save=True)

    ## Gerando metafeatures e instanciando a classe RankingAlgorithm ##
    df_metafeatures = metafeatures.run()
    metalearning = RankingAlgorithm(performance_dict=performance_dict, classifiers_used=models_list, metafeatures=df_metafeatures)

    for accd in [0.001, 0.01, 0.1]:
        resultado_final = dict()
        print("\nGerando resultados para AccD: {}".format(accd))
        for base in performance_dict.keys():
            recomended_rank = metalearning.ARR_ranking(base=base, n_neighbors=2, AccD=accd)
            ideal_rank = metalearning.ideal_ranking(base=base, AccD=accd)
            spearman_coef = StatisticalTest.spearman_test(recomended_rank, ideal_rank)
            print("{} : {}".format(base, spearman_coef))
            resultado_final[base] = {
                'recommended_rank' : recomended_rank,
                'ideal_rank' : ideal_rank,
                'sperman_coef' : spearman_coef
            }

        if save:
            print("\nSalvando modelo final em {}".format('results/pickle/results_accd_'+ str(accd) +'.p'))  
            with open('results/pickle/results_accd_'+ str(accd) +'.p', 'wb') as fp:
                pickle.dump(resultado_final, fp, protocol=pickle.HIGHEST_PROTOCOL)

    Utils.gera_excel()

    print("Execução concluída.")