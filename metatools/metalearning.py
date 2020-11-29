import itertools
import pickle
import numpy as np
import pandas as pd
from collections import Counter

""" 
    Pacote metalearning.py
    Contém um coleção de algoritmos utilizados em Metalearning
"""

class RankingAlgorithm:
    """ Está é a classe principal do projeto. Com ela calculamos o ranking recomendado (rr) de cada algoritmo em `classifiers_used` para um base de dados e o ranking ideal (ri) dos algoritmos. O método estático `statistic_test()` calcula o teste de \"...\" para rr e ri.
    Parâmetros:
    ----------
        - classifiers_used: list, not-null
            Lista com os algoritmos que serão utilizados para treinar 
        
        - performance_dict: dict, not_null
            Dicionário com informações sobre desempenho e tempo de execução de uma lista de algoritmos aplicados a uma série de bases. Mais informações na classe `classificadores.Classifiers`.
        - is_ideal: bool, default=False
            Se is_ideal = True, o método `ARR` será calculado para cada Fold do cross-is_ideal.
    
    """

    def __init__(self, classifiers_used, performance_dict, metafeatures, is_ideal=False):
        #with open('metatools/models/metafeatures.p', 'rb') as fp:
        #    metafeatures = pickle.load(fp)
        self.df_metafeatures = metafeatures
        self.performace_dict = performance_dict
        self.is_ideal = is_ideal
        self.classifiers_used= classifiers_used
        self.general_algorithm_performance = self._general_performance(classifiers_used=classifiers_used,
                                                                      performance_dict=performance_dict)


    def _general_performance(self, classifiers_used, performance_dict):
        """ Retorna a média geral de desempenho de cada algoritmo em `classifiers_used`. """
        general_algorithm_performance = dict()
        for algorithm in classifiers_used:
            means_ = list()
            for base in performance_dict.keys():
                means_.append(np.mean(performance_dict[base][algorithm][0]))
            general_algorithm_performance[algorithm] = np.mean(means_)

        return general_algorithm_performance

    def ARR(self, datasets, alg1, alg2, AccD, is_ideal):
        """Calcula ARR do `alg1` em relação ao `alg2` levando em consideração a importância relativa entre acurácia e tempo de execução `AccD`.
        
        Parâmetros:
        -----------
        - datasets: <str>
        - alg1: <str>
        - alg2: <str>
        - AccD: double
        - is_ideal:
            Se o valor `True`, calcula ARR para o ranking ideal. `False` calcula para o ranking recomendado.

        Retorna:
        --------
        - arr_values: list()
            Para cada base em `datasets` retorna o ARR entre os algoritmos `alg1` e `alg2`. Se `is_ideal=True`, arr_values utiliza o desempenho dos algoritmos em cada Fold N das bases. Se `is_ideal=False`, arr_values utiliza a média dos desempenhos de cada algoritmos.
        """

        arr_values = list()
        if is_ideal:
            for base in datasets:
                for j, performance in enumerate(self.performace_dict[base][alg1][0]):
                    SR = (self.performace_dict[base][alg1][0][j]) / (self.performace_dict[base][alg2][0][j])
                    denominador = 1 + AccD * np.log( (self.performace_dict[base][alg1][1][j]) / (self.performace_dict[base][alg2][1][j]) )
                    resultado  = SR / denominador
                    arr_values.append(resultado)

        else:
            for base in datasets:
                SR = (np.mean(self.performace_dict[base][alg1][0])) / (np.mean(self.performace_dict[base][alg2][0]))
                denominador = 1 + AccD * np.log( (np.mean(self.performace_dict[base][alg1][1])) / (np.mean(self.performace_dict[base][alg2][1])) )
                resultado  = SR / denominador
                arr_values.append(resultado)

        return arr_values 

    def get_ARR(self, closest_datasets, classifiers_used, AccD, is_ideal):
        """ Gera o ARR de cada `classifiers_used` nas bases `closest_datasets`

        Parâmetros:
        -----------
        - closest_datasets :: list(<str>)
            bases de dados utiliza no cálculo do ARR.
        - classifiers_used :: list(<str)
            lista de algoritmos de classificação candidatos.

        Retorna:
        --------
        df_ARR :: pandas.DataFrame()
            Tabela indexados pelos algoritmos em `classifiers_used`. df_ARR[i][j] contém o ARR do algoritmo i em relação ao algoritmo j.
        """
        null_matriz = np.zeros(shape=(len(classifiers_used),len(classifiers_used)))
        df_ARR = pd.DataFrame(data=null_matriz ,columns=classifiers_used, index=classifiers_used)
        algorithms_pairs = list(itertools.product(classifiers_used, classifiers_used))
        for par_algorithm in algorithms_pairs:
            if (par_algorithm[0] == par_algorithm[1]):
                pass
            else:
                arr_ = self.ARR(datasets=closest_datasets, 
                                alg1=par_algorithm[0], 
                                alg2=par_algorithm[1],
                                AccD=AccD,
                                is_ideal=is_ideal)
                if is_ideal:
                    df_ARR.loc[par_algorithm[0], par_algorithm[1]] = np.mean(arr_)
                else:
                    df_ARR.loc[par_algorithm[0], par_algorithm[1]] = np.prod(arr_)**(1/len(closest_datasets))
        return df_ARR

    def ARR_ranking(self, base, n_neighbors, AccD):
        """ Gera o ranking dos algoritmos recomendados para `base` utilizando o método de ranqueamento ARR para `n_neighbors` vizinhos mais próximos.
        
        Parâmetros:
        -----------
        - base :: pd.DataFrame(). not-null
            Base de dados que desejamos obter o rank de algoritmos.
        - n_neighbors :: int, not-null
            Vizinhos mais próximos de `n_neighbors`
        - AccD :: double, not-null
            Pode ser qualquer valor dentre [0.1, 0.01, 0.01]. Este parâmetro é fornecido pelo usuário e representa a quantidade de precisão que ele está disposto a negociar por uma aceleração ou desaceleração 10 vezes maior. Para exemplo, AccD = 10% significa que o usuário está disposto a trocar 10% de precisão por 10 vezes acelerar / desacelerar.
        Retorna:
        --------
        - ARR_rank :: dict()
            ranking dos algoritmos candidatos.             
        """
        knn_ = KNN()

        closest_datasets = knn_.run(n_neighbors=n_neighbors,
                                                  df_metafeatures=self.df_metafeatures, 
                                                  target=base)

        df_ARR = self.get_ARR(closest_datasets=closest_datasets, 
                            classifiers_used=self.classifiers_used,
                            AccD=AccD,
                            is_ideal=False)

        ARR_rank = dict()
        n_algorithms = 7

        for indice in df_ARR.index:
            ARR_rank[indice] = ( np.sum(df_ARR.loc[indice,:]) / n_algorithms , self.general_algorithm_performance[indice] )

        ARR_rank = {k+1: v for k, v in enumerate(sorted(ARR_rank.items(), key=lambda item: item[1][0], reverse=True))}
        return ARR_rank

    def ideal_ranking(self, base, AccD):
        """ Gera o ranking ideal de uma `base`. """
        selected_bases = [k for k in self.performace_dict.keys() if k not in [base]]
        classifiers_used= self.classifiers_used
        matriz_ARR = np.zeros(shape=(len(classifiers_used),len(classifiers_used)))
        df_ARR = pd.DataFrame(data=matriz_ARR ,columns=classifiers_used, index=classifiers_used)
        ARR_rank = dict()

        df_ARR = self.get_ARR(closest_datasets=selected_bases, 
                                 classifiers_used=self.classifiers_used,
                                 AccD=AccD,
                                 is_ideal=True)
        for indice in df_ARR.index:
            ARR_rank[indice] = ( np.sum(df_ARR.loc[indice,:]) / len(classifiers_used) , self.general_algorithm_performance[indice] )

        ARR_rank = {k+1: v for k, v in enumerate(sorted(ARR_rank.items(), key=lambda item: item[1][0], reverse=True))}
        return ARR_rank

class KNN:
    """ Implementação do algoritmo KNN modificado para gerar uma matriz de distâncias entre meta-features. """

    def __init__(self):
        pass

    def _dist(self, metafeatures, dataset1, dataset2):
        a = metafeatures.iloc[metafeatures[metafeatures['base'] == dataset1].index[0], ~metafeatures.columns.isin(['base'])]
        b = metafeatures.iloc[metafeatures[metafeatures['base'] == dataset2].index[0], ~metafeatures.columns.isin(['base'])]
        distancia = np.abs(a - b)

        #import ipdb; ipdb.set_trace();
        for ind in distancia.index:
            if max(metafeatures[ind]) > 0:
                distancia[ind] = distancia[ind] / max(metafeatures[ind])
            else:
                if distancia[ind] == 0:
                    distancia[ind] = 0
                else:
                    distancia[ind] = 1
        return np.sum(distancia)

    def train(self, metafeatures, df_column_name='base'):
        """ Calcula a matriz de distâncias da base `metafeatures`.
        
        Parâmetros:
        ----------
        - metafeatures :: pd.DataFrame()
            Base com as metafeatures. Mais informações em "METAFEATURES"
        """
        database_pairs = list(itertools.combinations(metafeatures[df_column_name], 2))

        matriz_de_distancias = np.zeros(shape=(metafeatures[df_column_name].shape[0],
                                               metafeatures[df_column_name].shape[0]))

        df_distancias = pd.DataFrame(data=matriz_de_distancias,
                                     columns=metafeatures[df_column_name].values, 
                                     index=metafeatures[df_column_name].values)

        for base1, base2 in database_pairs:
            df_distancias.loc[base1, base2] = df_distancias.loc[base2, base1] = self._dist(
                                                                    metafeatures=metafeatures, 
                                                                    dataset1=base1,
                                                                    dataset2=base2)

        return df_distancias

    def predict(self, n_neighbors, df_distances, base):
        """ Retorna os `n_neighbors` mais próximos de base.
        
        Parâmetros:
        - n_neighbors :: int, not-null
            Número de vizinhos próximos.
        - df_distances :: pandas.DataFrame(), not-null
            Matriz de distâncias entre as metafeatures de cada base.
        - base :: string, not-null
            Base alvo da predição.
        """
        #import ipdb; ipdb.set_trace()
        #dists_ = np.sort(df_distances[base])[1:n_neighbors+1]
        #n_closest = df_distances.index[df_distances[base].isin(dists_)]
        n_closest = df_distances.sort_values(by=base)[1:n_neighbors+1]
        return n_closest

    def run(self, df_metafeatures, target, n_neighbors):
        df_distances =  self.train(metafeatures=df_metafeatures)
        n_closest = self.predict(n_neighbors=n_neighbors, df_distances=df_distances, base=target)
        return n_closest

class StatisticalTest:
    @staticmethod
    def spearman_test(recomended_rank, ideal_rank):
        """ Retorna o coeficiente de Spearman enter dois ranks. Os ranks enviados devem seguir o seguinte formato:
        {
            Position_alg_x : ('alg_x', ARR_x, desempenho medio alg_x),
            Position_alg_x : ('alg_y', ARR_y, desempenho medio alg_y),
            Position_alg_x : ('alg_z', ARR_z, desempenho medio alg_z)
        }
        """
        algorithm_pos_rr = dict()
        for item in recomended_rank.items():
            algorithm_pos_rr[item[1][0]] = item[0]

        algorithm_pos_ri = dict()
        for item in ideal_rank.items():
            algorithm_pos_ri[item[1][0]] = item[0]

        algorithms_ = algorithm_pos_rr.keys()

        diff_ = list()
        for algorithm in algorithms_:
            diff_.append((algorithm_pos_rr[algorithm] - algorithm_pos_ri[algorithm])**2)

        spearman_coef = 1 - (6*(np.sum(diff_))) / (len(algorithms_)**3 - len(algorithms_))
        return spearman_coef

    @staticmethod
    def mean_average_correlation(rankings_set, baseline):
        """ Compara um conjunto de rankings com um baseline. """

        mean_average_correlation = dict()
        for key, rank in rankings_set.items():
            rs_knn = np.mean([item[1]['spearman_coef'] for item in rank.items()])
            rs_baseline = np.mean([item[1]['spearman_coef'] for item in baseline['accd_0001'].items()])
            count = dict(Counter([1 if item[1]['spearman_coef'] >= rs_baseline else 0 for item in rank.items()]))
            mean_average_correlation[key] = (rs_knn, count)
        return mean_average_correlation

    @staticmethod
    def friedman_test():
        """ a distribution-free hypothesis test on the difference between more than two population means. Could perform correction factor if many ties occurs.
        
        Colocar M na cdf qui².

        Olhar tabela qui² M.

        https://people.richland.edu/james/lecture/m170/tbl-chi.html
        """

        pass 