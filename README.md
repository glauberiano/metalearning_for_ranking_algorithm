# metalearning_for_ranking_algorithm

O objetivo desse projeto é replicar os experimentos conduzinhos em  (BRAZDIL; SOARES; COSTA, 2003). O artigo encontra-se no repositório.

## Iniciando

Este tutorial mostrará como realizar a instalação do pacote e suas principais funções.

### Pré-requisitos

Não é necessário a instalação de nenhum pacote além dos padrões do python.

### Instalação

Para copiar o projeto, clone o repositório localmente:

```
git clone https://github.com/glauberiano/metalearning_for_ranking_algorithm.git
```
Isso salvará o projeto no diretório escolhido.


No terminal, vá para o diretório da pasta e execute:

```
python main.py
```

A função `main` tem as seguintes responsabilidade:
- Criar o diretório `datasets/bases_tratadas/` e realizar o download das bases que foram utilizadas diretamente do UCL.
- Criar o diretório `metatools/models/`, nela será salve em formato `pickle`, um dicionário contendo informações sobre desempenho e tempo de execução de cada algoritmo de classicação selecionado nas bases de dados, esses valores são guardado para cada Fold N utilizando na validação-cruzada. E, tambem em formato `pickle`, um `pandas.DataFrame()` das metafeatures de cada base.
- Criar o diretório `results/pickle`, onde serão salvos em formato `pickle`, os ranking obtidos ao executar as funções do algoritmo de ranqueamento utilizando diferentes valores de AccD, uma variável que leva em consideração a taxa de desempenho obtido por tempo de execução.

## Utilizando o pacote

Caso não opite pela utilização do main. Alguns métodos podem ser executados individualmente para servir a quaisquer outros fins. Segue uma explicação sobre os principais pacotes e métodos.

Modulo download.py
----
Responsável por realizar o download das bases do diretório UCI. Novas bases podem ser adicionadas seguindo  o modelo apresentado.

__Exemplo__
```python
>>> from datasets import download
>>> download.run()
Baixando bases de dados do UCI
abaloneadult
banknote
car
chess1
chess2
contraceptive
Download concluído.
```

Modulo metafeatures.py
-----------
Extrai um conjunto de características de uma base de dados.

__Exemplo__
```python
>>> from metatools import metafeatures
>>> df = metafeatures.run()
>>> df
            base n_examples pro_symb_attrs prop_attr_outliers  class_entropy
0        abalone       4177              1                  0       3.602007
1          adult      32561             14                  0       0.796384
2       banknote       1372              0                  0       0.991128
3            car       1728              6                  0       1.205741
4         chess1       3196             36                  0       0.998576
5         chess2      28056              6                  0       3.504159
6  contraceptive       1473              9                  0       1.539035
```


Classe RankingAlgorithm
----
Nesta classe esta implementado o algoritmo de ranking Adjusted Rate of Ratios (ARR). O classe precisa ser inicializada com um dicionário de performance e uma lista dos algoritmos utilizados. 

Example:
------
```python
>>> import pickle
>>> performance_dict = pickle.load(open('metatools/models/performance_dict.p','rb'))
>>> models_list = ['DecisionTree','RandomForest','Knn10','Knn5','Knn1','LDA','NaiveBayes']
>>>
>>> from metatools.metalearning import RankingAlgorithm
>>> ranking = RankingAlgorithm(performance_dict=performance_dict, classifiers_used=models_list)
>>> recomended_rank = metalearning.ARR_ranking(base='abalone', n_neighbors=2, AccD=accd)
>>> recomended_rank
{1: ('DecisionTree', (1.218342761284374, 0.7563956332469395)), 
2: ('RandomForest', (1.1652332108771621, 0.7563509488032218)), 
3: ('Knn10', (1.1104470637896706, 0.7319053522713357)), 
4: ('Knn5', (1.0785604546908567, 0.7128458097177036)), 
5: ('Knn1', (0.9314077313709658, 0.6472733703791613)), 
6: ('LDA', (0.8939102031215063, 0.6753846338630222)), 
7: ('NaiveBayes', (0.6380924560113563, 0.5294468484269127))}

>>> ideal_rank = ranking.ideal_ranking(base='abalone', AccD=accd)
>>> ideal_rank
{1: ('DecisionTree', (1.3079295554254797, 0.7563956332469395)), 
2: ('Knn10', (1.0352151423294413, 0.7319053522713357)), 
3: ('Knn5', (1.0096716618243968, 0.7128458097177036)), 
4: ('LDA', (0.9540711335934194, 0.6753846338630222)), 
5: ('RandomForest', (0.9188081206169452, 0.7563509488032218)), 
6: ('Knn1', (0.8858036022832777, 0.6472733703791613)), 
7: ('NaiveBayes', (0.7887868599085172, 0.5294468484269127))}
```

É possível comparar os ranking gerados com o método de Spearman utilizando:

```python
>>>> from metatools.metalearning import StatisticalTest
>>>> spearman_coef = StatisticalTest.spearman_test(recomended_rank, ideal_rank)
>>>> spearman_coef
0.964285714
```

## Authors

* **Lucas Mazim de Sousa** - [Glauberiano](https://github.com/glauberiano)

## Referências

BRAZDIL, P.; SOARES, C.; COSTA, J. Ranking learning algorithms: Using ibl and meta-learning on accuracy and
time results. Machine Learning, v. 50, p. 251–277, 03 2003. page.11

## Extras

Tips for good READMME.md
    https://gist.github.com/PurpleBooth/109311bb0361f32d87a2