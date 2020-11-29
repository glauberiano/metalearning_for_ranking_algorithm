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


## Classe RankingAlgorithm

Nesta classe esta implementado o algoritmo de ranking Adjusted Rate of Ratios (ARR). Os ranks gerados tem o seguinte formato. 

Example:
------
```python
>>>> from metatools.metalearning import RankingAlgorithm
>>>> ranking = RankingAlgorithm(performance_dict=performance_dict, classifiers_used=models_list)
>>>> recomended_rank = metalearning.ARR_ranking(base=['abalone'], n_neighbors=2, AccD=accd)
>>>> recomended_rank
{1: ('DecisionTree', (1.218342761284374, 0.7563956332469395)), 
2: ('RandomForest', (1.1652332108771621, 0.7563509488032218)), 
3: ('Knn10', (1.1104470637896706, 0.7319053522713357)), 
4: ('Knn5', (1.0785604546908567, 0.7128458097177036)), 
5: ('Knn1', (0.9314077313709658, 0.6472733703791613)), 
6: ('LDA', (0.8939102031215063, 0.6753846338630222)), 
7: ('NaiveBayes', (0.6380924560113563, 0.5294468484269127))}
```

É possível comparar os ranking gerados com o método de Spearman utilizando:

```python
>>>> from metatools.metalearning import StatisticalTest
>>>> spearman_coef = StatisticalTest.spearman_test(recomended_rank, ideal_rank)
>>>> spearman_coef
0.964285714
```


## Fórmulas

$ ARR_ap = 10 $

## Authors

* **Lucas Mazim de Sousa** - [Glauberiano](https://github.com/glauberiano)

## Referências

BRAZDIL, P.; SOARES, C.; COSTA, J. Ranking learning algorithms: Using ibl and meta-learning on accuracy and
time results. Machine Learning, v. 50, p. 251–277, 03 2003. page.11

## Extras

Tips for good READMME.md
    https://gist.github.com/PurpleBooth/109311bb0361f32d87a2


            
Formato: 
{ 'Abalone' : 
    {
        'recommended_rank' : 
            {1 : ('alg_x', ARR_x, performance media alg_x),
                2 : ('alg_y', ARR_y, performance media alg_y),
                3 : ('alg_z', ARR_z, performance media alg_z)},
        'recommended_rank' : 
            {1 : ('alg_x', ARR_x, performance media alg_x),
                2 : ('alg_y', ARR_y, performance media alg_y),
                3 : ('alg_z', ARR_z, performance media alg_z)},
        'spearman_coef' : double,
    }
}
