# metalearning_for_ranking_algorithm

O objetivo desse projeto é replicar os experimentos conduzinhos em  (BRAZDIL; SOARES; COSTA, 2003). O artigo encontra-se no repositório.

## Iniciando

Nesse tutorial mostrará como copiar projeto localmente 

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


## Running the tests

Para testar o desempenho do ranks gerados, os autores utilizaram o coeficiente de correlação entre ranks de Spearman,  teste de Friedman e o procedimento de comparações múltiplas de Dunn. Porém, os métodos não foram implementados até esse momento.


### Formato de algumas variáveis

Exemplos

```python
>>> import pickle

>>> dataset_performance = pickle.load(open('metatools/models/performance_dict.p','rb'))
>>> dataset.keys()
dict_keys(['abalone', 'banknote', 'car' ...])
```

A variável `dataset_performance` tem o seguinte formato:
{   
    Base_1 : 
        {
            'Algoritmo_classifição_1' : ([desempenho do algoritmo 1 em cada Fold N da base 1],
                                        [tempo de execução em cada FOld N da base i]),
            ...

            'Algoritmo_classifição_k' : ([desempenho do algoritmo k em cada Fold N da base 1],
                                        [tempo de execução em cada FOld N da base i]),                                    
        },
    ....

    Base_i : 
    {
        'Algoritmo_classifição_1' : ([desempenho do algoritmo 1 em cada Fold N da base i],
                                    [tempo de execução em cada FOld N da base i]),
        ...

        'Algoritmo_classifição_k' : ([desempenho do algoritmo k em cada Fold N da base i],
                                    [tempo de execução em cada FOld N da base i]),                                    
    }
}   

## Contributing

Please read ....... for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Lucas Mazim de Sousa** - *Initial work* - [Glauberiano](https://github.com/glauberiano)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the ..... file for details


## Referências

BRAZDIL, P.; SOARES, C.; COSTA, J. Ranking learning algorithms: Using ibl and meta-learning on accuracy and
time results. Machine Learning, v. 50, p. 251–277, 03 2003. page.11

## Extras

Tips for good READMME.md
    https://gist.github.com/PurpleBooth/109311bb0361f32d87a2
