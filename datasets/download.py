import pandas as pd
import numpy as np
import os

def run():
    if not os.path.exists('datasets/bases_tratadas/'):
        os.makedirs('datasets/bases_tratadas/')

    print("Baixando bases de dados do UCI")

    # dict_of_datasets[<nome da base> ::  str] = <endereço da base> :: str 
    dict_of_datasets = {
        'abalone' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
        ,'adult' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        ,'banknote' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
        ,'car' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        ,'chess1' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data'
        ,'chess2' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data'
        ,'contraceptive' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data'
    }

    # dict_of_attributes[<nome da base> ::  str] = (<lista_de_atributor> :: list(), <variável_resposta> :: str)
    dict_of_attributes = {
        'abalone' : (['Sex', 'Length', 'Diamater', 'Height', 'Whole weight', 'Stucked weight', 'Viscera weight', 'Shell weight', 'Rings'], 'Rings')
        ,'adult' : (['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target'], 'target')
        ,'banknote' : ([str(int) for int in range(4)] + ['target'], 'target')
        ,'car' : (['buying','maint','doors','persons','lug_boot','safety', 'target'], 'target')
        ,'chess1' : ([str(int) for int in range(36)] + ['target'], 'target')
        ,'chess2' : (['w_king_column','w_king_row','w_rook_column','w_rook_row','b_king_column','b_king_row','target'],'target')
        ,'contraceptive' : (['Wife-age','Wife-education','Husband-education','Number-children','Wife-religion','Wife-is-working','Husband-occupation','Standard-of-living','Media-exposure','target'], 'target')
    }

    for key in dict_of_datasets:
        print(key)
        name = 'datasets/bases_tratadas/' + key + '.csv'
        df = pd.read_csv(dict_of_datasets[key]
                        ,header=None
                        ,names=dict_of_attributes[key][0])

        df.to_csv(name) 
    print("Download concluído")