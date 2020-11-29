#import sys
#sys.path.append('../')

import os
import pandas
import numpy as np
import pickle

def run(PATH = 'datasets/bases_tratadas/'):
    print("Criando metafeatures para as bases em {}".format(PATH))
    if not os.path.exists('metatools/models/'):
        os.makedirs('metatools/models/')

    dict_of_attributes = get_dataset_attributes()

    df_metafeatures = pandas.DataFrame(columns=['base','n_examples','pro_symb_attrs','prop_attr_outliers','class_entropy'])
    bases = os.listdir(PATH)

    for base in bases:
        df_ = pandas.read_csv(PATH + base, index_col=[0])
        dataset_info = (base[:-4], df_, dict_of_attributes[base[:-4]][1])
        df_metafeatures.loc[len(df_metafeatures)] = gera_metafeatures(dataset_info)

    with pandas.ExcelWriter('metatools/models/metafeatures.xlsx') as writer:  
        df_metafeatures.to_excel(writer, index=False, float_format="%.3f") 

    with open('metatools/models/metafeatures.p', 'wb') as fp:
        pickle.dump(df_metafeatures, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    return df_metafeatures


def gera_metafeatures(dataset_info):
    """Calcula as metafeatures seguindo as informações passadas em `dataset_info`
    
    Metafeatures implementadas: número de exemplos, quantidade de atributos categóricos, proporção de outliers, entropia de classe. """

    dataset_name = dataset_info[0]
    dataset = dataset_info[1]
    target_name = dataset_info[2]

    metafeatures = list()

    #adicionando nome da base
    metafeatures.append(dataset_name)

    #adicionando o número de exemplos
    metafeatures.append(dataset.shape[0])

    #proportion of symbolic attributes (prop.symb.attrs)
    metafeatures.append(sum(dataset.loc[:, ~dataset.columns.isin([target_name])].dtypes != float))

    #proportion of attributes with outliers (prop.attr.outliers)
    metafeatures.append(get_outliers(dataset=dataset))

    # entropy of classes (class.entropy)
    metafeatures.append(get_class_entropy(dataset=dataset, target_name=target_name))

    # canonical correlation of the most discriminating single linear combination of numeric \
# attributes and the class distribution (can.cor)
    # metafeatures.append(get_canonicalCor)

    return metafeatures

def get_outliers(dataset):
    """ """

    n_outliers = 0
    for col in dataset.columns:
        if (dataset[col].dtypes == float) | (dataset[col].dtypes == int):
            ratio_ = mean_variance(dataset[col])
            if ratio_ < 0.7:
                n_outliers += 1
    return n_outliers


def get_class_entropy(dataset, target_name):
    """ """
    size_ = len(dataset[target_name])
    target_frequency = dataset[target_name].value_counts()
    entropy = 0

    for value in dataset[target_name].unique():
        class_prob = target_frequency[value] / size_
        entropy += (-class_prob) * np.log2(class_prob)
    return entropy

def mean_variance(array):
    mean1 = np.mean(array)
    alpha = 0.05
    lim = int(len(array) * alpha)
    mean2 = np.mean(array[lim:-lim])
    return mean1 / mean2


def get_dataset_attributes():
    """ Atributos de acordo com o módulo `datasets/download.py` """
    dict_of_attributes = {
        'abalone' : (['Sex', 'Length', 'Diamater', 'Height', 'Whole weight', 'Stucked weight', 'Viscera weight', 'Shell weight', 'Rings'], 'Rings')
        ,'adult' : (['age''workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target'], 'target')
        ,'banknote' : ([str(int) for int in range(4)] + ['target'], 'target')
        ,'car' : (['buying','maint','doors','persons','lug_boot','safety', 'target'], 'target')
        ,'chess1' : ([str(int) for int in range(36)] + ['target'], 'target')
        ,'chess2' : (['w_king_column','w_king_row','w_rook_column','w_rook_row','b_king_column','b_king_row','target'],'target')
        ,'contraceptive' : (['Wife-age','Wife-education','Husband-education','Number-children','Wife-religion','Wife-is-working','Husband-occupation','Standard-of-living','Media-exposure','target'], 'target')
    }
    return dict_of_attributes