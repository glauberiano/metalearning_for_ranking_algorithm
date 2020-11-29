import pickle
import pandas 
import os

class Datasets:
    """ Define métodos para realizar o download automático das bases utilizadas no projeto."""

    @staticmethod
    def dict_of_datasets():
        """ Retorna um dicionário com o URL das bases de dados. """
        dict_of_datasets = {
            'abalone' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
            ,'adult' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
            ,'banknote' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
            ,'car' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
            ,'chess1' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king-pawn/kr-vs-kp.data'
            ,'chess2' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data'
            ,'contraceptive' : 'https://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data'
        }
        return dict_of_datasets

    @staticmethod
    def dict_of_attributes():
        """ Retorna um dicionário com o nome das colunas e da variável resposta de cada base."""
        dict_of_attributes = {
                'abalone' : (['Sex', 'Length', 'Diamater', 'Height', 'Whole weight', 'Stucked weight', 'Viscera weight', 'Shell weight', 'Rings'], 'Rings')
                ,'adult' : (['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','target'], 'target')
                ,'banknote' : ([str(int) for int in range(4)] + ['target'], 'target')
                ,'car' : (['buying','maint','doors','persons','lug_boot','safety', 'target'], 'target')
                ,'chess1' : ([str(int) for int in range(36)] + ['target'], 'target')
                ,'chess2' : (['w_king_column','w_king_row','w_rook_column','w_rook_row','b_king_column','b_king_row','target'],'target')
                ,'contraceptive' : (['Wife-age','Wife-education','Husband-education','Number-children','Wife-religion','Wife-is-working','Husband-occupation','Standard-of-living','Media-exposure','target'], 'target')
            }
        return dict_of_attributes

class Utils:
    """ Outros métodos do projeto."""
    @staticmethod
    def gera_excel():
        if not os.path.exists('results/csv/'):
            os.makedirs('results/csv/')

        """ Gera um .csv no diretório `results` com o resultado obtido pelos ranqueamentos gerados no projeto, replicando os resultados do Artigo \"...\". """
        results1 = pickle.load(open('results/pickle/results_accd_0.1.p','rb'))
        results2 = pickle.load(open('results/pickle/results_accd_0.01.p','rb'))
        results3 = pickle.load(open('results/pickle/results_accd_0.001.p','rb'))

        # AccD = 0.01
        df1 = pandas.DataFrame(results1['abalone'])
        df2 = pandas.DataFrame(results1['adult'])
        df3 = pandas.DataFrame(results1['banknote'])
        df4 = pandas.DataFrame(results1['car'])
        df5 = pandas.DataFrame(results1['chess1'])
        df6 = pandas.DataFrame(results1['chess2'])
        df7 = pandas.DataFrame(results1['contraceptive'])

        with pandas.ExcelWriter('results/csv/resultados_accd_01.xlsx') as writer1:  
            df1.to_excel(writer1, sheet_name='abalone')
            df2.to_excel(writer1, sheet_name='adult')
            df3.to_excel(writer1, sheet_name='banknote')
            df4.to_excel(writer1, sheet_name='car')
            df5.to_excel(writer1, sheet_name='chess1')
            df6.to_excel(writer1, sheet_name='chess2')
            df7.to_excel(writer1, sheet_name='contraceptive')

        # AccD = 0.1
        df10 = pandas.DataFrame(results2['abalone'])
        df20 = pandas.DataFrame(results2['adult'])
        df30 = pandas.DataFrame(results2['banknote'])
        df40 = pandas.DataFrame(results2['car'])
        df50 = pandas.DataFrame(results2['chess1'])
        df60 = pandas.DataFrame(results2['chess2'])
        df70 = pandas.DataFrame(results2['contraceptive'])

        with pandas.ExcelWriter('results/csv/resultados_accd_001.xlsx') as writer2:  
            df10.to_excel(writer2, sheet_name='abalone')
            df20.to_excel(writer2, sheet_name='adult')
            df30.to_excel(writer2, sheet_name='banknote')
            df40.to_excel(writer2, sheet_name='car')
            df50.to_excel(writer2, sheet_name='chess1')
            df60.to_excel(writer2, sheet_name='chess2')
            df70.to_excel(writer2, sheet_name='contraceptive')

        # AccD = 1
        df11 = pandas.DataFrame(results3['abalone'])
        df21 = pandas.DataFrame(results3['adult'])
        df31 = pandas.DataFrame(results3['banknote'])
        df41 = pandas.DataFrame(results3['car'])
        df51 = pandas.DataFrame(results3['chess1'])
        df61 = pandas.DataFrame(results3['chess2'])
        df71 = pandas.DataFrame(results3['contraceptive'])

        with pandas.ExcelWriter('results/csv/resultados_accd_0001.xlsx') as writer3:  
            df11.to_excel(writer3, sheet_name='abalone')
            df21.to_excel(writer3, sheet_name='adult')
            df31.to_excel(writer3, sheet_name='banknote')
            df41.to_excel(writer3, sheet_name='car')
            df51.to_excel(writer3, sheet_name='chess1')
            df61.to_excel(writer3, sheet_name='chess2')
            df71.to_excel(writer3, sheet_name='contraceptive')