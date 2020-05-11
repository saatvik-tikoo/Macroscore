import os
import pandas
import numpy as np


class Macroscore:
    def __init__(self, label_type, feature_type='all', encoding='oneHot', specify_features=False,
                 features=None, fileName=''):
        # label_type: {'pvalue.label', 'O.within.CI.R', 'Meta.analysis.significant'}
        self.label_type = label_type
        # features: {'all', 'common', 'network'}
        self.feature_type = feature_type
        # encoding: {'oneHot', 'label', 'DictVectroize'}
        self.encoding = encoding
        # if only specific features are required.
        self.specify_features = specify_features
        # list of features if specify_features is true
        self.features = features
        self.fileName = fileName
        self.df = None
        self.path_head = '../DataExtraction/WOS/RPPdataConverted'

    def get_data(self):
        if not self.specify_features:
            self.df = pandas.read_excel(self.fileName, encoding='ansi')
        else:
            self.df = pandas.read_excel(self.fileName, encoding='ansi', usecols=self.features)

    def __label_addition__(self, row):
        try:
            if 'X' in str(row['P.value.R']) or str(row['P.value.R']) == 'nan' or 'significant' in str(row['P.value.R']):
                return 0
            if '<' in str(row['P.value.R']):
                val = "{0:.4f}".format(float(row['P.value.R'].split('<')[1].strip()) - float(row['P.value.R'].split('<')[1].strip()) / 10)
            elif '>' in str(row['P.value.R']):
                val = "{0:.4f}".format(float(row['P.value.R'].split('>')[1].strip()) + float(row['P.value.R'].split('>')[1].strip()) / 10)
            elif '=' in str(row['P.value.R']):
                val = row['P.value.R'].split('=')[1].strip()
            elif 'not significant' in str(row['P.value.R']) and row['Direction.R'] == 'same':
                return 1
            elif float(row['P.value.R']) <= 0.05 and row['Direction.R'] == 'same':
                return 1
            else:
                return 0

            if float(val) <= 0.05 and row['Direction.R'] == 'same':
                return 1
            else:
                return 0
        except ValueError:
            return 0

    def __clean_pvalue__(self, prop):
        for i, row in self.df.iterrows():
            if 'significant' in str(row[prop]):
                self.df.at[i, prop] = '1'
            if '<' in str(row[prop]):
                self.df.at[i, prop] = "{0:.4f}".format(float(row[prop].split('<')[1].strip()) - float(row[prop].split('<')[1].strip()) / 2)
            elif '>' in str(row[prop]):
                self.df.at[i, prop] = "{0:.4f}".format(float(row[prop].split('>')[1].strip()) + float(row[prop].split('>')[1].strip()) / 2)
            elif '=' in str(row[prop]):
                self.df.at[i, prop] = row[prop].split('=')[1].strip()
            elif 'not significant' in str(row[prop]):
                self.df.at[i, prop] = '0'

    def __clean_float_result__(self, prop):
        for i, row in self.df.iterrows():
            try:
                if float(row[prop]):
                    continue
            except ValueError:
                self.df.at[i, prop] = 0

    def __clean_effect_size__(self, prop):
        for i, row in self.df.iterrows():
            if '=' in str(row[prop]):
                self.df.at[i, prop] = float(str(row[prop]).split('=')[1].strip())

    def __get_baseline__(self):
        total_true = total_false = total_val = 0
        for i, row in self.df.iterrows():
            total_val += 1
            if row[self.label_type] == 1:
                total_true += 1
            else:
                total_false += 1
        print("Total rows is= ", total_val)
        print("Total papers reproducible", total_true)
        print("total_true % is= ", (total_true / total_val) * 100)
        print("total_false % is=", (total_false / total_val) * 100)

    def __clean_data__(self):
        self.df = self.df.replace(to_replace='X', value=np.nan)
        self.df = self.df.dropna(subset=['DOI', 'Reported.P.value.O', 'Direction.R', 'Meta.analysis.significant', 'O.within.CI.R'])
        self.df = self.df.replace(to_replace=np.nan, value=0)
        self.__clean_pvalue__('Reported.P.value.O')
        self.__clean_float_result__('Surprising.result.O')
        self.__clean_float_result__('Exciting.result.O')
        self.__clean_effect_size__('Effect.size.O')

        if self.label_type == 'pvalue.label':
            self.df['pvalue.label'] = self.df.apply(lambda paper: self.__label_addition__(paper), axis=1)

    def __get__common_features__(self):
        self.__clean_data__()

        self.__get_baseline__()

        # Changing the data-types of some fields so as not to include them in encoding
        self.df = self.df.astype({'Reported.P.value.O': 'float64', 'Exciting.result.O': 'float64', 'Surprising.result.O': 'float64', 'N.O': 'float64',
                                  'Effect.size.O': 'float64'})
        return self.df

    def __get_network_features__(self):
        print('Initial Shape is: ', self.df.shape)
        if self.feature_type == 'network':
            self.__clean_data__()
            self.__get_baseline__()
        for i, row in self.df.iterrows():
            name = row['DOI'].replace('/', '_')
            print('----- Getting details for ', row['DOI'], '-------')

            # Get Citations related data
            file_name = self.path_head + '/{}/paper_{}.txt'.format(name, name)
            all_authors_paper = []
            if os.path.exists(file_name):
                df_doi = pandas.read_csv(file_name, sep='\t', lineterminator='\r', encoding="utf-16le", index_col=False,
                                         quotechar=None, quoting=3, usecols=['AU', 'NR', 'TC'])
                df_doi = df_doi.dropna()
                all_authors_paper = df_doi['AU'].str.split(';')[0]
                self.df.at[i, 'References'] = df_doi['NR'][0]
                self.df.at[i, 'Citation.count.paper.O'] = df_doi['TC'][0]
                print('Total references', df_doi['NR'][0])
                print('Total Citations', df_doi['TC'][0])

            # Get References data: References to same authors
            file_name = self.path_head + '/{}/citations_{}.txt'.format(name, name)
            if os.path.exists(file_name):
                df_doi = pandas.read_csv(file_name, sep='\t', lineterminator='\r', encoding="utf-16le", index_col=False,
                                         quotechar=None, quoting=3, usecols=['AU'])
                df_doi = df_doi.dropna()
                cnt_authors = 0
                for _, row_internal in df_doi.iterrows():
                    authors_ref = row_internal['AU'].split(';')
                    for j in authors_ref:
                        if j in all_authors_paper:
                            cnt_authors += 1
                self.df.at[i, 'References.to.self'] = cnt_authors
                print('Where they have referred themselves: ', cnt_authors)

            # Get Authors data
            for author_col in ['1st.author.O', 'Senior.author.O']:
                folder_name = self.path_head + '/{}/'.format(name)
                keyword = row[author_col]
                if os.path.exists(folder_name):
                    for file in os.listdir(folder_name):
                        if keyword in file:
                            df_doi = pandas.read_csv(folder_name + '/' + file, sep='\t', lineterminator='\r', encoding="utf-16le", index_col=False,
                                                     quotechar=None, quoting=3, usecols=['TC'])
                            df_doi = df_doi.dropna()
                            self.df.at[i, author_col + '.papers'] = cnt_authors
                            print('Number of papers of ' + author_col + ' :', df_doi.shape[0])
                            res = 0
                            for _, row_internal in df_doi.iterrows():
                                res += row_internal['TC']
                            self.df.at[i, author_col + '.citations.of.all.papers'] = res
                            print('Number of Citations of  ' + author_col + ' :', res)
                            break

    def get_feature(self):
        if self.feature_type.lower() == 'common':
            self.__get__common_features__()
        elif self.feature_type.lower() == 'network':
            self.__get_network_features__()
        elif self.feature_type.lower() == 'all':
            self.__get__common_features__()
            self.__get_network_features__()
        else:
            print('Wrong features asked: ', self.features)
            return
        print('Final Shape is: ', self.df.shape)
        self.df.to_excel('data/new_data.xlsx')
        self.__get_baseline__()


if __name__ == '__main__':
    req_columns = ['Study.Title.O', 'Authors.O', 'Volume.O', 'Citation.count.paper.O', 'Number.of.Authors.O', 'DOI', 'Citation.Count.1st.author.O',
                   'Reported.P.value.O', 'Exciting.result.O', 'Surprising.result.O', 'N.O', 'Effect.size.O',
                   'Institution.prestige.1st.author.O', 'Institution.prestige.senior.author.O', 'O.within.CI.R', 'P.value.R',
                   'Direction.R', 'Meta.analysis.significant', 'Citation.count.senior.author.O', '1st.author.O',
                   'Senior.author.O']

    mscore = Macroscore('Meta.analysis.significant', feature_type='common', specify_features=True, features=req_columns,
                        fileName='data/RPPdata.xlsx')
    mscore.get_data()
    mscore.get_feature()
