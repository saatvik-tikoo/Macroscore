import pandas as pd
import urllib.parse

if __name__ == '__main__':
    data = pd.read_csv('data/CORD_19/metadata.csv', usecols=['doi'])
    embedding_filenames = ['data/covid19_kb/node2vec_2hops.emb', 'data/covid19_kb/node2vec_whole.emb']
    final_filenames = ['data/covid19_kb/finaldata_2hops.xlsx', 'data/covid19_kb/finaldata_whole.xlsx']

    for idx, file in enumerate(embedding_filenames):
        print('-----For File {}-----'.format(file))
        with open(file, 'r') as f_read:
            df = data.copy()
            lines = f_read.readlines()
            lines_dict = dict()
            cnt_total = cnt_available = 0
            print('-----Generating Dictionary-----')
            for idx_l, line in enumerate(lines):
                if idx_l > 0:
                    vals = line.split(' ')
                    vals[0] = urllib.parse.unquote(vals[0])
                    lines_dict[vals[0]] = vals[1:]
            print('-----Adding Rows-----')
            for i, row in df.iterrows():
                doi = row['doi']
                if doi in lines_dict:
                    cnt_available += 1
                    fields = lines_dict[doi]
                    for _, col in enumerate(fields):
                        df.at[i, 'new_feature_' + str(_ + 1)] = col

                cnt_total += 1
                if cnt_total % 1000 == 0:
                    print('Total Rows checked: {} and total found till now: {}'.format(cnt_total, cnt_available))
            print('-----Saving File-----')
            df.dropna(inplace=True)
            print(df.shape)
            df.to_excel(final_filenames[idx])
