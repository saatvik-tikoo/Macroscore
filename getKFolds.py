import pandas as pd

def get_random_kfolds(df, authors, output, k, fold, visited):
    if len(authors) <= 0:
        return

    for auth_idx in authors:
        if not visited[auth_idx]:
            visited[auth_idx] = True
            temp_df = df.loc[df['Authors.O'].str.contains(auth_idx, case=False)]
            if temp_df.shape[0] > 0:
                for _, row in temp_df.iterrows():
                    output.append({
                        'Doi': row['DOI'],
                        'Fold': fold % k + 1
                    })

                authors_new = set()
                for au_x in list(temp_df['Authors.O']):
                    au_x = str(au_x)
                    for j in au_x.split(','):
                        auth = j.strip().lower()
                        if auth not in authors_new and not visited[auth]:
                            authors_new.add(auth)

                df = df.loc[df.index.difference(temp_df.index)]
                get_random_kfolds(df, list(authors_new), output, k, fold, visited)

def main():
    data = pd.read_excel('data/new_data.xlsx', encoding='ansi')
    data.dropna(inplace=True)

    authors = set()
    visited = dict()

    for au in list(data['Authors.O']):
        au = str(au)
        for j in au.split(','):
            author = j.strip().lower()
            if author not in authors:
                authors.add(author)
                visited[author] = False

    output = []
    df = data.copy()
    fold = 0
    k = 10
    for auth_idx in authors:
        if not visited[auth_idx]:
            visited[auth_idx] = True
            temp_df = df.loc[df['Authors.O'].str.contains(auth_idx, case=False)]
            if temp_df.shape[0] > 0:
                for _, row in temp_df.iterrows():
                    output.append({
                        'Doi': row['DOI'],
                        'Fold': (fold % k) + 1
                    })

                authors_new = set()
                for au_x in list(temp_df['Authors.O']):
                    au_x = str(au_x)
                    for j in au_x.split(','):
                        auth = j.strip().lower()
                        if auth not in authors_new and not visited[auth]:
                            authors_new.add(auth)

                df = df.loc[df.index.difference(temp_df.index)]
                get_random_kfolds(df, list(authors_new), output, k, fold, visited)
                fold += 1

    for obj in output:
        data.at[data.where((data['DOI'] == obj['Doi'])).dropna(subset=['DOI']).index, 'Fold_Id'] = obj['Fold']

    data.to_excel('data/new_data.xlsx')


if __name__ == '__main__':
    main()
