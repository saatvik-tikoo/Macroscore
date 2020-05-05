# Old plan
# For each folder get to each file and parse rt_urls_list.
# Label each row as 1 if this row has relvent information
# Also add another column that tracks if this row has any data in the tweet or the url that can be linked to the
# metadata.csv
# about what we need else mark it 0.
# Relvent information means: If the field contains the any string from my list of sources or a doi or a title
# Before that create a new file that has only those dois that are present in the MAG data
# Saving the file: If the file exists then append else create a new one

import pandas as pd
import os


def main():
    # get required data
    print('----------Getting Metadata File----------')
    df_metadata = pd.read_csv('data/metadata.csv', usecols=['doi', 'title', 'source_x', 'pmcid', 'pubmed_id',
                                                            'journal'], dtype={'DOI': object, 'title': object,
                                                                               'source_x': object, 'pmcid': object,
                                                                               'pubmed_id': object, 'journal': object})
    df_metadata.dropna(subset=['doi'], inplace=True)

    print('----------Getting Covid_papers File----------')
    df_paper = pd.read_csv('data/covid_papers.csv', usecols=['DOI'], dtype={'DOI': object})
    mag_papers = set(df_paper.dropna())

    # list of sources that can be connected to covid based on the list of source from metadata csv
    trusted_sources = ['biorxiv', 'czi', 'mackle', 'vrani', 'elsevier', 'enrico', 'marogna', 'schalnenberger',
                       'glossop', 'medrxiv', 'researchgate', 'jungoh', 'pmc', 'who', 'pubmed', 'embase',
                       'cochranelibrary', 'uptodate', 'doi']

    print('----------File Lookup started----------')
    list_of_folders = os.listdir('/data/covid')
    for folder in list_of_folders:
        print('----------In Folder {}----------'.format(folder))
        if ".zip" not in folder:
            list_of_files = os.listdir('/data/covid/{}'.format(folder))
            for file in list_of_files:
                print('----------In File {}----------'.format(file))
                data = pd.read_csv('/data/covid/{}/{}'.format(folder, file))
                data['matchToCovidDataSet_doi'] = -1

                count_source_matches = 0
                count_metadata_match = 0
                for i, row in data.iterrows():
                    print('----------Running Search to match Trusted Sources in row {} of {}----------'.format(i, file))
                    for source in trusted_sources:
                        if source.lower() in str(row['rt_urls_list']).lower() or \
                                (source.lower() in str(row['text']) and 'http' in str(row['text'])):
                            data.at[i, 'SourceInURL'] = 1
                            count_source_matches += 1

                    print('----------Running Search to match to the metadata file in row {} of {}----------'.format(i, file))
                    for j, paper in df_metadata.iterrows():
                        if paper['doi'] in mag_papers and (paper['doi'].lower() in row['rt_urls_list'] or
                                                           paper['doi'].lower() in row['text'] or
                                                           (not pd.isna(paper['title']) and paper['title'].lower() in row['text']) or
                                                           (not pd.isna(paper['pmcid']) and paper['pmcid'].lower() in row['rt_urls_list']) or
                                                           (not pd.isna(paper['pmcid']) and paper['pmcid'].lower() in row['text']) or
                                                           (not pd.isna(paper['pubmed_id']) and paper['pubmed_id'].lower() in row['rt_urls_list']) or
                                                           (not pd.isna(paper['pubmed_id']) and paper['pubmed_id'].lower() in row['text'])):
                            data.at[i, 'matchToCovidDataSet_doi'] = row['matchToCovidDataSet_doi'] + ', ' + paper['doi']
                            count_metadata_match += 1

                if folder not in os.listdir('./data/covid'):
                    print('----------Creating now folder----------')
                    os.mkdir('./data/covid/{}'.format(folder))

                print('----------Adding new file to the folder----------')
                data.to_csv('./data/covid/{}/{}'.format(folder, file))

                print('----------Results for this loop are----------')
                print('Number of Matched Sources: ', count_source_matches)
                print('Number of Metadata Matches: ', count_metadata_match)


if __name__ == '__main__':
    main()
