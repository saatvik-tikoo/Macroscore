import pandas as pd
import time
from selenium import webdriver
import os
import json

def main():
    papers = pd.read_excel('data/RPPdata.xlsx')
    dois = list(papers['DOI'].dropna())
    print("Starting Browser...")
    driver = webdriver.Chrome(executable_path='C:\\Users\\Saatvik\\Documents\\ChromeDriver\\chromedriver.exe')
    Initial_path = 'C:\\Users\\Saatvik\\Downloads'
    doi_to_file_name = []
    for doi in dois:
        try:
            driver.get('https://sci-hub.tw/' + str(doi))
            driver.find_element_by_xpath('//*[@id="buttons"]/ul/li/a').click()
            time.sleep(30)
            filename = max([Initial_path + "\\" + f for f in os.listdir(Initial_path)], key=os.path.getctime)
            doi_to_file_name.append({
                'doi': doi,
                'file': filename.split('\\')[-1]
            })
        except:
            print(doi, 'is giving error')
            continue
    with open('data/doi_to_file_name_data.json', 'w') as outfile:
        json.dump(doi_to_file_name, outfile)

    driver.close()

def put_names():
    df = pd.read_excel('data/RPPdata.xlsx')
    with open('data/doi_to_file_name_data.json') as outfile:
        data = json.load(outfile)
        new_data = []
        for i in data:
            new_data.append({
                'doi': i['doi'],
                'file': i['file'],
                'title': df.loc[df['DOI'] == i['doi'], 'Study.Title.O'].values[0]
            })

    with open('data/doi_to_file_name_data.json', 'w') as outfile:
        json.dump(new_data, outfile)


if __name__ == '__main__':
    put_names()
