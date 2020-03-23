import pandas as pd
import time
from selenium import webdriver

if __name__ == '__main__':
    papers = pd.read_excel('data/RPPdata.xlsx')
    dois = list(papers['DOI'].dropna())
    print(len(dois))
    print("Starting Browser...")
    driver = webdriver.Chrome(executable_path='C:\\Users\\Saatvik\\Documents\\ChromeDriver\\chromedriver.exe')
    for doi in dois:
        try:
            driver.get('https://sci-hub.tw/' + str(doi))
            time.sleep(5)
            driver.find_element_by_xpath('//*[@id="buttons"]/ul/li/a').click()
            time.sleep(10)
        except:
            print(doi, 'is giving error')
            continue
    driver.close()
