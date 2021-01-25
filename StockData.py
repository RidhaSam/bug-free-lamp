import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

textFile = input("Enter path to text file(two backslashes required)\n")
prelimDf = pd.read_csv(textFile, header=None)
symbList = prelimDf[:][0].values.tolist()
for i in range(len(symbList)):
    symbList[i] = symbList[i].upper()
boughtList = prelimDf[:][1].values.tolist()

currentList = []
bookList = []
recessList = []
peList = []
divdList = []
epsList = []
pftMgList = []
cashList = []
debtList = []
cdList = []
dteList = []
crrRatList = []
ocfList = []
errorList = []

def numberLetterReader(money):
    letterIndex = money.find("B")
    if (letterIndex == -1):
        letterIndex = money.find("M")
        value = float(money[:letterIndex])
    else:
        value = float(money[:letterIndex])*1000
    return value

for symb in symbList:
    mainURL = "https://finance.yahoo.com/quote/" + symb + "?p=" + symb
    statURL = "https://finance.yahoo.com/quote/" + symb + "/key-statistics?p=" + symb
    page1 = requests.get(mainURL)
    page2 = requests.get(statURL)
    soup1 = BeautifulSoup(page1.content, 'html.parser')
    soup2 = BeautifulSoup(page2.content, 'html.parser')

    stat0 = [entry.text for entry in soup1.find_all('span', {'class':'Trsdu(0.3s)'})]
    stat1 = [entry.text for entry in soup2.find_all('td', {'class':'Fw(500) Ta(end) Pstart(10px) Miw(60px)'})]

    if (len(stat1) == 0):
        print(symb + " is an incorrect symbol, it is omitted from the data")
        error = 'N/A'
        pftMgList.append(error)
        cashList.append(error)
        debtList.append(error)
        bookList.append(error)
        divdList.append(error)
        ocfList.append(error)
        crrRatList.append(error)
        dteList.append(error)
        epsList.append(error)
        currentList.append(error)
        peList.append(error)
        cdList.append(error)
        recessList.append(error)
        errorList.append(symb)
        continue
    
    pftMgList.append(stat1[39])
    try:
        cashList.append(round(numberLetterReader(stat1[-8]),2))
    except:
        cashList.append('N/A') 
    try:
        debtList.append(round(numberLetterReader(stat1[-6]),2))
    except:
        debtList.append('N/A') 
    bookList.append(stat1[-3])
    divdList.append(stat1[19])
    try:
        ocfList.append(round(numberLetterReader(stat1[-2]),2))
    except:
        ocfList.append('N/A') 
    crrRatList.append(stat1[-3])
    dteList.append(stat1[-5])
    epsList.append(stat1[-10])
    currentList.append(stat0[0])
    peList.append(stat0[-3])
    try:
        cdList.append(round((numberLetterReader(stat1[-8])/numberLetterReader(stat1[-6])),2))
    except:
        print("Cash/Debt cannot be calculated")
        cdList.append('N/A') 
    try:
        tempdf = yf.download(symb.upper(),start='2009-01-01',end='2009-01-03',progress=False)
        recessList.append(round(tempdf.iloc[0]['Close'],2))
    except:
        print("Historical Data unavailable")
        recessList.append('N/A')         

data = {'Current':currentList, 'Bought':boughtList, 'Book':bookList, '2009':recessList, 'PE':peList, 'Divd':divdList, 
'EPS':epsList, 'Profit Mg':pftMgList, 'Cash':cashList, 'Debt':debtList,'C/D':cdList,'Debt to Eq':dteList, 'Curnt Ratio':crrRatList, 'Op Csh Flow':ocfList}
df = pd.DataFrame(data = data, index = symbList)
df = df.drop(errorList)
try:
    cdSortedDf = df.sort_values('C/D', ascending = False)
    cdSortedDf.to_csv('stockDataCDsorted.csv')
except:
    print("Data cannot be sorted.")
    df.to_csv('stockDataCSV.csv')