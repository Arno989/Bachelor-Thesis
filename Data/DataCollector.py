import csv, io, json, os, requests
import pandas as pd
from datetime import datetime
from DataPoint import DataPoint
import matplotlib.pyplot as plt
from time import sleep

testKeys = ["TSLA", "IBM", "AMD", "NVDA", "AAPL", "MSFT"]
largeCap = ["AAPL", "AMZN", "MSFT", "GOOG", "GOOGL", "BABA", "FB", "BR", "V", "TSM", "TSLA", "WMT", "JNJ", "PG", "MA", "NVDA", "HD", "JPM", "UNH", "VZ", "ADBE", "DIS", "PYPL", "CRM", "INTC", "NFLX", "CMCSA", "KO", "MRK", "BAC", "PFE", "T", "NKE", "NVS", "PEP", "TM", "ABT", "ORCL", "SAP", "TMO", "CSCO", "MCD", "COST", "ABBV", "ASML", "XOM", "DHR", "AVGO", "UPS", "AMGN", "LLY", "ACN", "TMUS", "MDT", "NEE", "ZM", "CVX", "UNP", "BMY", "CHL", "QCOM", "TXN", "CHTR", "SNY", "LIN", "NVO", "LOW", "PM", "SHOP", "HON", "LMT", "IBM", "AMT", "RY", "SBUX", "JD", "WFC", "SNE", "GSK", "BA", "AMD", "MMM", "NOW", "FIS", "C", "HDB", "BUD", "UN", "TOT", "SPGI", "INTU", "BLK", "TD", "DCUE", "BTI", "MDLZ", "ISRGD"]
symbols = []

with open("./symbols.csv") as csvFile:
    csvReader = csv.DictReader(csvFile)
    for row in csvReader:
        symbols.append(row["symbol"])
    print(f"{len(symbols)} symbols loaded")


# https://generator.email/
# https://twelvedata.com/apikey
KEYS = ["5627bb8b1c27492da019304564e9713d", "2fd97503e09f49188de9d41661c927e9", "d2d81638c4174160a4cf71c5c76fb6a5", "af668983f3c546a397ce86eb30dbe63b", "fe741479c1dd4ce7b4568af68d6dc0e3", "7002591a17a64ff7b0c71f754e9538fd", "d20f3b619b0d42f79b9dd881efb2e19d", "16d65454ea354cde84efcf49ecc550ce", "6b6bb02dc9f74d1aa0d9b1ac728d3bcd", "503c681f55284e9c809a29770cd70de9", "91c75dfc1c9644679ef0400546673713", "cedc0d253b384a329982cdd4b3622701", "254afe60b5ce4eb8acc7ead229214d62", "37991b12baa4423eabdfd0d21c8f4da1", "62755fd44f9a4f5486055ac58743e863", "110fd050d7754faea7e9b6d80aa38e39", "5526a5c7e84144da9282d4ea7d5e8c22", "248652eaaffe4a46bb9cdf843e0ee712", "f9aec74e9c5f4ea3bbddd1d4995912d6", "0959f40cd340445dab9bb4a5ffb874e0", "80fbf7216f2244a986c74d6e481d70cc", "845229f208ba420f886b136ab078d706", "5fc24aa883544caaa83f7177a078b39b", "128cc23b4f934ea29c14f2a45aef12d4", "c2b08dda505f45f190f1ebc652a01185", "4f0ab8fa2d4a47c08233bc626b3a338e", "8ea588fc6736415fb3e511433b9f0d91", "1c29cad5712245c0acd32f79cb100cb8", "920a35b1fe3b407eb22818f4dbe35f38", "91c75dfc1c9644679ef0400546673713", "774038194c7542be97b66b15798b30f2"]
KEY_index = 0
API_key = KEYS[KEY_index]

dataDir = "./Prices"



def setNextKey():
    global KEY_index, API_key, KEYS
    if KEY_index >= len(KEYS)-1:
        KEY_index = 0
    else:
        KEY_index += 1
    
    API_key = KEYS[KEY_index]
    #sleep(5/len(KEYS))
 

def getEarliestTimeStamp(symbol: str):
    url = f"https://api.twelvedata.com/earliest_timestamp?symbol={symbol}&interval=1min&apikey={API_key}"
    r = requests.get(url).json()
    setNextKey()
    
    try:
        return r["unix_time"]
    except KeyError:
        if r["code"] == 429:
            print("API limit reached, retrying in 1m")
            sleep(60)
            setNextKey()
            try:
                return r["unix_time"]
            except KeyError:
                print(r)
                return None
        else:
            return None


def getLastTimeStamp(symbol: str):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&apikey={API_key}&outputsize=1"
    r = requests.get(url).json()
    setNextKey()
    
    try:
        return datetime.strptime(r["values"][0]["datetime"], "%Y-%m-%d %H:%M:%S").timestamp()
    except KeyError:
        if r["code"] == 429:
            print("API limit reached, retrying in 1m")
            sleep(60)
            setNextKey()
            try:
                return datetime.strptime(r["values"][0]["datetime"], "%Y-%m-%d %H:%M:%S").timestamp()
            except KeyError:
                print(r)
                exit()
        else:
            print(r["code"])
        

def getLastRecord(symbol: str):
    if os.path.isfile(f"{dataDir}/{symbol}.csv"):
        data = pd.read_csv(f"{dataDir}/{symbol}.csv")
        print(f"[{symbol}]\tData found, latest record at {data.iloc[-1]['timestamp']}")
        return datetime.strptime(data.iloc[-1]["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
    else:
        print(f"[{symbol}]\tNo data found")
        return None


def getDataSet(symbol:str, startTime=None, endTime=None):
    if startTime == None and endTime == None:
        API_URL = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&apikey={API_key}&format=json&outputsize=5000"
        print(f"[{symbol}]\tFetching the last 5000 datapoints")
    elif endTime == None:
        API_URL = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&apikey={API_key}&format=json&order=ASC&start_date={int(startTime)}&outputsize=5000"
        print(f"[{symbol}]\tFetching 5000 datapoints starting from {datetime.utcfromtimestamp(int(startTime))}")
    else:
        API_URL = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&apikey={API_key}&format=json&order=ASC&start_date={int(startTime)}&end_date={int(endTime)}"
        print(f"[{symbol}]\tFetching datapoints from {datetime.utcfromtimestamp(int(startTime))} to {datetime.utcfromtimestamp(int(endTime))}")
        
    r = requests.get(API_URL).json()
    rDataPoints = []
    if "code" not in r:
        for dp in r["values"]:
            rDataPoints.append(
                DataPoint(
                    symbol,
                    datetime.strptime(dp["datetime"], "%Y-%m-%d %H:%M:%S"),
                    float(dp["open"]),
                    float(dp["high"]),
                    float(dp["low"]),
                    float(dp["close"]),
                    float(dp["volume"])
                )
            )
        
        print(f"{len(rDataPoints)} datapoints recieved")
        writeDataSet(rDataPoints)
    else:
        print('No data found')
    return rDataPoints
    


def writeDataSet(dataset = list):
    symbol=dataset[0].symbol
    if os.path.isfile(f"{dataDir}/{symbol}.csv"):
        with open(f'{dataDir}/{symbol}.csv', mode='a') as data_file:
            writer = csv.DictWriter(data_file, lineterminator='\n', delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
            for dataPoint in dataset:
                writer.writerow(dataPoint.__dict__)
    else:
        with open(f'{dataDir}/{symbol}.csv', mode='w+') as data_file:
            writer = csv.DictWriter(data_file, lineterminator='\n', delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, fieldnames=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
            writer.writeheader()
            for dataPoint in dataset:
                writer.writerow(dataPoint.__dict__)

        
def getAllSymbolData (symbol: str):
    startTime = getEarliestTimeStamp(symbol)
    endTime = getLastTimeStamp(symbol)
    lastRecord = getLastRecord(symbol)
    if startTime != None:
        if endTime != lastRecord:
            if lastRecord != None:
                startTime = lastRecord+1
            while startTime < endTime:
                sleep(1)
                response = getDataSet(symbol, startTime=startTime)
                if len(response) != 0:
                    startTime = response[-1].timestamp.timestamp()
                else:
                    startTime += 300000
                setNextKey()
        else:
            print(f"[{symbol}]\tData for symbol already up to date")
    else:
        print(f"[{symbol}]\tSymbol not found, skipping fetch")


def iterateSymbols(symbols: list):
    for symbol in symbols:
        getAllSymbolData(symbol)

iterateSymbols(testKeys)