list = [
    "BBCA",
    "BBRI",
    "BMRI",
    "BBNI",
    "BRIS",
    "SMMA",
    "MEGA",
    "BNGA",
    "TOWR",
    "ARTO",
    "BNLI",
    "NISP",
    "CASA",
    "PNBN",
    "BDMN",
    "MKPI",
    "BTPN",
    "BINA",
    "CTRA",
    "SRTG",
    "BBHI",
    "BSDE",
    "PWON",
    "BNII",
    "BBTN",
    "BSIM",
    "BBSI",
    "MPRO",
    "BFIN",
    "BANK",
    "ADMF",
    "LIFE",
    "APIC",
    "BBKP",
    "INPP",
    "BJBR",
    "BMAS",
    "PLIN",
    "RISE",
    "JRPT",
    "BTPS",
    "SMRA",
    "BJTM",
    "MAYA",
    "MFIN",
    "DUTI",
    "DMAS",
    "BBMD",
    "SDRA",
    "AGRO",
    "BKSL",
    "LPKR",
    "KPIG",
    "NOBU",
    "GMTD",
    "MASB",
    "AMAR",
    "RDTX",
    "TUGU",
    "YULE",
    "BBYB",
    "NIRO",
    "MTLA",
    "ASRI",
    "APLN",
    "BACA",
    "MCOR",
    "SFAN",
    "KIJA",
    "SKRN",
    "AGRS",
    "BABP",
    "BKSW",
    "BCAP",
    "MMLP",
    "AMOR",
    "BNBA",
    "VICO",
    "CFIN",
    "PNBS",
    "TRIM",
    "DILD",
    "IMJS",
    "LPCK",
    "BCIC",
    "BGTG",
    "DNAR",
    "AMAG",
    "BSBK",
    "SMIL",
    "OMRE",
    "INPC",
    "TIFA",
    "WOMF",
    "BEKS",
    "BVIC",
    "PANS",
    "ADCP",
    "POLL",
    "GWSA",
    "FMII",
    "LPGI",
    "BBLD",
    "BPFI",
    "RELI",
    "JIHD",
    "SMDM",
    "GSMF",
    "PPRO",
    "HDFA",
    "MDLN",
    "MREI",
    "TRIN",
    "VRNA",
    "AHAP",
    "ASRM",
    "INDO",
    "ELTY",
    "GOLD",
    "SAGE",
    "SWID",
    "BKDP",
    "URBN",
    "AMAN",
    "FUJI",
    "BPTR",
    "MTWI",
    "EMDE",
    "TRUS",
    "HOMI",
    "ASBI",
    "TRJA",
    "RODA",
    "CITY",
    "PEGE",
    "STAR",
    "VAST",
    "VINS",
    "SATU",
    "ESTA",
    "PAMG",
    "PURI",
    "ASDM",
    "NICK",
    "CBPE",
    "ASJT",
    "LPPS",
    "KREN",
    "ASPI",
    "JGLE",
    "NASA",
    "NZIA",
    "ASMI",
    "PUDP",
    "BIPP",
    "KOTA",
    "PADI",
    "MINA",
    "TRUE",
    "RELF",
    "IPAC",
    "AKSI",
    "KBAG",
    "BCIP",
    "CSIS",
    "ATAP",
    "JMAS",
    "TARA",
    "REAL",
    "DADA",
    "POLA",
    "BIKA",
    "MGNA",
    "WIDI",
    "MTSM",
    "HDIT",
    "IBFN",
    "HBAT",
    "MPIX",
    "MSIE",
    "MENN",
    "XCID",
    "XSPI",
    "GRIA",
    "WINR",
    "KOCI",
    "XCIS"
]

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome()

# handle ctrl+c
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    driver.quit()
    sys.exit()

# signal.signal(signal.SIGINT, signal_handler)
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="ta"
)

# cursor object c
c = db.cursor()

data_insert = """
INSERT INTO baru_yakin (kode, tahun, PER, PBR, ROE, DER, DPR) VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

datalist = []
try:
    for i in list:
        print(f"Getting data for {i}")
        temp = []
        driver.get(f"https://id.tradingview.com/symbols/IDX-{i}/financials-statistics-and-ratios/?statistics-period=FY")
        driver.execute_script('window.scrollBy(0, 1000)')
        time.sleep(3)

        # get 2019 data
        totalDate = len(driver.find_elements('xpath','//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[1]/div[4]/div'))
        for y in range(1, totalDate):
            yearnow = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[1]/div[4]/div[{y}]/div/div[1]').text
            if int(yearnow) < 2019 or int(yearnow) > 2021:
                continue

            per = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[8]/div[5]/div[{y}]/div/div').text.encode('ascii', 'ignore').decode().strip('K')
            
            pbr = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[11]/div[5]/div[{y}]/div/div').text.encode('ascii', 'ignore').decode().strip('K')

            roe = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[15]/div[5]/div[{y}]/div/div').text.encode('ascii', 'ignore').decode().strip('K')

            try:
                der = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[24]/div[5]/div[{y}]/div/div').text.encode('ascii', 'ignore').decode().strip('K')
            except:
                der = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[29]/div[5]/div[{y}]/div/div').text.encode('ascii', 'ignore').decode().strip('K')
            
            temp.append({
                int(yearnow) : {
                    'PER' : float(0 if per=='—' or per=='' else per),
                    'PBR' : float(0 if pbr=='—' or pbr=='' else pbr),
                    'ROE' : float(0 if roe=='—' or roe=='' else roe),
                    'DER' : float(0 if der=='—' or der=='' else der),
                }
            })
        print(temp)


        # get dpr data
        driver.get(f'https://id.tradingview.com/symbols/IDX-{i}/financials-dividends/')
        driver.execute_script('window.scrollBy(0, 500)')
        time.sleep(2)
        totalDate = len(driver.find_elements('xpath','//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[1]/div[4]/div'))
        current = 0
        for x in range(1, totalDate+1):
            currentDate = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[1]/div[4]/div[{x}]/div/div').text
            if int(currentDate) < 2019 or int(currentDate) > 2021:
                continue
            try:
                dpr = driver.find_element('xpath',f'//*[@id="js-category-content"]/div[2]/div/div/div[6]/div[2]/div/div[1]/div[2]/div[5]/div[{x}]/div/div').text.encode('ascii', 'ignore').decode().strip('K')
            except:
                dpr = 0
            temp[current][int(currentDate)]['DPR'] = float(0 if dpr=='—' or dpr =='' else dpr)
            current += 1
        
        print(temp)
        
        for index, s in enumerate(temp):
            date = 2019+index
            # for tahun in range(2019, 2022):
            try:
                c.execute(data_insert, (i, date, s[date]['PER'], s[date]['PBR'], s[date]['ROE'], s[date]['DER'], s[date]['DPR']))
            except:
                c.execute(data_insert, (i, date, 0, 0, 0, 0, 0))
            db.commit()
            # datalist.append({
            #     'kode' : i,
            #     'tahun' : date,
            #     'PER' : s[date]['PER'],
            #     'PBR' : s[date]['PBR'],
            #     'ROE' : s[date]['ROE'],
            #     'DER' : s[date]['DER'],
            #     'DPR' : s[date]['DPR']
            # })
        
        print(temp)
    
    # # export to csv
    # import pandas as pd

    # df = pd.DataFrame(datalist)

    # df.to_csv('datanya.csv', index=False)

    # print("Data has been saved to datanya.csv")

    db.close()
    driver.quit()

except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)