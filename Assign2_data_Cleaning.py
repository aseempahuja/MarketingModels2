"""
1. read  the file
2. make chenages
change obj to categoriacl
3. save into a csv
"""

import pandas as pd

df=pd.read_excel('BMK6107_2.xlsx')

df_m=pd.DataFrame(columns=['s7','s8','cargo3','engHyb','engElec','p35','p40'], dtype=bool)
df.ix[df['carpool']=='yes', 'cp_d']=\
    True
