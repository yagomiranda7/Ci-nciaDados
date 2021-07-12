# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:34:58 2021

@author: asus
"""

import pandas as pd
import math
from datetime import datetime


data = pd.read_csv('C:/Users/asus/.spyder-py3/blood_donation_hist.csv', delimiter= ';', header=0, encoding='latin-1', index_col=False)
data['donation_date'] = pd.to_datetime(data['donation_date'])

def information(id_valido):
  least= data[data['patient_id']==id_valido]['donation_date'].min()
  most= data[data['patient_id']==id_valido]['donation_date'].max()
  actual = '20210401'
  actual = datetime.strptime(actual, '%Y%m%d')
  daysl=(actual- least)
  daysm=(actual - most)
  monthsl=math.floor((daysl.days)/30)
  monthsm=math.floor((daysm.days)/30)
  count_vol= data.groupby('patient_id')['volume_donated_cc'].count()
  sum_vol= data.groupby('patient_id')['volume_donated_cc'].sum()
  voltot = count_vol[id_valido]
  volsum = sum_vol[id_valido]
  dic={"months_since_last_donation": monthsm  , "number_of_donations": voltot, "total_volume_donated_cc": volsum, "months_since_first_donation": monthsl}
  return print(dic)

idpaciente=input('Digite o id do paciente: ')

information(int(idpaciente))
