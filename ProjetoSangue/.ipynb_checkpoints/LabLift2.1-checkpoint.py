import joblib
import numpy as np
import pandas as pd
import math
from datetime import datetime

#Importação dos dados
data = pd.read_csv('C:/Users/asus/.spyder-py3/blood_donation_hist.csv', delimiter= ';', header=0, encoding='latin-1', index_col=False)
data['donation_date'] = pd.to_datetime(data['donation_date'])

#Carragamento do modelo ML
model = joblib.load('blood_donation_model.joblib')

#Receber ID do paciente
ID_paciente = input('Digite o ID do paciente: ')
ID_paciente = int(ID_paciente)

#Função que traz as informações
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
  teste =np.array([[monthsm, voltot, volsum, monthsl]])
  return teste

classe = model.predict(information(ID_paciente))[0]
mom= int(information(ID_paciente)[:,0])
vot= int(information(ID_paciente)[:,1])
vos= int(information(ID_paciente)[:,2])
mol= int(information(ID_paciente)[:,3])
dic={"months_since_last_donation": mom  , "number_of_donations": vot, "total_volume_donated_cc": vos, "months_since_first_donation": mol, "prediction:": classe}
print(dic)