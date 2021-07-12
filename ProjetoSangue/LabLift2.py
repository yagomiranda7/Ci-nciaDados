# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:04:13 2021

@author: asus
"""
import joblib
import numpy as np
import pandas as pd
import math
from datetime import datetime

#joblib.dump(clf_rf,'blood_donation_model.joblib')

import os
from flask import Flask, request, render_template, make_response

app = Flask(__name__, static_url_path='/static', template_folder='template')      # Iniciando a aplicação.
model = joblib.load('blood_donation_model.joblib')    # Carregando o modelo em disco para a memória da nossa aplicação.

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
  teste =np.array([[monthsm, voltot, volsum, monthsl]])
  return teste


@app.route('/')
def display_gui():
    return render_template('templates.html')              # renderizando o um template html

@app.route('/verificar', methods=['POST'])          # recebe a requisição, coleta os dados a partir dos requests, esses dados compõe as variáveis em uma amostra de teste e faz a predição.
def verificar():
	ID_paciente = request.form['patient_id']
	

# Na função acima temos todos os atributos que foram usados para treinar o modelo.	
	#print(":::::: Dados de Teste ::::::")
	#print("Sexo: {}".format(sexo))
	#print("Numero de Dependentes: {}".format(dependentes))
	#print("Casado: {}".format(casado))
	#print("Educacao: {}".format(educacao))
	#print("Trabalha por conta propria: {}".format(trabalho_conta_propria))
	#print("Rendimento: {}".format(rendimento))
	#print("Valor do emprestimo: {}".format(valoremprestimo))
	#print("\n")

# Fazendo a predição:
	classe = model.predict(information(ID_paciente))[0]
    #mom= information(ID_paciente)[:,0]
    #vot= information(ID_paciente)[:,1]
    #vos= information(ID_paciente)[:,2]
    #mol= information(ID_paciente)[:,3]
    #dic={"months_since_last_donation": mom  , "number_of_donations": vot, "total_volume_donated_cc": vos, "months_since_first_donation": mol, "prediction:": classe}
    #print(dic)
	return render_template('templates.html',classe=str(classe))

if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5500))
        app.run(host='0.0.0.0', port=port)