# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:04:39 2020

@author:Yago
"""
#Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import math
from math import floor


#Leitura dos arquivos
data1 = pd.read_csv('mfeat-fac.txt', delimiter= '\s+', header=None, index_col=False)
data2 = pd.read_csv('mfeat-fou.txt', delimiter= '\s+', header=None, index_col=False)
data3 = pd.read_csv('mfeat-kar.txt', delimiter= '\s+', header=None, index_col=False)

#Normalizar cada data de arquivo
df1 = pd.DataFrame(data1)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled1 = min_max_scaler.fit_transform(df1)
df_normalized1 = pd.DataFrame(np_scaled1)

#Criar a Matriz de dissimilaridades
df_normalized1 = df_normalized1.values
distancias1 = pdist(df_normalized1, metric='euclidean')
distancias1 = squareform(distancias1)  


df2 = pd.DataFrame(data2)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled2 = min_max_scaler.fit_transform(df2)
df_normalized2 = pd.DataFrame(np_scaled2)

df_normalized2 = df_normalized2.values
distancias2 = pdist(df_normalized2, metric='euclidean')
distancias2 = squareform(distancias2)   #Matriz de dissimilaridades

df3 = pd.DataFrame(data3)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled3 = min_max_scaler.fit_transform(df3)
df_normalized3 = pd.DataFrame(np_scaled3)

df_normalized3 = df_normalized3.values
distancias3 = pdist(df_normalized3, metric='euclidean')
distancias3 = squareform(distancias3)   #Matriz de dissimilaridades

distancia_arr = [distancias1, distancias2,distancias3]   #Array das 3 matrizes de dissimila


m=1.6    #Coeficiente de fuzzificação
p=3      #Numero de matrizes simultanes
n=2000   #Numero de elementos
T=150  #Limite de iterações
K=10   #Num de grupos
er=10**-10  #Erro
coef= 1/(m-1)
iteration = 0

j2_min = 99999999
gp2_min = [[]]
obj_k_min_global = []


while (iteration < 10 ):

  Gp = np.random.rand(K,n)   #linhas = grupos; colunas = objeto
  Gp2 = Gp/Gp.sum(axis=0,keepdims=1)   #Matriz de grau de pertinencia; somatorio nos grupos =1

  alfa = np.zeros((K,p), dtype=np.float64)
  alfa[0:K, 0:p] = 1.0          #Matriz inicial de pesos (para matriz unica, não muda nada)
  J = 0
  J_2 = 1
  counter = 1
  obj_k_min = []

  for k in range(K):
    arg_min_k = 999999
    index_min_k = 0
    for h in range(n):
      sum1 = 0.0
      for i in range(n):
        sum2 = 0.0 
        for j in range(p):
          sum2 += alfa[k,j] * distancia_arr[j][i,h]
        sum1 += (Gp2[k,i]**m) * sum2
      if (arg_min_k > sum1 and (h not in obj_k_min)):
        arg_min_k = sum1
        index_min_k = h
    obj_k_min.append(index_min_k)     #Cálculo dos protótipos iniciais de cada grupo

  for k in range(K):
    for i in range(n):
      sum_geral = 0
      for h in range(K):
        sum_arr_1 = 0
        sum_arr_2 = 0
        for j in range(p):
          sum_arr_1 += alfa[k,j] * distancia_arr[j][i,obj_k_min[k]]
          sum_arr_2 += alfa[h,j] * distancia_arr[j][i,obj_k_min[h]]
        if (sum_arr_2 != 0):
          sum_geral += (sum_arr_1 / sum_arr_2)**coef 
        if (sum_geral == 0):
          Gp2[k,i] = 1
        else:
          Gp2[k,i] = sum_geral**-1     #Cálculo da matriz de grau de pertinencia

  for i in range(n):
      for k in range(K):
        if (Gp2[k,i] == 1):
          for j in range(K):
            if(j != k):
              Gp2[j,i] = 0      #Atulização de pertinencia 0 em grupos que os objetos que são protótipos de outro grupo

  for k in range(K):
    for i in range(n):
      somatorio = 0
      for j in range(p):
        somatorio +=  alfa[k,j] * distancia_arr[j][i,obj_k_min[k]]
      J += somatorio * (Gp2[k,i] **m)  #Cálculo da função objetivo

  while (er < abs(J - J_2) or T > counter ):    #Condição de parada das iterações
    obj_k_min = []
    for k in range(K):
      arg_min_k = 999999
      index_min_k = 0
      for h in range(n):
        sum1 = 0.0
        for i in range(n):
          sum2 = 0.0 
          for j in range(p):
            sum2 += alfa[k,j] * distancia_arr[j][i,h]
          sum1 += (Gp2[k,i]**m) * sum2
        if (arg_min_k > sum1 and (h not in obj_k_min)):
          arg_min_k = sum1
          index_min_k = h
      obj_k_min.append(index_min_k)    

    for k in range(K):
      for j in range(p):
        produtorio = 1
        for h in range(p):
          somatorio = 0
          somatorio_2 = 0
          for i in range(n):
            somatorio += (Gp2[k,i]**m) * distancia_arr[h][i,obj_k_min[k]]
            somatorio_2 += (Gp2[k,i]**m) * distancia_arr[j][i,obj_k_min[k]]
          produtorio *= somatorio
        produtorio = (produtorio**(1/p)) / somatorio_2
        alfa[k,j] = produtorio     #Cálculo das matrizes de pesos

    for k in range(K):
      for i in range(n):
        sum_geral = 0
        for h in range(K):
          sum_arr_1 = 0
          sum_arr_2 = 0
          for j in range(p):
            sum_arr_1 += alfa[k,j] * distancia_arr[j][i,obj_k_min[k]]
            sum_arr_2 += alfa[h,j] * distancia_arr[j][i,obj_k_min[h]]
          if (sum_arr_2 != 0):
            sum_geral += (sum_arr_1 / sum_arr_2)**coef
          if (sum_geral == 0):
            Gp2[k,i] = 1
          else:
            Gp2[k,i] = sum_geral**-1     
    
    for i in range(n):
      for k in range(K):
        if (Gp2[k,i] == 1):
          for j in range(k):
            if(j != k):
              Gp2[j,i] = 0
    
    J_2 = 0
    for k in range(K):
      for i in range(n):
        somatorio = 0
        for j in range(p):
          somatorio +=  alfa[k,j] * distancia_arr[j][i,obj_k_min[k]]
        J_2 += somatorio * (Gp2[k,i] **m)
    J = J_2
    counter += 1
    print(counter)
  if (J < j2_min):      #Condição para pegar a menor função objetiva de todas.
    j2_min = J  
    gp2_min = Gp2
    obj_k_min_global = obj_k_min   

  print(iteration)
  iteration += 1 

# fim do while ////////////////////////////////////////


#Criação da partição Crisp
arr = np.zeros((n,1))

for j in range(n):
  max_arr = 0
  index_max_value = 0
  for i in range(K):
    if ( gp2_min[i,j] > max_arr):
      max_arr = gp2_min[i,j]
      index_max_value = i
  arr[j] = index_max_value


array_of_array = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]  #Divisão dos grupos da partição crisp

for obj in range(10):
  for x in range(2000):
    if(arr[x] == obj):
      array_of_array[obj].append(x)

aux_array=np.zeros((10,10))
for i in range(10):
  for j in range(len(array_of_array[i])):

    aux_array[i][math.floor(array_of_array[i][j]/200)] += 1


array_real = np.zeros((10,1))
for i in range (10):
  array_real[i] = np.argmax(aux_array[i])

#Cálculo do erro
tot=0
for i in range(n):
   a=floor(i/200)
   if array_real[int(arr[i])] == a: 
       tot+=0
   else:
        tot+=1
erro=(tot/2000)*100

#Cálculo do indice de Rand
sum1 = 0
for i in range(10):
  for j in range(10):
    sum1 += (aux_array[i][j]*(aux_array[i][j] -1))/ 2 
sum_a = 0
for i in range(10):
  sum_a += (len(array_of_array[i]) * (len(array_of_array[i]) -1) )/2

sum_b = ((200 * 199)/2) * 10
binomial_n2 = 2000*1999/2
i_rand = (sum1 - (( sum_a * sum_b) / binomial_n2)) / (0.5 *( sum_a + sum_b) - ((sum_a * sum_b)/binomial_n2))

#Cálculo do F-measure
f_measure = 0
for j in range(10):
  max_i = []
  for i in range(10):
    if aux_array[i][j] != 0:
     max_i.append((  (aux_array[i][j] * aux_array[i][j]) / (len(array_of_array[i]) * 200)) / ((aux_array[i][j]/len(array_of_array[i])) + (aux_array[i][j]/200)) )
  f_measure += 200/2000 * np.amax(max_i)
print(f_measure)

Sum=0
Sum2=0
Vpc=0

#Partition Coef
for k in range(K):
   for i in range(n):
      Sum +=(gp2_min[k,i])**2
Vpc=(1/n)*Sum
      
#Modifi Part
Vmpc= 1-((K/(K-1))*(1-Vpc))

#Partition Entropy
for k in range(K):
   for i in range(n):
      if (gp2_min[k,i]!=0):
        Sum2 += (gp2_min[k,i])*(math.log(gp2_min[k,i]))
Vpe=(-1/n)*Sum2

df1['classe'] = arr
df2['classe'] = arr
df3['classe'] = arr

#Criação dos novos arquivos com a coluna classe
df1.to_csv('new_fac.csv') 
df2.to_csv('new_fou.csv')
df3.to_csv('new_kar.csv')

print('Grau de partição:',gp2_min)
print('Prototipos finais:',obj_k_min_global)
print('Partição fuzzy:', array_of_array)
for i in range(10):
  print('Tamanho do grupo: ' + str(i) ,len(array_of_array[i]))

print('Modified partition coeficient:', Vmpc)
print('Partition entropy:', Vpe)
print('Erro:', erro)
print('F_measure:', f_measure)
print('Indice de Rand:',i_rand)
