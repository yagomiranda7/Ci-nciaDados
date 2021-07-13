#Importação das bibliotecas
import numpy as np
from sklearn import preprocessing
import pandas as pd
import math 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from statistics import stdev
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin

#Leitura dos arquivos
data_fac = pd.read_csv('./new_fac.csv', delimiter= ',', header=0, index_col=0)
data_fou = pd.read_csv('./new_fou.csv', delimiter= ',', header=0, index_col=0)
data_kar = pd.read_csv('./new_kar.csv', delimiter= ',', header=0, index_col=0)


#Divisão e Normalização dos arquvos em features e classe

df_fac = pd.DataFrame(data_fac)
features_1 = np.array(df_fac.iloc[:,0:216])
classe_1 = np.array(df_fac['classe'])

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled1 = min_max_scaler.fit_transform(features_1)
df_normalized_1 = pd.DataFrame(np_scaled1)

df_fou = pd.DataFrame(data_fou)
features_2 = np.array(df_fou.iloc[:,0:76])
classe_2 = np.array(df_fou['classe'])

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled2 = min_max_scaler.fit_transform(features_2)
df_normalized_2 = pd.DataFrame(np_scaled2)

df_fac = pd.DataFrame(data_kar)
features_3 = np.array(df_fac.iloc[:,0:64])
classe_3 = np.array(df_fac['classe'])

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled3 = min_max_scaler.fit_transform(features_3)
df_normalized_3 = pd.DataFrame(np_scaled3)

acc_arr = []
acc_arr_gauss = []
acc_arr_parzen = []

score_gauss = 0
score_parzen = 0
score_knn = 0
N = 30
L=3

#Divisão das features e classe em treino e teste
for index_run in range(N):
    df_normalized_1_train, df_normalized_1_test, classe_1_train, classe_1_test = train_test_split( 
                 df_normalized_1, classe_1, test_size = 0.20, random_state=None) 
    
    df_normalized_2_train, df_normalized_2_test, classe_2_train, classe_2_test = train_test_split( 
                 df_normalized_2, classe_2, test_size = 0.20, random_state=None) 
    
    df_normalized_3_train, df_normalized_3_test, classe_3_train, classe_3_test = train_test_split( 
                 df_normalized_3, classe_3, test_size = 0.20, random_state=None) 
    
   ############################## KNN ##################################
    
   #Uso da validação para obtero melhor K de cada arquivo
    test_accuracy_1 = np.empty(50) 
    test_accuracy_2 = np.empty(50) 
    test_accuracy_3 = np.empty(50) 
    for i in range(1,50): 
        knn = KNeighborsClassifier(n_neighbors=i) 
        knn.fit(df_normalized_1_train,classe_1_train)
        test_accuracy_1[i] = knn.score(df_normalized_1_test, classe_1_test)
        
        knn = KNeighborsClassifier(n_neighbors=i) 
        knn.fit(df_normalized_2_train,classe_2_train)
        test_accuracy_2[i] = knn.score(df_normalized_2_test, classe_2_test)
        
        knn = KNeighborsClassifier(n_neighbors=i) 
        knn.fit(df_normalized_3_train,classe_3_train)
        test_accuracy_3[i] = knn.score(df_normalized_3_test, classe_3_test)
    
#Cálculo das predição probabilisticas já com o melhor K
    knn = KNeighborsClassifier(n_neighbors=np.argmax(test_accuracy_1)+1) 
    pred_1 = cross_val_predict(knn, df_normalized_1, classe_1, cv=10, method='predict_proba')
    
    knn = KNeighborsClassifier(n_neighbors=np.argmax(test_accuracy_2)+1) 
    pred_2 = cross_val_predict(knn, df_normalized_2, classe_2, cv=10, method='predict_proba')
    
    knn = KNeighborsClassifier(n_neighbors=np.argmax(test_accuracy_3)+1) 
    pred_3 = cross_val_predict(knn, df_normalized_3, classe_3, cv=10, method='predict_proba')
    
    #Cáculo da probabilidade a priori
    priori=np.zeros(10, dtype=np.float64)
    for t in range(10):
      for i in range(2000):
         if classe_1[i]==t:
           priori[t]+= 1
           
    #Uso da equação da regra da soma proposta
    val= np.zeros((2000,10), dtype=np.float64)
    for i in range(2000):
        for j in range(10):
            val[i,j]=((1-L)*(priori[j]/2000)+pred_1[i,j]+pred_2[i,j]+pred_3[i,j])
            
    classe_final = np.zeros((2000,1))
    for i in range(2000):
        classe_final[i] = np.argmax(val[i])
    #Cálculo da acurácia da de cada iteração
    acc = 0 
    for i in range(2000):  
      if (classe_1[i] == classe_final[i]):
          acc +=1
    acc = acc/2000
    acc_arr.append(acc)
    
    ################################ KNN #################
    
    
    ################################ GAUSS #################
    gauss = GaussianNB() 
    k_fold_1 = KFold(n_splits=10, shuffle=True, random_state=None)
    pred_1_gauss = cross_val_predict(gauss, df_normalized_1, classe_1, cv=k_fold_1, method='predict_proba')
    
    gauss = GaussianNB() 
    k_fold_2 = KFold(n_splits=10, shuffle=True, random_state=None)
    pred_2_gauss = cross_val_predict(gauss, df_normalized_2, classe_2, cv=k_fold_2, method='predict_proba')
    
    gauss = GaussianNB()
    k_fold_3 = KFold(n_splits=10, shuffle=True, random_state=None)
    pred_3_gauss = cross_val_predict(gauss, df_normalized_3, classe_3, cv=k_fold_3, method='predict_proba')
    
    
    val_gauss= np.zeros((2000,10), dtype=np.float64)
    for i in range(2000):
        for j in range(10):
            val_gauss[i,j]=((1-L)*(priori[j]/2000)+pred_1_gauss[i,j]+pred_2_gauss[i,j]+pred_3_gauss[i,j])
            
    classe_final_gauss = np.zeros((2000,1))
    for i in range(2000):
        classe_final_gauss[i] = np.argmax(val_gauss[i])
    
    acc = 0 
    for i in range(2000):  
      if (classe_1[i] == classe_final_gauss[i]):
          acc +=1
    acc = acc/2000
    acc_arr_gauss.append(acc)
    ################################GAUS#################
    
    
    ################################PARZEN################
    
    class KDEClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, bandwidth=1.0, kernel='gaussian'):
            self.bandwidth = bandwidth
            self.kernel = kernel
            
        def fit(self, X, y):
            self.classes_ = np.sort(np.unique(y))
            training_sets = [X[y == yi] for yi in self.classes_]
            self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                          kernel=self.kernel).fit(Xi)
                            for Xi in training_sets]
            #self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
            #                  for Xi in training_sets]
            return self
            
        def predict_proba(self, X):
            logprobs = np.array([model.score_samples(X)
                                 for model in self.models_]).T
            result = np.exp(logprobs)
            return result / result.sum(1, keepdims=True)
              
        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), 1)]
        
      #Uso da validação para obtero melhor H de cada arquivo  
    def best_bandwith(X,classe_train, classe_test, base_test):
        best_band_index = 0
        best_band_acc = 0
        band = 0
        arr = []
        while ( band <= 1):
            band += 0.1       
            kde = KDEClassifier(bandwidth=band).fit(X, classe_train)
            c = kde.predict(base_test)
            acc_parsen = 0 
            for i in range(400):  
              if (classe_test[i] == c[i]):
                  acc_parsen +=1
            acc_parsen = acc_parsen/400
            arr.append(acc_parsen)
            if (acc_parsen > best_band_acc):
                best_band_acc = acc_parsen
                best_band_index = band 
                                
        return best_band_index, best_band_acc, arr
    

    c, best_acc_1,acc_parsen_1 = best_bandwith(df_normalized_1_train,classe_1_train ,classe_1_test, df_normalized_1_test)
       
    c_2, best_acc_2,acc_parsen_2 = best_bandwith( df_normalized_2_train,classe_2_train, classe_2_test,df_normalized_2_test)
       
    c_3, best_acc_3, acc_parsen_3= best_bandwith(df_normalized_3_train,classe_3_train, classe_3_test,df_normalized_3_test)
    
    #Uso do classificador já com o melhor H
    parzen = KDEClassifier(bandwidth=c) 
    k_fold_parzen = KFold(n_splits=10, shuffle=True, random_state=None)
    pred_1_parzen = cross_val_predict(parzen, df_normalized_1, classe_1, cv=k_fold_parzen, method='predict_proba')
    
    parzen = KDEClassifier(bandwidth=c_2) 
    k_fold_parzen = KFold(n_splits=10, shuffle=True, random_state=None)
    pred_2_parzen = cross_val_predict(parzen, df_normalized_2, classe_2, cv=k_fold_parzen, method='predict_proba')
    
    parzen = KDEClassifier(bandwidth=c_3) 
    k_fold_parzen = KFold(n_splits=10, shuffle=True, random_state=None)
    pred_3_parzen = cross_val_predict(parzen, df_normalized_3, classe_3, cv=k_fold_parzen, method='predict_proba')
        
    
    val_parzen = np.zeros((2000,10), dtype=np.float64)
    for i in range(2000):
        for j in range(10):
            val_parzen[i,j]=((1-L)*(priori[j]/2000)+pred_1_parzen[i,j]+pred_2_parzen[i,j]+pred_3_parzen[i,j])
            
    classe_final_parzen = np.zeros((2000,1))
    for i in range(2000):
        classe_final_parzen[i] = np.argmax(val_parzen[i])
    
    acc = 0 
    for i in range(2000):  
      if (classe_1[i] == classe_final_parzen[i]):
          acc +=1
    acc = acc/2000
    acc_arr_parzen.append(acc)
    
   ################################PARZEN################  
   
   #Cálculo dos scores dos algoritmos, levando em conta as acurácias em cada iteração
    if(acc_arr_parzen[index_run] > acc_arr_gauss[index_run]  and  acc_arr_parzen[index_run] > acc_arr[index_run] ):
        score_parzen +=1 
        if(acc_arr_gauss[index_run] > acc_arr[index_run]): 
            score_gauss +=2
            score_knn +=3
        elif(acc_arr_gauss[index_run] == acc_arr[index_run]): 
            score_gauss +=2.5
            score_knn +=2.5
        else:
            score_gauss +=3
            score_knn +=2
            
    elif(acc_arr_gauss[index_run] > acc_arr_parzen[index_run]  and  acc_arr_gauss[index_run] > acc_arr[index_run] ):
        score_gauss +=1 
        if(acc_arr_parzen[index_run] > acc_arr[index_run]): 
            score_parzen +=2
            score_knn +=3
        elif(acc_arr_parzen[index_run] == acc_arr[index_run]): 
            score_parzen +=2.5
            score_knn +=2.5
        else:
            score_parzen +=3
            score_knn +=2
    elif(acc_arr[index_run] > acc_arr_parzen[index_run]  and  acc_arr[index_run] > acc_arr_gauss[index_run] ):
        score_knn +=1 
        if(acc_arr_parzen[index_run] > acc_arr_gauss[index_run]): 
            score_parzen +=2
            score_gauss +=3
        elif(acc_arr_parzen[index_run] == acc_arr_gauss[index_run]): 
            score_parzen +=2.5
            score_gauss +=2.5
        else:
            score_parzen +=3
            score_gauss +=2
    
     ################################  LETRA B  ################ 

#Cálculo do intervalo pontual da taxa de acerto para cada classificador
def intervalo_confianca(data, confianca=0.05):
    desvio_padrao = stdev(data)
    media = np.mean(data)
    n = len(data)
    h = desvio_padrao * 1.96/math.sqrt(n)
    return media - h, media + h

knn_intervalo_negativo,knn_intervalo_positivo = intervalo_confianca(acc_arr)
print("O intervalo de acerto para o KNN com intervalo de confiança de 95% será: "+"{:.4f}".format(knn_intervalo_negativo)+ "% <= acerto <= "+"{:.4f}".format(knn_intervalo_positivo)+"%")

gauss_intervalo_negativo,gauss_intervalo_positivo = intervalo_confianca(acc_arr_gauss)
print("O intervalo de acerto para o Gaussian NB com intervalo de confiança de 95% será: "+"{:.4f}".format(gauss_intervalo_negativo)+ "% <= acerto <= "+"{:.4f}".format(gauss_intervalo_positivo)+"%")

parzen_intervalo_negativo,parzen_intervalo_positivo = intervalo_confianca(acc_arr_parzen)
print("O intervalo de acerto para a janela de parzen com intervalo de confiança de 95% será: "+"{:.4f}".format(parzen_intervalo_negativo)+ "% <= acerto <= "+"{:.4f}".format(parzen_intervalo_positivo)+"%")
    ################################  LETRA B  ################ 
  
    ################################  LETRA C  ################ 
score_gauss = score_gauss /N
score_parzen = score_parzen /N
score_knn = score_knn /N
 
#Uso do teste de Friedman para testar a hipótese
xf = ((12*N)* ((score_parzen**2) +( score_gauss**2) + (score_knn**2) - (L*((L+1)**2)/4 )))/(L * (L+1)) 
ff = ((N-1) * xf)/((N*(L-1) -xf))
ff_tab = 3.158    #Tabela F com alfa=0.05 e F(2,58)

score_arr = [score_parzen, score_knn, score_gauss]
score_arr.sort(reverse=True)
print('A hipótese nula é que não existe diferenças significativas entre as acurácias dos classificadores ')
if (ff > ff_tab):
    print('Hipótese nula rejeitada pelo teste de hipotése com alfa=0.05')
else:
    print('O teste de Hipotóse falhou em rejeitar a Hipótese Nula')
    
w = 2.343 * math.sqrt(L*(L+1)/(6*N))

if(w > (score_arr[0] - score_arr[2])):
        print('O pós teste não é poderoso o suficiente para detectar diferenças significativas entre os classificadores com alfa igual a 5%')
else:
        print('O pós teste pôde identificar dois grupos de classificadores com diferenças significativas entre si')