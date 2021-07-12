
#Projeto 4


#Importação da biblioteca
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.show()

#Carga dos arquivos
dfmat = pd.read_csv(r'C:\Users\asus\Downloads\data_student-mat.csv', delimiter= ';', header=0, index_col=False)
dfpor = pd.read_csv(r'C:\Users\asus\Downloads\data_student-por.csv', delimiter= ';', header=0, index_col=False)


#Transformação das colunas usadas na chave em string
dfmat[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]] = dfmat[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]].astype(str)
dfpor[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]] = dfpor[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]].astype(str)

#Criação da Chave
dfmat['Chave'] = dfmat["school"]+dfmat["sex"]+dfmat["age"]+dfmat["address"]+ dfmat["famsize"]+dfmat['Pstatus']+dfmat["Medu"]+dfmat['Fedu']+dfmat["Mjob"]+dfmat["Fjob"]+dfmat["reason"]+dfmat["nursery"]+dfmat["internet"]
dfpor['Chave'] = dfpor["school"]+dfpor["sex"]+dfpor["age"]+dfpor["address"]+ dfpor["famsize"]+dfpor['Pstatus']+dfpor["Medu"]+dfpor['Fedu']+dfpor["Mjob"]+dfpor["Fjob"]+dfpor["reason"]+dfpor["nursery"]+dfpor["internet"]


#Criação do DataFrame Total (com full join)
dftotal = pd.merge(dfmat, dfpor, on='Chave', how='outer', suffixes=('_mat','_port'), indicator=True)

#Gráficos de quantidade de alunos por gênero e por escola
sns.catplot(x="sex_mat", kind='count', data=dftotal);
sns.catplot(x="sex_port", kind='count', data=dftotal);

#Gráficos de quantidade de alunos por razão de escolha da escola

sns.catplot(x="reason_mat", kind='count', hue='school_mat', data=dftotal);
sns.catplot(x="reason_port", kind='count', hue='school_port', data=dftotal);

#Gráficos do tamanho da família do aluno e  por escola

sns.catplot(x="famsize_mat", kind='count', hue='school_mat', data=dftotal);
sns.catplot(x="famsize_port", kind='count', hue='school_port', data=dftotal);

#Gráficos de quantidade de consumo de alcóol pelos alunos por escola

sns.catplot(x="Dalc_mat", kind='count', hue='school_mat', data=dftotal);
sns.catplot(x="Dalc_port", kind='count', hue='school_port', data=dftotal);




