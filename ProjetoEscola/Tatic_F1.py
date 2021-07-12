#Projeto 1

#Importação da biblioteca
import pandas as pd


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

#Cálculo da quantidade de alunos mesclados
dfmesc = dftotal[dftotal['_merge']=='both']

#Filtro solicitado
df1 = dfmesc[(dfmesc['G1_mat'] > 10) & (dfmesc['G1_port'] > 10) & (dfmesc['Mjob_mat']=='teacher')]

index_df1 = df1.index
number_df1 = len(index_df1)

print('A quantidade de alunos que tiraram mais que 10 nas duas disciplinas e tem a mãe como professora é: '+str(number_df1))

df1.to_csv('Alunos_F1.csv', sep=';', index=False)
print('O arquivo csv com os ' + str(number_df1) +' alunos foi gerado')
