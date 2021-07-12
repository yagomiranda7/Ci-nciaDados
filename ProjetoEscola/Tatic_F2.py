#Projeto 2

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

#Filtro solicitado
df2 = dftotal[(((dftotal['G1_mat'] < 6) & (dftotal['G2_mat'] < 6)) | ((dftotal['G1_port'] < 6) & (dftotal['G2_port'] < 6))) & ((dftotal['Mjob_mat']=='teacher') | (dftotal['Mjob_port']=='teacher')) ]

index_df2 = df2.index
number_df2 = len(index_df2)

print('A quantidade de alunos que tiraram menos que 6 no G1 e G2 em uma das duas disciplinas e tem a mãe como professora é: '+str(number_df2))

df2.to_csv('Alunos_F2.csv', sep=';', index=False)
print('O arquivo csv com os ' + str(number_df2) +' alunos foi gerado')
