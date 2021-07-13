#Projeto 3


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

#Preenchimento de colunas  NaN com frase específica
dftotal[['schoolsup_mat','schoolsup_port']]=dftotal[['schoolsup_mat','schoolsup_port']].fillna('Nao faz essa disciplina')
dftotal[['school_mat','school_port']]=dftotal[['school_mat','school_port']].fillna('Nao faz')


#Filtro Solicitado
df3 = dftotal[(dftotal['schoolsup_mat']=='yes') | (dftotal['schoolsup_port']=='yes')].groupby(['school_port','school_mat']).size().reset_index(name='Total')

#Quantiade de alunos por escola
dfschool_GP = dftotal[(dftotal['school_mat']=='GP') | (dftotal['school_port']=='GP')].groupby(['school_mat','school_port']).size().reset_index(name='Total')
dfschool_MS = dftotal[(dftotal['school_mat']=='MS') | (dftotal['school_port']=='MS')].groupby(['school_mat','school_port']).size().reset_index(name='Total')

qtd_GP = dfschool_GP['Total'].sum()
qtd_MS = dfschool_MS['Total'].sum()


#DataFrame feito 
df3_GP=df3[(df3['school_port']=='GP') | (df3['school_mat']=='GP')]
df3_MS=df3[(df3['school_port']=='MS') | (df3['school_mat']=='MS')]

#Quantidade de alunos que tiveram ajuda extra-curricular por colégio
qtd_GP3=df3_GP['Total'].sum()
qtd_MS3=df3_MS['Total'].sum()

# Porcentagem de Alunos que tiveram ajuda extra-curricular por colégio
pct_GP3=round(qtd_GP3*100/qtd_GP,2)
pct_MS3=round(qtd_MS3*100/qtd_MS,2)

print('A porcentagem de alunos que tiveram ajuda extra curriular no colégio GP foi de: ' + str(pct_GP3)+' %')
print('A porcentagem de alunos que tiveram ajuda extra curriular no colégio MS foi de: ' + str(pct_MS3)+' %')


