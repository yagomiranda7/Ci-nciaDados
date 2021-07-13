#Projeto 0

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

# Troca do Nan por vazio(''), para que aparece nos agrupamentos
dftotal[['school_mat','school_port']]=dftotal[['school_mat','school_port']].fillna('Nao faz')

#Criação do DataFarame dos excluídos
dfexc = dftotal[(dftotal['_merge']=='left_only') | (dftotal['_merge']=='right_only')]

#Cálculo da quantidade de alunos não mesclados por disciplina

index_exc_mat = dfexc[dfexc['_merge']=='left_only'].index
number_exc_mat = len(index_exc_mat)

index_exc_port = dfexc[dfexc['_merge']=='right_only'].index
number_exc_port = len(index_exc_port)

total_exc = number_exc_mat+number_exc_port

#Agrupamento da base excluída pelas escolas e separação por escola
dfexc_byschool = dfexc.groupby(['school_mat','school_port']).size().reset_index(name='Contagem')
exc_GP = dfexc_byschool[(dfexc_byschool['school_mat']=='GP') | (dfexc_byschool['school_port']=='GP')]
exc_MS = dfexc_byschool[(dfexc_byschool['school_mat']=='MS') | (dfexc_byschool['school_port']=='MS')]

#Total de alunos excluídos por escola
total_gp = exc_GP['Contagem'].sum()
total_ms = exc_MS['Contagem'].sum()

#Criação do DataFarame apenas dos mesclados 
dfmesc = dftotal[dftotal['_merge']=='both']

#Cálculo da quantidade de alunos mesclados
index_mesc = dfmesc.index
number_mesc = len(index_mesc)




#df3 = dftotal[(dftotal['schoolsup_mat']=='yes') | (dftotal['schoolsup_port']=='yes')].groupby(['school_port','school_mat']).size().reset_index(name='Total')

print('O número total de alunos(as) que só faz uma das duas disciplinas é: ' + str(total_exc))
print('O número total de alunos(as) que faz apenas matemática é: '+ str(number_exc_mat))
print('O número total de alunos(as) que faz apenas portugês é: '+ str(number_exc_port))
print('O número total de alunos(as) que só faz uma das duas disciplinas no colégio GP é: '+ str(total_gp))
print('O número total de alunos(as) que só faz uma das duas disciplinas no colégio MS é: '+ str(total_ms))
print('O número total de alunos(as) que faz as duas disciplinas é: '+ str(number_mesc))
