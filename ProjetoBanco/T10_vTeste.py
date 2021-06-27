import pandas as pd
import numpy as np


#Início da Balancete

dfbalan=pd.read_csv(r'C:\Users\YAGUIAR\Desktop\Balancete_Teste.csv', header=0, sep=';')

dfbalan['Conta']= dfbalan['Conta'].str.replace('.', '')
dfbalan['Conta']= dfbalan['Conta'].str.replace('-', '')
dfbalan['Saldo']= dfbalan['Saldo'].str.replace(',', '.')


dfbalan['Saldo']=dfbalan['Saldo'].astype(float)

dfbalan['ContaR']=dfbalan['Conta'].apply(lambda x: x[0:4])

dfbalan['Competencia2']=dfbalan['Competencia'].apply(lambda x: x[6:10]+'-'+x[3:5])
dfbalan['Chave']=dfbalan['Rating']+dfbalan['Competencia2']
dfbalan2=dfbalan.groupby('Chave')['Saldo'].sum().reset_index()

#Início da CADOC

dftotal=pd.DataFrame()

df = pd.read_csv(r'C:/Users/YAGUIAR/Desktop/BACEN_3040_20210403213904_1/04MAI2021_2.csv', chunksize=100000, dtype='object', sep='|', header=0, usecols=['Venc_v110','Venc_v120','Venc_v130','Venc_v140','Venc_v150','Venc_v160',
    'Venc_v165','Venc_v170','Venc_v175','Venc_v180','Venc_v190','Venc_v199','Venc_v205','Venc_v210','Venc_v220',
    'Venc_v230', 'Venc_v240', 'Venc_v245','Venc_v250','Venc_v255','Venc_v260','Venc_v270','Venc_v280',
    'Venc_v290','Venc_v310','Venc_v320','Venc_v330','Cli_Cd','Op_Contrt','Op_Mod','Op_ProvConsttd','Doc3040_DtBase','Op_ClassOp','Op_DiaAtraso','Op_NatuOp','Op_CaracEspecial']) 
for f in df:
    df = pd.DataFrame(f)
    
    df['Chave'] = df['Cli_Cd']+df['Op_Contrt']+df['Op_Mod']+df['Doc3040_DtBase']
    df = df.drop_duplicates(subset=['Chave'])
    
    df['Op_Mod_2']=df['Op_Mod'].apply(lambda x: x[0:2])
                                                                                                                                       
    df=df[df['Op_ClassOp']!='HH']  
    
    df[['Venc_v110','Venc_v120','Venc_v130','Venc_v140','Venc_v150','Venc_v160',
    'Venc_v165','Venc_v170','Venc_v175','Venc_v180','Venc_v190','Venc_v199','Venc_v205','Venc_v210','Venc_v220',
    'Venc_v230', 'Venc_v240', 'Venc_v245','Venc_v250','Venc_v255','Venc_v260','Venc_v270','Venc_v280',
    'Venc_v290','Venc_v310','Venc_v320','Venc_v330']] = df[['Venc_v110','Venc_v120','Venc_v130','Venc_v140','Venc_v150','Venc_v160',
    'Venc_v165','Venc_v170','Venc_v175','Venc_v180','Venc_v190','Venc_v199','Venc_v205','Venc_v210','Venc_v220',
    'Venc_v230', 'Venc_v240', 'Venc_v245','Venc_v250','Venc_v255','Venc_v260','Venc_v270','Venc_v280',
    'Venc_v290','Venc_v310','Venc_v320','Venc_v330']].fillna(0)
                                                            
    df[['Venc_v110','Venc_v120','Venc_v130','Venc_v140','Venc_v150','Venc_v160',
    'Venc_v165','Venc_v170','Venc_v175','Venc_v180','Venc_v190','Venc_v199','Venc_v205','Venc_v210','Venc_v220',
    'Venc_v230', 'Venc_v240', 'Venc_v245','Venc_v250','Venc_v255','Venc_v260','Venc_v270','Venc_v280',
    'Venc_v290','Venc_v310','Venc_v320','Venc_v330']] = df[['Venc_v110','Venc_v120','Venc_v130','Venc_v140','Venc_v150','Venc_v160',
    'Venc_v165','Venc_v170','Venc_v175','Venc_v180','Venc_v190','Venc_v199','Venc_v205','Venc_v210','Venc_v220',
    'Venc_v230', 'Venc_v240', 'Venc_v245','Venc_v250','Venc_v255','Venc_v260','Venc_v270','Venc_v280',
    'Venc_v290','Venc_v310','Venc_v320','Venc_v330']].astype(float)
                                                            
    df['Saldo_a_Vencer'] = df['Venc_v110']+df['Venc_v120']+df['Venc_v130']+df['Venc_v140']+df['Venc_v150']
    +df['Venc_v160']+df['Venc_v165']+df['Venc_v170']+df['Venc_v175']+df['Venc_v180']+df['Venc_v190']+df['Venc_v199']
    
    df['Saldo_Vencido']= df['Venc_v205']+df['Venc_v210']+df['Venc_v220']+df['Venc_v230']+df['Venc_v240']+df['Venc_v245']
    +df['Venc_v250']+df['Venc_v255']+df['Venc_v260']+df['Venc_v270']+df['Venc_v280']+df['Venc_v290']
    
    df2 = df.groupby(['Doc3040_DtBase', 'Op_ClassOp'])['Saldo_a_Vencer','Saldo_Vencido'].sum().reset_index()
    
    df2['Saldo_Devedor_Total'] = df2['Saldo_a_Vencer']+df2['Saldo_Vencido']
         
    dftotal=pd.concat([dftotal, df2], axis=0)
    
conditionlist = [
        (dftotal['Op_ClassOp'] == 'AA'),
        (dftotal['Op_ClassOp'] == 'A'),
        (dftotal['Op_ClassOp'] == 'B'),
        (dftotal['Op_ClassOp'] == 'C'),
        (dftotal['Op_ClassOp'] == 'D'),
        (dftotal['Op_ClassOp'] == 'E'),
        (dftotal['Op_ClassOp'] == 'F'),
        (dftotal['Op_ClassOp'] == 'G'),
        (dftotal['Op_ClassOp'] == 'H') ]
choicelist = ['3111', '3121', '3131','3141','3151','3161','3171','3181','3191']
dftotal['Conta_Desc'] = np.select(conditionlist, choicelist, default='Not Specified')
    
conditionlist2 = [
        (dftotal['Op_ClassOp'] == 'AA'),
        (dftotal['Op_ClassOp'] == 'A'),
        (dftotal['Op_ClassOp'] == 'B'),
        (dftotal['Op_ClassOp'] == 'C'),
        (dftotal['Op_ClassOp'] == 'D'),
        (dftotal['Op_ClassOp'] == 'E'),
        (dftotal['Op_ClassOp'] == 'F'),
        (dftotal['Op_ClassOp'] == 'G'),
        (dftotal['Op_ClassOp'] == 'H') ]
choicelist2 = ['3.1.1.10.00.0', '3.1.2.10.00.3', '3.1.3.10.00.6','3.1.4.10.00.9','3.1.5.10.00.2','3.1.6.10.00.5','3.1.7.10.00.8','3.1.8.10.00.1','3.1.9.10.00.4']
dftotal['Conta'] = np.select(conditionlist2, choicelist2, default='Not Specified')

conditionlist3 = [
        (dftotal['Conta_Desc'] == '3111'),
        (dftotal['Conta_Desc'] == '3121'),
        (dftotal['Conta_Desc'] == '3131'),
        (dftotal['Conta_Desc'] == '3141'),
        (dftotal['Conta_Desc'] == '3151'),
        (dftotal['Conta_Desc'] == '3161'),
        (dftotal['Conta_Desc'] == '3171'),
        (dftotal['Conta_Desc'] == '3181'),
        (dftotal['Conta_Desc'] == '3191') ]
choicelist3 = ['OPERACOES DE CREDITO NIVEL AA', 'OPERACOES DE CREDITO NIVEL A', 'OPERACOES DE CREDITO NIVEL B','OPERACOES DE CREDITO NIVEL C','OPERACOES DE CREDITO NIVEL D','OPERACOES DE CREDITO NIVEL E','OPERACOES DE CREDITO NIVEL F','OPERACOES DE CREDITO NIVEL G','OPERACOES DE CREDITO NIVEL H']
dftotal['Descrição_Conta'] = np.select(conditionlist3, choicelist3, default='Not Specified')


dfgeral=dftotal.groupby(['Doc3040_DtBase','Op_ClassOp','Conta_Desc','Conta','Descrição_Conta'])['Saldo_Devedor_Total'].sum().reset_index()
dfgeral['Chave']=dfgeral['Op_ClassOp']+dfgeral['Doc3040_DtBase']

result = pd.merge(dfbalan2, dfgeral, on=['Chave','Chave'], how='outer')

result['Diferenca_CADOC_Balancete']=result['Saldo_Devedor_Total']-result['Saldo']
result['Diferenca_Saldo_Percentual']=(result['Saldo_Devedor_Total']-result['Saldo'])*100/result['Saldo']
result['Diferença_Acima_5%'] = ['Sim' if x > 5 else 'Não' for x in result['Diferenca_Saldo_Percentual']]

result=result.rename({'Saldo': 'Saldo_Balancete', 'Saldo_Devedor_Total': 'Saldo_Devedor_CADOC'}, axis=1, inplace=False)
result=result.drop(['Conta_Desc', 'Chave', 'Conta'], axis=1, inplace=False)


result=result.round(5)

result[['Saldo_Devedor_CADOC', 'Saldo_Balancete', 'Diferenca_CADOC_Balancete', 'Diferenca_Saldo_Percentual']]=result[['Saldo_Devedor_CADOC', 'Saldo_Balancete', 'Diferenca_CADOC_Balancete', 'Diferenca_Saldo_Percentual']].astype(str)
result = result[['Doc3040_DtBase','Op_ClassOp','Descrição_Conta','Saldo_Devedor_CADOC','Saldo_Balancete','Diferenca_CADOC_Balancete','Diferenca_Saldo_Percentual','Diferença_Acima_5%']]
result.to_csv('TesteTeste.csv',sep = ';', index=False)




    
    
    
    
    