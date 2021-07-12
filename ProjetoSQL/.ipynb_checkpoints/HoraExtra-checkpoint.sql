DROP TABLE IF EXISTS  #BSE_VALOR, #BSE_MESES

SELECT * 
INTO #BSE_VALOR
FROM (
SELECT B.Nome, TP2.*, ROW_NUMBER()
OVER(PARTITION BY TP2.MATRICULA ORDER BY TP2.MATRICULA) AS RN
FROM (
SELECT TP0.Matricula, SUM(Valor_rubrica) as Valor_Total
FROM (
SELECT Nome, Matricula, Competencia, valor_rubrica, rubrica
FROM BSE_FLHA_PGMNTO2
where rubrica = '55' or rubrica='56') TP0
GROUP BY Matricula)TP2
INNER JOIN BSE_FLHA_PGMNTO2 B
ON TP2.Matricula = B.Matricula) TP3
WHERE RN=1



SELECT TP2.Matricula, count(Meses) as Total_Meses
into #BSE_MESES
FROM (
SELECT TP1.*,
CASE
  WHEN qnt_meses =2
    THEN 1
  ELSE 1
  END Meses
FROM (
SELECT Matricula,Competencia, count(*) as qnt_meses
FROM BSE_FLHA_PGMNTO2
where rubrica = '55' or rubrica='56'
group by Matricula,Competencia) TP1 
)TP2
group by Matricula

SELECT a.NOME, A.Matricula, A.Valor_Total, B.Total_Meses
FROM #BSE_VALOR A
INNER JOIN #BSE_MESES B
ON A.Matricula = B.Matricula
ORDER BY Valor_Total


