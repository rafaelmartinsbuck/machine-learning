# Project: Identify Fraud from Enron Email
### Rafael Buck


## Sobre a Enron

Em 2000, Enron era uma das maiores empresas dos Estados Unidos. Já em 2002, ela colapsou e quebrou devido a uma fraude que envolveu 
grande parte da corporação. Resultando em uma investigação federal, muitos dados que são normalmente confidenciais, se tornaram públicos, 
incluindo dezenas de milhares de e-mails e detalhes financeiros para os executivos dos mais altos níveis da empresa. Neste projeto, 
iremos bancar detetives, e colocar nossas habilidades na construção de um modelo preditivo que visará determinar se um funcionário é 
ou não um funcionário de interesse (POI). Um funcionário de interesse é um funcionário que participou do escândalo da empresa Enron. 
Os dados financeiros e sobre e-mails dos funcionários investigados neste caso de fraude já foram previamente 
combinados (arquivo `"final_project_dataset.pkl"`), o que significa que eles foram indiciados, fecharam acordos com o governo, 
ou testemunharam em troca de imunidade no processo.

## Instruções

Como está sendo utilizado o Python 3.5, foram necessárias os seguintes ajustes nos códigos de `"tester.py"`:

    print("some text") # No lugar de  print "some text"
    open(file_name, "rb") # No lugar de open(file_name, "r")
    open(file_name, "wb") # No lugar de open(file_name, "w")
    clf.fit(np.array(features_train), np.array(labels_train)) # Para a rotina funcionar com o XGBoost

Para executar o projeto, portanto:
- possuir Python 3.5 ou superior instalado
- colocar todos os arquivos em um mesmo diretório (inclusive o dataset `"final_project_dataset.pkl"`)
- executar o comando `"python poi_id.py"`

## Objetivo do trabalho e estratégia de análise

O objetivo desse projeto é criar um modelo que, dadas certas características financeiras e não-financeiras (*features*), seja capaz de 
prever com precisão se um funcionário é uma *Person fo Interest* (POI) de uma fraude (*target*). Nesse projeto vamos lidar com um problema 
de classificação usando *Machine Learning* e aprendizagem supervisionada, pois as POIs estão sendo agrupados em duas categorias: aqueles 
que temos um funcionário que é um POI (valor `1.0`) e aqueles que não o são (valor `0.0`).

Como etapa de pré-processamento deste projeto, foram combinados os dados da base "Enron email and financial" em um dicionário, onde cada 
par chave-valor corresponde a uma pessoa. A chave do dicionário é o nome da pessoa, e o valor é outro dicionário, que contém o nome de 
todos os atributos e seus valores para aquela pessoa. Os atributos nos dados possuem basicamente três tipos: atributos financeiros, de 
email e rótulos POI (pessoa de interesse).

- **atributos financeiros**: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (todos em dólares americanos (USD))

- **atributos de email**: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (as unidades aqui são geralmente em número de emails; a exceção notável aqui é o atributo ‘email_address’, que é uma string)

- **rótulo POI**: [‘poi’] (atributo objetivo lógico (booleano), representado como um inteiro)

#### Validação

O melhor estimador para o AdaBoost (melhor classificador para o problema) foi:

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, criterion='gini'), 
                             n_estimators=50, 
                             learning_rate=.8)
                             
Os resultados atingidos por esse modelo final foram:
    
    Accuracy: 0.81620	
    Precision: 0.33050	
    Recall: 0.36900	
    F1: 0.34869	
    F2: 0.36060
	Total predictions: 15000	
    True positives:  738	False positives: 1495	False negatives: 1262	True negatives: 11505
    
Nele conseguimos um alto *Recall* e os scores F1 e F2, otimizando a identificação de POIs.
