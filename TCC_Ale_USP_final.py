#!/usr/bin/env python
# coding: utf-8

# Nome: Alexandre Scarambone - TCC - Customer Segmentation
# 
# Agradecimentos/ Fontes:
# 
# (1) Notebook (Kaggle) por FABIEN DANIEL (DS - França): https://www.kaggle.com/fabiendaniel/customer-segmentation/notebook
# 
# (2) https://practicaldatascience.co.uk/machine-learning/how-to-use-k-means-clustering-for-customer-segmentation

# In[1]:


pip install matplotlib==3.2.0 #para rodar o grafico


# In[2]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import plotly.express as px
from numpy.random import seed
#from tensorflow.random import set_seed

sns.set(rc={'figure.figsize':(15, 6)})

init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Carga dos dados

# In[3]:


#__________________
# read the datafile
df_ecommerce = pd.read_csv('data.csv',encoding="ISO-8859-1",
                         dtype={'Customer ID': str,'Invoice': str})
print('Dataframe dimensions:', df_ecommerce.shape)


# ___
# ## Explorando o conteúdo das variáveis
# 
# Descrição variáveis DF: 
# 
# **InvoiceNo**: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.  <br>
# **StockCode**: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. <br>
# **Description**: Product (item) name. Nominal. <br>
# **Quantity**: The quantities of each product (item) per transaction. Numeric.	<br>
# **InvoiceDate**: Invice Date and time. Numeric, the day and time when each transaction was generated. <br>
# **UnitPrice**: Unit price. Numeric, Product price per unit in sterling. <br>
# **CustomerID**: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. <br>
# **Country**: Country name. Nominal, the name of the country where each customer resides.<br>
# 
# 

# In[4]:


df_ecommerce.head()


# In[5]:


#____________________________________________________________
# gives some infos on columns types and numer of null values
tab_info=pd.DataFrame(df_ecommerce.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df_ecommerce.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df_ecommerce.isnull().sum()/df_ecommerce.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
display(tab_info)


# ### Pré-processamento dos dados e análises iniciais

# ∼ 25% das entradas não são atribuídas a um cliente específico. A partir dos dados disponíveis, é impossível imputar valores para estes clientes e essas entradas são, portanto, inúteis para o nosso propósito. Então, iremos excluir do dataframe:

# In[6]:


df_ecommerce.dropna(axis = 0, subset = ['CustomerID'], inplace = True)
print('Dataframe dimensions:', df_ecommerce.shape)
#____________________________________________________________
# gives some infos on columns types and numer of null values
tab_info=pd.DataFrame(df_ecommerce.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df_ecommerce.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df_ecommerce.isnull().sum()/df_ecommerce.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
display(tab_info)


# In[7]:


df_ecommerce.shape


# Verificando entradas duplicadas

# In[8]:


print('Entradas duplicadas: {}'.format(df_ecommerce.duplicated().sum()))


# In[9]:


dup = df_ecommerce[df_ecommerce.duplicated()].sort_values(by=['CustomerID','InvoiceNo','StockCode'])
dup.head()


# In[10]:


dup.query('CustomerID == 12748 & InvoiceNo == "550320" & StockCode == "22890"')


# In[11]:


df_ecommerce.drop_duplicates(keep='first',inplace = True)
df_ecommerce.query('CustomerID == 12748 & InvoiceNo == "550320" & StockCode == "22890"')


# In[12]:


print('Entradas duplicadas: {}'.format(df_ecommerce.duplicated().sum()))


# In[13]:


df_ecommerce.shape


# Verificando se existem preços menores que 0

# In[14]:


df_ecommerce[df_ecommerce['UnitPrice']<0]


# ---
# **Atributos constantes**
# 
# Verificando se temos atributos constantes.
# 
# Apesar do valor informativo, ao construir modelos computacionais, por exemplo classificadores, esses atributos representam um aumento na dimensionalidade sem contribuir para a tarefa principal, classificação.
# 

# In[15]:


att_const = np.array(df_ecommerce.columns[df_ecommerce.nunique() <= 1])
print(att_const)


# In[16]:


df_ecommerce['InvoiceDate'] = pd.to_datetime(df_ecommerce['InvoiceDate'])


# ### Separação da base de dados
# 
# O dataframe contém informações para um período de 12 meses. Posteriormente, um dos objetivos será desenvolver um modelo capaz de caracterizar e antecipar os hábitos dos clientes que visitam o site e isso, desde a sua primeira visita. Para poder testar o modelo de forma realista, divido o conjunto de dados retendo os primeiros 10 meses para desenvolver o modelo e os dois meses seguintes para testá-lo:

# In[17]:


print(df_ecommerce['InvoiceDate'].min(), '->',  df_ecommerce['InvoiceDate'].max())


# In[18]:


set_treinamento = df_ecommerce[df_ecommerce['InvoiceDate'] < '2011-10-1']
set_teste         = df_ecommerce[df_ecommerce['InvoiceDate'] >= '2011-10-1']
df_ecommerce = set_treinamento.copy(deep = True)


# In[19]:


df_initial = df_ecommerce


# In[20]:


df_initial.shape


# In[21]:


print(set_teste['InvoiceDate'].min(), '->',  set_teste['InvoiceDate'].max())


# In[22]:


print(df_initial['InvoiceDate'].min(), '->',  df_initial['InvoiceDate'].max())


# ### Análises iniciais

# País de origem (transações)

# In[23]:


temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp = temp.reset_index(drop = False)
countries = temp['Country'].value_counts()


# In[24]:


data = dict(type='choropleth',
locations = countries.index,
locationmode = 'country names', z = countries,
text = countries.index, colorbar = {'title':'Num. de pedidos.'},
colorscale=[[0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']],    
reversescale = False)
#_______________________
layout = dict(title='Número de pedidos por país',
geo = dict(showframe = True, projection={'type':'mercator'}))
#______________
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap, validate=False)


# Vemos que o conjunto de dados é amplamente dominado por pedidos feitos do Reino Unido.

# ___
# ### Customers and products
# 
# O DF contêm $\sim$270,000 entradas. Número de usuários e produtos:

# In[25]:


pd.DataFrame([{'produtos': len(df_initial['StockCode'].value_counts()),    
               'transações': len(df_initial['InvoiceNo'].value_counts()),
               'clientes': len(df_initial['CustomerID'].value_counts()),  
              }], columns = ['produtos', 'transações', 'clientes'], index = ['quantidade'])


#  Verifica-se que os dados dizem respeito a 3.658 usuários e que compraram 3.527 produtos diferentes. O número total de transações realizadas é da ordem de $\sim$15.920.
# 
# 

# Número de produtos adquiridos em cada transação:

# In[26]:


temp = df_initial.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Número de produtos'})
nb_products_per_basket[:10].sort_values('CustomerID')


# Podemos observar alguns pontos interessantes:
# 
# - a existência de entradas com o prefixo C para a variável InvoiceNo: indicando transações que foram canceladas
# - usuários que vieram apenas uma vez no site e compraram apenas um produto (ex: nº 12346)
# - a existência de usuários frequentes que compram um grande número de itens em cada pedido

# Pedidos cancelados

# Em primeiro lugar, contamos número de transações correspondentes aos pedidos cancelados:

# In[27]:


nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x:int('C' in x))
display(nb_products_per_basket[:5])
#______________________________________________________________________________________________
n1 = nb_products_per_basket['order_canceled'].sum()
n2 = nb_products_per_basket.shape[0]
print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))


# Notamos que o número de cancelamentos é bastante grande (∼ 17% do número total de transações). 

# In[28]:


display(df_initial.sort_values('CustomerID')[:5])


# Vemos que, quando um pedido é cancelado, temos outras transações no dataframe, em sua maioria idênticas, exceto pelas variáveis Quantity e InvoiceDate. Vamos verificar se isso é verdade para todas as entradas. Para isso, decido localizar as entradas que indicam uma quantidade negativa e verifico se existe sistematicamente um pedido que indique a mesma quantidade (mas positivo), com a mesma descrição (CustomerID, Description e UnitPrice):

# In[29]:


df_check = df_initial[df_initial['Quantity'] < 0][['CustomerID','Quantity',
                                                   'StockCode','Description','UnitPrice']]
for index, col in  df_check.iterrows():
    if df_initial[(df_initial['CustomerID'] == col[0]) & (df_initial['Quantity'] == -col[1]) 
                & (df_initial['Description'] == col[2])].shape[0] == 0: 
        print(df_check.loc[index])
        print(15*'-'+'>'+' HYPOTHESIS NOT FULFILLED')
        break


# Vemos que a hipótese inicial não é cumprida devido à existência de uma entrada de 'Desconto'. Eu verifico novamente a hipótese, mas desta vez descartando as entradas de 'Desconto':

# In[30]:


df_check = df_initial[(df_initial['Quantity'] < 0) & (df_initial['Description'] != 'Discount')][
                                 ['CustomerID','Quantity','StockCode',
                                  'Description','UnitPrice']]

for index, col in  df_check.iterrows():
    if df_initial[(df_initial['CustomerID'] == col[0]) & (df_initial['Quantity'] == -col[1]) 
                & (df_initial['Description'] == col[2])].shape[0] == 0: 
        print(index, df_check.loc[index])
        print(15*'-'+'>'+' HYPOTHESIS NOT FULFILLED')
        break


# Mais uma vez, descobrimos que a hipótese inicial não foi verificada. Portanto, os cancelamentos não correspondem necessariamente a pedidos que teriam sido feitos anteriormente.
# 
# Neste ponto, decido criar uma nova variável no dataframe que indica se parte do comando foi cancelado. Para os cancelamentos sem contrapartidas, alguns deles provavelmente se devem ao fato de as ordens de compra terem sido realizadas antes de dezembro de 2010 (ponto de entrada do banco de dados). Abaixo, faço um censo dos pedidos cancelados e verifico a existência de contrapartes:

# In[31]:


df_cleaned = df_initial.copy(deep = True)
df_cleaned['QuantityCanceled'] = 0

entry_to_remove = [] ; doubtfull_entry = []

for index, col in  df_initial.iterrows():
    if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
    df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                         (df_initial['StockCode']  == col['StockCode']) & 
                         (df_initial['InvoiceDate'] < col['InvoiceDate']) & 
                         (df_initial['Quantity']   > 0)].copy()
    #_________________________________
    # Cancelation WITHOUT counterpart
    if (df_test.shape[0] == 0): 
        doubtfull_entry.append(index)
    #________________________________
    # Cancelation WITH a counterpart
    elif (df_test.shape[0] == 1): 
        index_order = df_test.index[0]
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        entry_to_remove.append(index)        
    #______________________________________________________________
    # Various counterparts exist in orders: we delete the last one
    elif (df_test.shape[0] > 1): 
        df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
        for ind, val in df_test.iterrows():
            if val['Quantity'] < -col['Quantity']: continue
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index) 
            break            


# Na função acima, verifiquei os dois casos:
# 
# 1. um pedido de cancelamento existe sem contrapartida
# 2. há pelo menos uma contraparte com exatamente a mesma quantidade
# 
# O índice do pedido de cancelamento correspondente é mantido respectivamente nas listas doubtfull_entry e entry_to_remove, cujos tamanhos são:

# In[32]:


print("entry_to_remove: {}".format(len(entry_to_remove)))
print("doubtfull_entry: {}".format(len(doubtfull_entry)))


# Entre essas entradas, as linhas listadas na lista * doubtfull_entry * correspondem às entradas que indicam um cancelamento, mas para o qual não há comando prévio. Na prática, decido excluir todas essas entradas, que contam, respectivamente, ∼ 1,4% e 0,2% das entradas do dataframe.
# 
# Agora eu verifico o número de entradas que correspondem aos cancelamentos e que não foram excluídas com o filtro anterior:

# In[33]:


df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)
df_cleaned.drop(doubtfull_entry, axis = 0, inplace = True)
remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
print("nb of entries to delete: {}".format(remaining_entries.shape[0]))
remaining_entries[:5]


# In[34]:


df_cleaned.shape


# In[35]:


df_cleaned[(df_cleaned['CustomerID'] == 12346) & (df_cleaned['StockCode'] == '23166')]


# Se olharmos, por exemplo, para as compras do consumidor de uma das entradas acima e para o mesmo produto que foi cancelado, observamos que, a quantidade cancelada é maior que a soma das compras anteriores.

# In[36]:


df_cleaned[(df_cleaned['CustomerID'] == 14048) & (df_cleaned['StockCode'] == '22464')]


# In[37]:


list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
list_special_codes


# In[38]:


for code in list_special_codes:
    print("{:<15} -> {:<30}".format(code, df_cleaned[df_cleaned['StockCode'] == code]['Description'].unique()[0]))


# #### Basket Price 
# 
# Crio uma nova variável que indica o preço total de cada compra

# In[39]:


df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
df_cleaned.sort_values('CustomerID')[:5]


# Como podemos ver pela tabela abaixo, cada entrada do dataframe indica um valor para um único tipo de produto. Portanto, os pedidos são divididos em várias linhas. Consolido todas as compras feitas durante um único pedido para recuperar o valor total do pedido:

# In[40]:


df_cleaned[(df_cleaned['CustomerID'] == 12347) & (df_cleaned['InvoiceNo'] == '562032')]


# In[41]:


#___________________________________________
# soma de compras / usuário e pedido
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})
#_____________________
# data do pedido
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# seleção de entradas significativas:
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID')[:6]


# In[42]:


basket_price[(basket_price['CustomerID'] == 12347) & (basket_price['InvoiceNo'] == '562032')]


# In[43]:


df_cleaned[(df_cleaned['CustomerID'] == 12347) & (df_cleaned['InvoiceNo'] == '562032')]['TotalPrice'].sum()


# 

# Para ter uma visão global do tipo de pedido realizado neste conjunto de dados, determino como as compras são divididas de acordo com o total de prêmios:

# In[44]:


#____________________
# Décompte des achats
price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
count_price = []
for i, price in enumerate(price_range):
    if i == 0: continue
    val = basket_price[(basket_price['Basket Price'] < price) &
                       (basket_price['Basket Price'] > price_range[i-1])]['Basket Price'].count()
    count_price.append(val)
#____________________________________________
# Représentation du nombre d'achats / montant        
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
colors = ['yellowgreen', 'gold', 'wheat', 'c', 'violet', 'royalblue','firebrick']
labels = [ '{}<.<{}'.format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
sizes  = count_price
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
ax.pie(sizes, explode = explode, labels=labels, colors = colors,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow = False, startangle=0)
ax.axis('equal')
f.text(0.5, 1.01, "Detalhamento dos valores dos pedidos", ha='center', fontsize = 18);


# Pode-se ver que a grande maioria dos pedidos diz respeito a compras relativamente grandes, uma vez que ∼ 65% das compras são acima de £ 200.

# In[ ]:





# In[45]:


from datetime import timedelta
import matplotlib.pyplot as plt
#import squarify


# In[46]:


# Create snapshot date
snapshot_date = basket_price['InvoiceDate'].max() + timedelta(days=1)
print('snapshot_date: ' , snapshot_date)

# Grouping by CustomerID
data_process = basket_price.groupby(['CustomerID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'Basket Price': 'sum'})
# Rename the columns 
data_process.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'Basket Price': 'MonetaryValue'}, inplace=True)


# In[47]:


# última compra registrada no nosso banco de dados
basket_price['InvoiceDate'].max()


# In[48]:


data_process


# #### Unit Test 
# 
# Testando se o cálculo RFM deu certo

# In[49]:


df_CID_18282 = df_cleaned[(df_cleaned['CustomerID'] == 18282)]


# In[50]:


df_CID_18282['InvoiceNo'].nunique() #frequency


# In[51]:


df_CID_18282['InvoiceDate'].max()


# In[52]:


snapshot_date - df_CID_18282['InvoiceDate'].max() #recency


# In[53]:


df_CID_18282['TotalPrice'].sum() #MonetaryValue


# The traditional approach to use RFM model is to sort the customer data and then divide the data into five equal
# segments for each dimension of RFM [22]. The top 20% segment is assigned as a value of 5, the next 20% segment
# is assigned as a value of 4, and so on. Thus, each customer based on RFM model can be represented by one of 125 RFM
# cells, namely, 555, 554, 553, . . . , 111 [22, 25]. Chang et al. [19], on the other hand, use the original data rather than the
# coded number to perform RFM model

# #### Adicionando Lenght

# Reinartz and Kumar [26] addressed that RFM model cannot distinguish which customers have long-term or shortterm relationships with the company. The customer loyalty depending on the relationship between a company and its
# customers is established from a long-term customer relationship management [10]. Therefore, Chang and Tsay [10]
# extended RFM model to LRFM model by taking length (L) into consideration, where L is defined as the number of time
# periods (such as days) from the first purchase to the last purchase in the database.

# Combinações de pedidos do consumidor
# 
# Em uma segunda etapa, agrupo as diferentes entradas que correspondem ao mesmo usuário. Determino assim o número de compras efetuadas pelo usuário, bem como os valores mínimos, máximos, médios e o valor total gasto durante todas as visitas:

# In[54]:


#________________________________________________________________
# nb de visites et stats sur le montant du panier / utilisateurs
transactions_per_user=basket_price.groupby(by=['CustomerID']).agg({
    'Basket Price':'sum',
    'InvoiceDate':lambda x: (x.max() - x.min()).days})

transactions_per_user.rename(columns={'InvoiceDate': 'Lenght'}, inplace=True)
#for i in range(5):
#    col = 'categ_{}'.format(i)
#    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
#                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
#basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]


# In[55]:


df_cleaned[(df_cleaned['CustomerID'] == 12347)]['TotalPrice'].sum()


# In[56]:


df_cleaned[(df_cleaned['CustomerID'] == 12347)]['InvoiceDate'].max() - df_cleaned[(df_cleaned['CustomerID'] == 12347)]['InvoiceDate'].min()


# In[57]:


df_cleaned[(df_cleaned['CustomerID'] == 12348)]['InvoiceDate'].max() - df_cleaned[(df_cleaned['CustomerID'] == 12348)]['InvoiceDate'].min()


# In[58]:


df_cleaned[(df_cleaned['CustomerID'] == 12348)]['InvoiceDate'].max()


# In[59]:


df_cleaned[(df_cleaned['CustomerID'] == 12348)]['InvoiceDate'].min()


# Por fim, defino duas variáveis ​​adicionais que fornecem o número de dias decorridos desde a primeira compra (** FirstPurchase ) e o número de dias desde a última compra ( LastPurchase **):

# In[60]:


last_date = basket_price['InvoiceDate'].max().date()
print('Data de referência - extração: ', last_date)



first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase      = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test  = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)


transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['InvoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['InvoiceDate']


transactions_per_user[:5]


# In[61]:


df_cleaned[(df_cleaned['CustomerID'] == 12347)]['InvoiceDate'].min()


# In[62]:


df_cleaned[(df_cleaned['CustomerID'] == 12347)]['InvoiceDate'].max()


# ### Remoção outliers via IQR

# Escrevendo uma função para remoção de outliers segundo o IQR

# In[63]:


def remove_outliers_IQR(df, attributes, factor=2):
    """Funcao para remover outliers com base no IQR
    Parametros:
        - df : dataframe
        - attributes: atributos a considerar na remoção
        - factor: fator do IQR a considerar
    Retorno:
        dataframe com os outliers removidos
    """
    dfn = df.copy()
    
    for var in attributes:
        # verifica se variável é numerica
        if np.issubdtype(df[var].dtype, np.number):
            Q1 = dfn[var].quantile(0.25)
            Q2 = dfn[var].quantile(0.50)
            Q3 = dfn[var].quantile(0.75)
            IQR = Q3 - Q1
            
            # apenas inliers segundo IQR
            dfn = dfn.loc[(df[var] >= Q1-(IQR*factor)) & (df[var] <= Q3+(IQR*factor)),:]

    return dfn


# In[64]:


atts = 'Lenght'
transactions_per_user.boxplot(atts)
print(transactions_per_user[atts].unique())


# In[65]:


transactions_per_user2 = transactions_per_user


# In[66]:


transactions_per_user = remove_outliers_IQR(transactions_per_user, transactions_per_user.columns)
transactions_per_user.head()


# In[67]:


plt.figure(figsize=(9,4))
plt.subplot(121); transactions_per_user2.boxplot(['Lenght'])
plt.title('Original')

plt.subplot(122); transactions_per_user.boxplot(['Lenght']); 
plt.title('Apos remoção de outliers')


# In[68]:


data_process['MonetaryValue'].describe()


# In[69]:


atts = 'MonetaryValue'
data_process.boxplot(atts)
print(data_process[atts].unique())


# In[70]:


atts = 'Recency'
data_process.boxplot(atts)
print(data_process[atts].unique())


# In[71]:


atts = 'Frequency'
data_process.boxplot(atts)
print(data_process[atts].unique())


# In[ ]:





# In[72]:


data_process_noutli = remove_outliers_IQR(data_process, data_process.columns)
data_process_noutli.head()


# In[73]:


plt.figure(figsize=(9,4))
plt.subplot(121); data_process.boxplot(['MonetaryValue'])
plt.title('Original')

plt.subplot(122); data_process_noutli.boxplot(['MonetaryValue']); 
plt.title('Apos remoção de outliers')


# In[74]:


plt.figure(figsize=(9,4))
plt.subplot(121); data_process.boxplot(['Recency'])
plt.title('Original')

plt.subplot(122); data_process_noutli.boxplot(['Recency']); 
plt.title('Apos remoção de outliers')


# In[75]:


plt.figure(figsize=(9,4))
plt.subplot(121); data_process.boxplot(['Frequency'])
plt.title('Original')

plt.subplot(122); data_process_noutli.boxplot(['Frequency']); 
plt.title('Apos remoção de outliers')


# In[76]:


# Plot RFM distributions
plt.figure(figsize=(14,12))
# Plot distribution of R
plt.subplot(4, 1, 1); sns.distplot(data_process_noutli['Recency'])
# Plot distribution of F
plt.subplot(4, 1, 2); sns.distplot(data_process_noutli['Frequency'])
# Plot distribution of M
plt.subplot(4, 1, 3); sns.distplot(data_process_noutli['MonetaryValue'])
# Plot distribution of L
plt.subplot(4, 1, 4); sns.distplot(transactions_per_user['Lenght'])
# Show the plot
plt.show()


# In[77]:


data_process_noutli


# ### Análise descritiva

# In[ ]:





# In[ ]:





# ### Segmentação 

# The problem is that pandas.qcut chooses the bins/quantiles so that each one has the same number of records, but all records with the same value must stay in the same bin/quantile (this behaviour is in accordance with the statistical definition of quantile).
# 
# Rank your data with DataFrame.rank(method='first'). The ranking assigns a unique value to each element in the dataframe (the rank) while keeping the order of the elements (except for identical values, which will be ranked in order they appear in the array, see method='first')

# In[78]:


# --Calculate R and F groups--
# Create labels for Recency and Frequency
r_labels = range(4, 0, -1)
f_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(data_process_noutli['Recency'], q=4, labels=r_labels)
# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(data_process_noutli['Frequency'].rank(method='first'), q=4, labels=f_labels)
# Create new columns R and F 
data_process_noutli = data_process_noutli.assign(R = r_groups.values, F = f_groups.values)
data_process_noutli.head()


# In[79]:


ax = sns.boxplot(x='R', y='Recency', data=data_process_noutli)


# In[80]:


sns.set_palette('colorblind')
sns.relplot(x='R', y='Recency', data=data_process_noutli)


# In[81]:


data_process_noutli2 = data_process_noutli.query('Frequency < 30')
ax = sns.boxplot(x='F', y='Frequency', data=data_process_noutli2)


# In[82]:


sns.set_palette('colorblind')
sns.relplot(x='F', y='Frequency', data=data_process_noutli2)


# In[83]:


data_process_noutli['F'].value_counts().plot(kind='bar')


# In[84]:


# Create labels for MonetaryValue
m_labels = range(1, 5)
# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(data_process_noutli['MonetaryValue'], q=4, labels=m_labels)
# Create new column M
data_process_noutli = data_process_noutli.assign(M = m_groups.values)


# In[85]:


#data_process_noutli2 = data_process_noutli.query('MonetaryValue < 30')
ax = sns.boxplot(x='M', y='MonetaryValue', data=data_process_noutli)


# In[86]:


sns.set_palette('colorblind')
sns.relplot(x='M', y='MonetaryValue', data=data_process_noutli)


# In[87]:


data_process_noutli['M'].value_counts().plot(kind='bar')


# In[88]:


data_process_noutli['M'].value_counts()


# #### Join

# In[89]:


transactions_per_user.head()


# In[90]:


transactions_per_user['CustomerID'].duplicated().sum()


# In[91]:


data_process_noutli.head()


# In[92]:


data_process_noutli.reset_index(inplace=True)
data_process_noutli.head()


# In[93]:


data_process_noutli['CustomerID'].duplicated().sum()


# In[94]:


df_combined = data_process_noutli.set_index('CustomerID').join(transactions_per_user.set_index('CustomerID'))
df_combined_rfml = df_combined[['Recency','Frequency','MonetaryValue','Lenght','R','F','M']]


# In[95]:


df_combined_rfml


# In[96]:


# Create labels for Lenght
L_labels = range(1, 5)
# Assign these labels to three equal percentile groups 
L_groups = pd.qcut(df_combined_rfml['Lenght'].rank(method='first'), q=4, labels=L_labels)
# Create new column L
df_combined_rfml = df_combined_rfml.assign(L = L_groups.values)
df_combined_rfml.head(20)


# In[97]:


#data_process_noutli2 = data_process_noutli.query('MonetaryValue < 30')
ax = sns.boxplot(x='L', y='Lenght', data=df_combined_rfml)


# In[98]:


sns.set_palette('colorblind')
sns.relplot(x='L', y='Lenght', data=df_combined_rfml)


# In[99]:


df_combined_rfml['L'].value_counts().plot(kind='bar')


# In[100]:


df_combined_rfml


# In[101]:



df_combined_rfml2 = df_combined_rfml.query('Frequency < 30')

sns.set_palette('colorblind')
sns.relplot(x='Recency', y='Frequency', hue='M', col='L',  data=df_combined_rfml2)


# In[102]:


ax = sns.boxplot(x="F", y="MonetaryValue",

                 data=df_combined_rfml, palette="Set3")


# In[103]:


sns.set_palette('colorblind')
sns.relplot(x='Recency', y='Lenght', data=df_combined_rfml)


# In[104]:


#proximidade marca


# In[105]:


df_combined_rfml['Proximidade_Marca_Score'] = df_combined_rfml[['R','F']].sum(axis=1)
print(df_combined_rfml['Proximidade_Marca_Score'].head())


# In[ ]:





# In[106]:


# Concat RFM quartile values to create RFM Segments
def join_rfml(x): return str(x['R']) + str(x['F']) + str(x['M']) + str(x['L'])
df_combined_rfml['RFML_Segment_Concat'] = df_combined_rfml.apply(join_rfml, axis=1)
rfml = df_combined_rfml
rfml.head()


# From the output, you can see that we have our concatenated segments ready to be used for our segmentation, but wait, there is one issue…

# In[107]:


# Count num of unique segments
rfml_count_unique = rfml.groupby('RFML_Segment_Concat')['RFML_Segment_Concat'].nunique()
print(rfml_count_unique.sum())


# Having 166 different segments using the concatenate method quickly becomes unwieldy for any practical use. We will need a more concise way to define our segments.

# In[108]:


# Calculate RFM_Score
rfml['RFML_Score'] = rfml[['R','F','M','L']].sum(axis=1)
print(rfml['RFML_Score'].head())


# In[109]:


rfml


# In[110]:


# Define rfm_level function
def rfml_level(df):
    if df['RFML_Score'] >= 9:
        return 'Can\'t Loose Them'
    elif ((df['RFML_Score'] >= 7) and (df['RFML_Score'] < 9)):
        return 'Champions'    
    elif ((df['RFML_Score'] >= 6) and (df['RFML_Score'] < 7)):
        return 'Potential'
    elif ((df['RFML_Score'] >= 5) and (df['RFML_Score'] < 6)):
        return 'Promising'
    elif ((df['RFML_Score'] >= 4) and (df['RFML_Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'
# Create a new variable RFM_Level
rfml['RFML_Level'] = rfml.apply(rfml_level, axis=1)
# Print the header with top 5 rows to the console
rfml.head()


# In[111]:


# Calculate average values for each RFM_Level, and return a size of each segment 
rfml_level_agg = rfml.groupby('RFML_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Lenght': 'mean',
    'MonetaryValue': ['mean', 'count']
    
}).round(1)
# Print the aggregated dataset
print(rfml_level_agg)


# In[112]:


import squarify

#rfml_level_agg.columns = rfml_level_agg.columns.droplevel()
rfml_level_agg.columns = ['RecencyMean','FrequencyMean','LenghtMean','MonetaryMean','Count']
#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=rfml_level_agg['Count'], 
              label=['Can\'t Loose Them',
                     'Champions',                    
                     'Needs Attention',
                     'Potential', 
                     'Promising', 
                     'Require Activation'], alpha=.6 )
plt.title("RFML Segments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()


# In[ ]:





# In[113]:


rfml.reset_index(inplace=True)
rfml


# In[114]:



def labels(df):
    if df['Proximidade_Marca_Score'] >= 7:
        return 'Leal'   
    elif ((df['Proximidade_Marca_Score'] >= 5) and (df['Proximidade_Marca_Score'] < 7)):
        return 'Em processo de fidelização'    
    elif ((df['Proximidade_Marca_Score'] >= 4) and (df['Proximidade_Marca_Score'] < 5)):
        return 'Pouco relacionamento/ regular'
    else:
        return 'Clientes novos/ potenciais'
# Create a new variable RFM_Level
rfml['Proximidade_Marca'] = rfml.apply(labels, axis=1)
# Print the header with top 5 rows to the console
rfml.head(5)


# In[115]:


# Tabela de dupla entrada

tabela_dupla = pd.crosstab(index=rfml['RFML_Level'], columns=rfml['Proximidade_Marca'])

tabela_dupla


# In[116]:


from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27

tabela_dupla.plot.bar(stacked=True)

plt.legend(title='RFML Level vs Proximidade com a marca')

plt.show()


# In[117]:



# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27
ax = sns.boxplot(x='RFML_Level', y='MonetaryValue',data=rfml)


# In[118]:



# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27
ax = sns.boxplot(x='RFML_Level', y='MonetaryValue', hue='Proximidade_Marca',data=rfml)


# In[119]:


df_combined_rfml_final= rfml[['CustomerID','Recency','Frequency','MonetaryValue','Lenght','Proximidade_Marca_Score','RFML_Score']]


# In[120]:


df_combined_rfml_final


# In[121]:


df_combined_rfml_final.describe()


# we have a couple of issues to resolve first before clustering. K means expects our data to have equal variance, but the mean and std for each metric indicate that this isn’t the case. This is perfectly normal for RFM data, but it shows that we need to preprocess the data first to resolve this.

# In[122]:


fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df_combined_rfml_final.hist(ax = ax, bins=50)


# data are strongly skewed. Again, this is perfectly normal in retail, but it shows us that we need to transform the data to make the distribution a bit more “normal”, in statistical terms.

# ____
# ## Insights sobre categorias de produtos
# 
# No dataframe, os produtos são identificados exclusivamente por meio da variável **StockCode**. Uma breve descrição dos produtos é fornecida na variável **Descrição**. Nesta seção, pretendo utilizar o conteúdo desta última variável para agrupar os produtos em diferentes categorias.
# 
# ___
# ### Descrição dos produtos
# 
# Como primeiro passo, extraio da variável **Descrição** as informações que serão úteis. Para isso, utilizo a seguinte função:

# In[123]:


is_noun = lambda pos: pos[:2] == 'NN'

def keywords_inventory(dataframe, colonne = 'Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        
        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:                
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1                
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1
    
    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:  
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)            
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    print("Nb of keywords in variable '{}': {}".format(colonne,len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords


# Esta função toma como entrada o dataframe e analisa o conteúdo da coluna **Descrição** realizando as seguintes operações:
# 
# - extrair os nomes (próprios, comuns) que aparecem na descrição dos produtos
# - para cada nome, extraio a raiz da palavra e agrego o conjunto de nomes associados a essa raiz específica
# - contar o número de vezes que cada raiz aparece no dataframe
# - quando várias palavras são listadas para a mesma raiz, considero que a palavra-chave associada a essa raiz é o nome mais curto (isso seleciona sistematicamente o singular quando há variantes singular/plural)
# 
# O primeiro passo da análise é recuperar a lista de produtos:

# In[124]:


df_produits = pd.DataFrame(df_initial['Description'].unique()).rename(columns = {0:'Description'})


# Uma vez criada esta lista, utilizo a função que defini anteriormente para analisar a descrição dos vários produtos:

# In[125]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[126]:


keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)


# A execução desta função retorna três variáveis:
# - `keywords`: a lista de palavras-chave extraídas
# - `keywords_roots`: um dicionário onde as chaves são as raízes das palavras-chave e os valores são as listas de palavras associadas a essas raízes
# - `count_keywords`: dicionário listando o número de vezes que cada palavra é usada
# 
# Neste ponto, converto o dicionário `count_keywords` em uma lista, para ordenar as palavras-chave de acordo com suas ocorrências:

# In[127]:


list_products = []
for k,v in count_keywords.items():
    list_products.append([keywords_select[k],v])
list_products.sort(key = lambda x:x[1], reverse = True)


# Usando-o, crio uma representação das palavras-chave mais comuns:

# In[128]:


liste = sorted(list_products, key = lambda x:x[1], reverse = True)
#_______________________________
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 25))
y_axis = [i[1] for i in liste[:125]]
x_axis = [k for k,i in enumerate(liste[:125])]
x_label = [i[0] for i in liste[:125]]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13)
plt.yticks(x_axis, x_label)
plt.xlabel("Nb. of occurences", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align = 'center')
ax = plt.gca()
ax.invert_yaxis()
#_______________________________________________________________________________________
plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 25)
plt.show()


# ___
# ### Definindo as categorias dos produtos

# A lista obtida contém mais de 1400 palavras-chave e as mais frequentes aparecem em mais de 200 produtos. No entanto, ao examinar o conteúdo da lista, noto que alguns nomes são inúteis. Outros não carregam informações, como cores. Portanto, descarto essas palavras da análise que segue e também decido considerar apenas as palavras que aparecem mais de 13 vezes.

# In[129]:


list_products = []
for k,v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word) < 3 or v < 13: continue
    if ('+' in word) or ('/' in word): continue
    list_products.append([word, v])
#______________________________________________________    
list_products.sort(key = lambda x:x[1], reverse = True)
print('palavras preservadas:', len(list_products))


# ____
# ####  Data encoding
# 
# Agora vou usar essas palavras-chave para criar grupos de produtos. Em primeiro lugar, defino a matriz $X$ como:

#    
# |   | palavra 1  |  ...  | palavra j  | ...  | palavra N  |
# |:-:|---|---|---|---|---|
# | produit 1  | $a_{1,1}$  |     |   |   | $a_{1,N}$  |
# | ...        |            |     | ...  |   |   |
# |produit i   |    ...     |     | $a_{i,j}$    |   | ...  |
# |...         |            |     |  ... |   |   |
# | produit M  | $a_{M,1}$  |     |   |   | $a_{M,N}$   |

# onde o coeficiente $a_ {i, j}$ é 1 se a descrição do produto $i$ contiver a palavra $j$, e 0 caso contrário.

# In[130]:


liste_produits = df_cleaned['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))


# A matriz $X$ indica as palavras contidas na descrição dos produtos usando o princípio *one-hot-encoding*. Na prática, descobri que a introdução da faixa de preço resulta em grupos mais equilibrados em termos de números de elementos.
# Assim, adiciono 6 colunas extras a esta matriz, onde indico a faixa de preço dos produtos:

# In[131]:


threshold = [0, 1, 2, 3, 5, 10]
label_col = []
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(liste_produits):
    prix = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
    j = 0
    while prix > threshold[j]:
        j+=1
        if j == len(threshold): break
    X.loc[i, label_col[j-1]] = 1


# e para escolher os ranges adequados, verifico o número de produtos nos diferentes grupos:

# In[132]:


print("{:<8} {:<20} \n".format('faixa', 'número de produtos') + 20*'-')
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])    
    print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))


# In[133]:


X


# In[134]:


seed(1)
#set_seed(2)


# ____
# #### Criando clusters de produtos
# 
# Nesta seção, agruparei os produtos em diferentes classes. No caso de matrizes com codificação binária, a métrica mais adequada para o cálculo de distâncias é a [métrica de Hamming](https://en.wikipedia.org/wiki/Distance_de_Hamming). Observe que o método **kmeans** do sklearn usa uma distância euclidiana que pode ser usada, mas não é a melhor escolha no caso de variáveis ​​categóricas. No entanto, para usar a métrica de Hamming, precisamos usar o pacote [kmodes](https://pypi.python.org/pypi/kmodes/) que não está disponível no plateform atual. Portanto, eu uso o método **kmeans** mesmo que essa não seja a melhor escolha.
# 
# Para definir (aproximadamente) o número de clusters que melhor representam os dados, utilizo a silhouette score:

# In[135]:


matrix = X.values
for n_clusters in range(2,10):
    kmeans = KMeans(n_clusters = n_clusters,n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


# Na prática, as pontuações obtidas acima podem ser consideradas equivalentes, pois, dependendo da execução, serão obtidas pontuações de $ 0,1 \pm 0,05 $ para todos os clusters com `n_clusters` $> $ 3 (obtemos pontuações um pouco mais baixas para o primeiro cluster ). Por outro lado, descobri que além de 5 clusters, alguns clusters continham muito poucos elementos. Portanto, escolho separar o conjunto de dados em 5 clusters. A fim de garantir uma boa classificação a cada execução do notebook, eu itero até obter a melhor pontuação de silhueta possível, que é, no presente caso, em torno de 0,15:

# In[136]:


n_clusters = 5
silhouette_avg = -1
while silhouette_avg < 0.145:
    kmeans = KMeans(n_clusters = n_clusters,n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    
    #km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
    #clusters = km.fit_predict(matrix)
    #silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


# ___
# ####  Caracterizando o conteúdo dos clusters

# Eu verifico o número de elementos em cada classe:

# In[137]:


pd.Series(clusters).value_counts()


# ** a / _Pontuação intra-cluster de silhueta_ **
# 
# Para ter uma visão sobre a qualidade da classificação, podemos representar o silhouette score de cada elemento dos diferentes clusters. Este é o propósito da próxima figura que é retirada da [documentação do sklearn](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html):

# In[138]:


def graph_component_silhouette(n_clusters, lim_x, mat_size, sample_silhouette_values, clusters):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
    #____________________________
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax1.set_xlim([lim_x[0], lim_x[1]])
    ax1.set_ylim([0, mat_size + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        #___________________________________________________________________________________
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)        
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.8)
        #____________________________________________________________________
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.03, y_lower + 0.5 * size_cluster_i, str(i), color = 'red', fontweight = 'bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
        #______________________________________
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  


# In[139]:


#____________________________________
# define individual silouhette scores
sample_silhouette_values = silhouette_samples(matrix, clusters)
#__________________
# and do the graph
graph_component_silhouette(n_clusters, [-0.07, 0.33], len(X), sample_silhouette_values, clusters)


# ** b/ _Word Cloud_**
# 
# Agora podemos dar uma olhada no tipo de objetos que cada cluster representa. Para obter uma visão global de seus conteúdos, determino quais palavras-chave são as mais frequentes em cada um deles

# In[140]:


liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]

for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))


# e eu produzo o resultado como wordclouds:

# In[141]:


#________________________________________________________________________
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
#________________________________________________________________________
def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', 
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster nº{}'.format(increment-1))
#________________________________________________________________________
fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] # define the color of the words
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)            


# A partir desta representação, podemos ver que, por exemplo, um dos clusters contém objetos que podem estar associados a presentes (palavras-chave: Natal, embalagem, cartão, ...). Outro cluster prefere conter itens de luxo e joias (palavras-chave: colar, pulseira, renda, prata, ...). No entanto, também pode-se observar que muitas palavras aparecem em vários clusters e, portanto, é difícil distingui-las claramente.
# 
# ** c/ _Análise de Componentes Principais_ **
# 
# Para garantir que esses clusters sejam realmente distintos, analiso sua composição. Dado o grande número de variáveis ​​da matriz inicial, primeiro executo um PCA:

# In[142]:


pca = PCA()
pca.fit(matrix)
pca_samples = pca.transform(matrix)


# e, em seguida, verifique a quantidade de variação explicada por cada componente:

# In[143]:


fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(np.arange(1,matrix.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
            label='individual explained variance')
plt.xlim(0, 100)

ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('Explained variance', fontsize = 14)
plt.xlabel('Principal components', fontsize = 14)
plt.legend(loc='upper left', fontsize = 13);


# Vemos que o número de componentes necessários para explicar os dados é extremamente importante: precisamos de mais de 100 componentes para explicar 90% da variância dos dados. Na prática, decido manter apenas um número limitado de componentes, pois essa decomposição é realizada apenas para visualizar os dados:

# In[144]:


pca = PCA(n_components=50)
matrix_9D = pca.fit_transform(matrix)
mat = pd.DataFrame(matrix_9D)
mat['cluster'] = pd.Series(clusters)


# In[145]:


import matplotlib.patches as mpatches

sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

LABEL_COLOR_MAP = {0:'r', 1:'gold', 2:'b', 3:'k', 4:'c', 5:'g'}
label_color = [LABEL_COLOR_MAP[l] for l in mat['cluster']]

fig = plt.figure(figsize = (15,8))
increment = 0
for ix in range(4):
    for iy in range(ix+1, 4):    
        increment += 1
        ax = fig.add_subplot(2,3,increment)
        ax.scatter(mat[ix], mat[iy], c= label_color, alpha=0.4) 
        plt.ylabel('PCA {}'.format(iy+1), fontsize = 12)
        plt.xlabel('PCA {}'.format(ix+1), fontsize = 12)
        ax.yaxis.grid(color='lightgray', linestyle=':')
        ax.xaxis.grid(color='lightgray', linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        if increment == 9: break
    if increment == 9: break
        
#_______________________________________________
# I set the legend: abreviation -> airline name
comp_handler = []
for i in range(5):
    comp_handler.append(mpatches.Patch(color = LABEL_COLOR_MAP[i], label = i))

plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.97), 
           title='Cluster', facecolor = 'lightgrey',
           shadow = True, frameon = True, framealpha = 1,
           fontsize = 13, bbox_transform = plt.gcf().transFigure)

plt.show()


# ___
# ## Categorias de clientes
# 
# ### Formatando dados
# 
# Na seção anterior, os diferentes produtos foram agrupados em cinco clusters. Para preparar o restante da análise, um primeiro passo consiste em introduzir essas informações no dataframe. Para isso, crio a variável categórica **categ_product** onde indico o cluster de cada produto:

# In[146]:


corresp = dict()
for key, val in zip (liste_produits, clusters):
    corresp[key] = val 
#__________________________________________________________________________
df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)


# ___
# #### Agrupar produtos
# 
# Em um segundo passo, decido criar as variáveis ​​**categ_N** (com $ N \in [0: 4]$) que contém o valor gasto em cada categoria de produto:

# In[147]:


for i in range(5):
    col = 'categ_{}'.format(i)        
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace = True)
#__________________________________________________________________________________________________
df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4']][:5]


# Até agora, as informações relacionadas a um único pedido eram divididas em várias linhas do dataframe (uma linha por produto). Decido recolher a informação relativa a um determinado pedido e introduzi-la num único registo. Crio assim um novo dataframe que contém, para cada pedido, o valor da cesta, bem como a forma como está distribuído pelas 5 categorias de produtos:

# In[148]:


#___________________________________________
# somme des achats / utilisateur & commande
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})
#____________________________________________________________
# pourcentage du prix de la commande / categorie de produit
for i in range(5):
    col = 'categ_{}'.format(i) 
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price.loc[:, col] = temp['categ_{}'.format(i)] 
#_____________________
# date de la commande
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# selection des entrées significatives:
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending = True)[:5]


# ____
# #### Combinações de pedidos do consumidor
# 
# Em uma segunda etapa, agrupo as diferentes entradas que correspondem ao mesmo usuário. Determino assim o número de compras efetuadas pelo usuário, bem como os valores mínimos, máximos, médios e o valor total gasto durante todas as visitas:

# In[149]:


#________________________________________________________________
# nb de visites et stats sur le montant du panier / utilisateurs
transactions_per_user=basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['min','max','mean','sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]


# In[150]:


basket_price[(basket_price['CustomerID'] == 12347)]['Basket Price'].sum()


# In[151]:


basket_price[(basket_price['CustomerID'] == 12347)]['categ_0'].sum()


# In[152]:


basket_price[(basket_price['CustomerID'] == 12347)]['categ_0'].sum()/basket_price[(basket_price['CustomerID'] == 12347)]['Basket Price'].sum()*100


# Adicionando as infos acima no DF segmentado

# In[153]:


df_combined_rfml_final.head()


# In[154]:


transactions_per_user.head()


# In[155]:


df_combined = df_combined_rfml_final.set_index('CustomerID').join(transactions_per_user.set_index('CustomerID'))


# In[156]:


df_combined.reset_index(inplace=True)
df_combined.head()


# In[157]:


list_cols = ['Recency','Frequency','MonetaryValue','Lenght','Proximidade_Marca_Score','RFML_Score','min','max','mean','categ_0','categ_1','categ_2','categ_3','categ_4']
#_____________________________________________________________
selected_customers = df_combined.copy(deep = True)
#matrix = selected_customers[list_cols].values
matrix = np.array(selected_customers[list_cols])


# In[158]:


matrix


# Na prática, as diferentes variáveis ​​que selecionei têm faixas de variação bastante diferentes e antes de continuar a análise, crio uma matriz onde esses dados são padronizados:

# In[159]:


matrix.shape


# In[160]:


scaler = StandardScaler()
scaler.fit(matrix)
print('variables mean values: \n' + 90*'-' + '\n' , scaler.mean_)
scaled_matrix = scaler.transform(matrix)


# In[161]:


pca = PCA()
pca.fit(scaled_matrix)
pca_samples = pca.transform(scaled_matrix)


# e eu represento a quantidade de variação explicada por cada um dos componentes:

# In[162]:


fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(np.arange(1,matrix.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
            label='individual explained variance')
plt.xlim(0, 10)

ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('Explained variance', fontsize = 14)
plt.xlabel('Principal components', fontsize = 14)
plt.legend(loc='best', fontsize = 13);


#  ### K-means

# In[163]:


WCSS= []
K = range(1,10)
# considerando diversos valores de k
for k in K: 
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_matrix)
    # calcula a medida e armazena em uma lista 
    WCSS.append(kmeans.inertia_)

# mostra os resultados
plt.figure(figsize=(8,6))
plt.plot(K, WCSS, '-bo')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In such an ambiguous case, we may use the Silhouette Method. The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).

# In[164]:


for n_clusters_x in range(2,10):
    kmeans = KMeans(n_clusters=n_clusters_x)
    kmeans.fit(scaled_matrix)
    clusters = kmeans.predict(scaled_matrix)
    silhouette_avg = silhouette_score(scaled_matrix, clusters)
    print("For n_clusters =", n_clusters_x, "The average silhouette_score is :", silhouette_avg)


# #### Primeira rodada k = 3

# In[165]:


n_clusters = 3


# In[166]:


kmeans_f = KMeans(n_clusters=n_clusters)
kmeans_f.fit(scaled_matrix)
clusters_clients = kmeans_f.predict(scaled_matrix)
silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)
print('score de silhouette: {:<.3f}'.format(silhouette_avg))


# At first, I look at the number of customers in each cluster:

# In[167]:


pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns = ['nb. de clients']).T


# ** b/ _Pontuação de silhueta intra-cluster_ **
# 
# Assim como nas categorias de produtos, outra maneira de observar a qualidade da separação é observar as pontuações de silouhette em diferentes clusters:

# In[168]:


sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
#____________________________________
# define individual silouhette scores
sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
#__________________
# and do the graph
graph_component_silhouette(n_clusters, [-0.15, 0.55], len(scaled_matrix), sample_silhouette_values, clusters_clients)


# ** c/ _morfotipo de clientes_**
# 
# Nesta fase, verifiquei que os diferentes clusters são de facto disjuntos (pelo menos, de forma global). Resta entender os hábitos dos clientes em cada cluster. Para isso, começo adicionando ao dataframe `selected_customers` uma variável que define o cluster ao qual cada cliente pertence:

#  ### Agglomerative hierarchical method

# In[169]:


# Dendogram for Heirarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(figsize=(18,9))


Z = linkage(scaled_matrix, 'ward')
dendrogram(Z)  
plt.show(True)


# In[170]:


from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
    AgglomerativeC = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(scaled_matrix)
    labels = AgglomerativeC.labels_
    sil.append(silhouette_score(scaled_matrix, labels, metric = 'euclidean'))    


# In[171]:


k_x= np.arange(2,11)
k_x


# In[172]:


plt.plot(k_x,sil)


# In[173]:


k=3


# In[174]:


clustering = AgglomerativeClustering(n_clusters=k, linkage='ward').fit(scaled_matrix)


# In[175]:


silhouette_score(scaled_matrix, clustering.labels_, metric = 'euclidean')


# #### Escolha K-means

# In[176]:


selected_customers.loc[:, 'cluster'] = clusters_clients


# Em seguida, calculo a média do conteúdo desse dataframe selecionando primeiro os diferentes grupos de clientes. Dá acesso, por exemplo, ao preço médio das cestas, ao número de visitas ou aos montantes totais gastos pelos clientes dos diferentes clusters. Também determino o número de clientes em cada grupo (variável ** tamanho **):

# In[177]:


merged_df = pd.DataFrame()
for i in range(n_clusters):
    test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean())
    test = test.T.set_index('cluster', drop = True)
    test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
    merged_df = pd.concat([merged_df, test])
#_____________________________________________________
merged_df.drop('CustomerID', axis = 1, inplace = True)
print('number of customers:', merged_df['size'].sum())

merged_df = merged_df.sort_values('sum')


# In[178]:


merged_df


# #### Teste

# In[179]:


selected_customers.head(3)


# In[180]:


test = selected_customers.query('cluster == 2')
print(test['RFML_Score'].mean())
print(test['categ_1'].mean())


# #### Labeling e análise

# In[181]:


merged_df.round(1).sort_values(by=['RFML_Score', 'Proximidade_Marca_Score'],ascending=False)


# In[182]:


def _scale_data(data, ranges):
    (x1, x2) = ranges[0]
    d = data[0]
    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]

class RadarChart():
    def __init__(self, fig, location, sizes, variables, ranges, n_ordinate_levels = 6):

        angles = np.arange(0, 360, 360./len(variables))

        ix, iy = location[:] ; size_x, size_y = sizes[:]
        
        axes = [fig.add_axes([ix, iy, size_x, size_y], polar = True, 
        label = "axes{}".format(i)) for i in range(len(variables))]

        _, text = axes[0].set_thetagrids(angles, labels = variables)
        
        for txt, angle in zip(text, angles):
            if angle > -1 and angle < 181:
                txt.set_rotation(angle - 90)
            else:
                txt.set_rotation(angle - 270)
        
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid("off")
        
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],num = n_ordinate_levels)
            grid_label = [""]+["{:.0f}".format(x) for x in grid[1:-1]]
            ax.set_rgrids(grid, labels = grid_label, angle = angles[i])
            ax.set_ylim(*ranges[i])
        
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
                
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        self.ax.legend(*args, **kw)
        
    def title(self, title, *args, **kw):
        self.ax.text(0.9, 1, title, transform = self.ax.transAxes, *args, **kw)


# In[183]:


fig = plt.figure(figsize=(14,12))

attributes = ['Frequency', 'mean', 'sum', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
ranges = [[0.01, 7], [0.01, 1000], [0.01, 1900], [0.01, 30], [0.01, 30], [0.01, 30], [0.01, 30], [0.01, 30]]
index  = [0, 1, 2, 3, 4]

n_groups = n_clusters ; i_cols = 3
i_rows = n_groups//i_cols
size_x, size_y = (1/i_cols), (1/i_rows)

for ind in range(n_clusters):
   
    ix = ind%3 ; iy = i_rows - ind//3
    pos_x = ix*(size_x + 0.05) ; pos_y = iy*(size_y + 0.05)            
    location = [pos_x, pos_y]  ; sizes = [size_x, size_y] 
   
    #______________________________________________________
    data = np.array(merged_df.loc[index[ind], attributes])    
    radar = RadarChart(fig, location, sizes, attributes, ranges)
    radar.plot(data, color = 'b', linewidth=2.0)
    radar.fill(data, alpha = 0.2, color = 'b')
    radar.title(title = 'cluster nº{}'.format(index[ind]), color = 'r')
    ind += 1 


# Verifica-se, por exemplo, que alguns clusters correspondem a uma forte preponderância de compras numa determinada categoria de produtos. Outros clusters serão diferentes das médias da cesta (** média ), do valor total gasto pelos clientes ( soma ) ou do número total de visitas realizadas ( Frequência **).

# In[184]:


selected_customers


# In[185]:


# Define rfm_level function
def rfml_level(df):
    if df['RFML_Score'] >= 9:
        return 'Can\'t Loose Them'
    elif ((df['RFML_Score'] >= 7) and (df['RFML_Score'] < 9)):
        return 'Champions'    
    elif ((df['RFML_Score'] >= 6) and (df['RFML_Score'] < 7)):
        return 'Potential'
    elif ((df['RFML_Score'] >= 5) and (df['RFML_Score'] < 6)):
        return 'Promising'
    elif ((df['RFML_Score'] >= 4) and (df['RFML_Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Require Activation'
# Create a new variable RFM_Level
selected_customers['RFML_Level'] = selected_customers.apply(rfml_level, axis=1)
# Print the header with top 5 rows to the console
selected_customers.head()


# In[186]:


segments = {1:'bronze', 2:'silver',0:'gold'}
selected_customers['segment'] = selected_customers['cluster'].map(segments)
selected_customers.head()


# In[187]:


# Tabela de frequências absolutas

tab = pd.crosstab(index=selected_customers['segment'], columns='count')

tab


# In[188]:


tab = pd.crosstab(index=selected_customers['segment'], columns='count')

# Tabela de frequências relativas
tab/tab.sum()


# In[189]:


plot = tab.plot.pie(y='count')


# In[190]:



selected_customer2 = selected_customers.query('Frequency < 30')

sns.set_palette('colorblind')
sns.relplot(x='Recency', y='Frequency', hue='MonetaryValue', col='segment',  data=selected_customer2)


# In[191]:



# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27
ax = sns.boxplot(x='segment', y='MonetaryValue',data=selected_customers)


# ____
# ## Classificação dos clientes
# 
# Nesta parte, o objetivo será ajustar um classificador que classificará os consumidores nas diferentes categorias de clientes que foram estabelecidas na seção anterior. O objetivo é possibilitar essa classificação na primeira visita. Para cumprir este objetivo, testarei vários classificadores implementados no `scikit-learn`. 

# In[192]:


X = scaled_matrix
Y = selected_customers['cluster']


# #### Verificando desbalanceamento

# In[193]:


selected_customers['cluster'].value_counts().plot(kind = 'bar')


# In[194]:


from imblearn import over_sampling


# In[195]:


oversamp = over_sampling.SMOTE() # sampling_strategy pode ser usado para casos binários
Xo, Yo = oversamp.fit_resample(X, Y)
Yo.value_counts().plot(kind = 'bar')


# In[196]:


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Xo, Yo, train_size = 0.8)


# #### SVM

# In[197]:


from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

clf = svm.SVC(C=3, kernel='linear') #criação do modelo SVM linear referente a 3 classes
clf.fit(X_train, Y_train)


# In[198]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[199]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, y_pred, average = 'weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, y_pred, average = 'weighted'))


# #### AdaBoost

# In[200]:


clf = AdaBoostClassifier()
clf.fit(X_train, Y_train)


# In[201]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[202]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, y_pred, average = 'weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, y_pred, average = 'weighted'))


# #### Árvore de decisão

# In[203]:


from sklearn import tree
# Cria o modelo usando o criterio Gini
model = tree.DecisionTreeClassifier(criterion = 'gini', random_state = 101)
# Ajusta o modelo usando os dados de treinamento
model.fit(X_train,Y_train)
# realizar a predição
y_pred = model.predict(X_test) 


# In[204]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, Y_test)
print('Accuracy:', score)

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, y_pred, average = 'weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, y_pred, average = 'weighted'))


# In[205]:


plt.figure(figsize=(15,10))
tree.plot_tree(model.fit(X_train,Y_train),filled=True)
plt.show(True)


# ##### Se usarmos a medida de entropia.

# In[206]:


from sklearn import tree
# Cria o modelo usando o criterio Gini
model = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 101)
# Ajusta o modelo usando os dados de treinamento
model.fit(X_train,Y_train)
# realizar a predição
y_pred = model.predict(X_test) 


# In[207]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, Y_test)
print('Accuracy:', score)

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Y_test, y_pred, average = 'weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Y_test, y_pred, average = 'weighted'))


# In[208]:


plt.figure(figsize=(15,10))
tree.plot_tree(model.fit(X_train,Y_train),filled=True)
plt.show(True)


# ## Florestas aleatórias

# In[209]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(n_estimators = 50)
clf.fit(X_train, Y_train)


# In[210]:


pred_y = clf.predict(X_test)


# In[211]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred_y, Y_test)
print('Accuracy:', accuracy)


# In[212]:


from sklearn.metrics import confusion_matrix

confusion_matrix(pred_y, Y_test)
pd.crosstab(pred_y, Y_test, rownames=['True'], colnames=['Predicted'], margins=True)


# In[213]:


from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score

print('Accuracy:', accuracy_score(pred_y, Y_test))
print('F1 score:', f1_score(Y_test, pred_y, average="macro"))
print('Precision:', precision_score(Y_test, pred_y, average="macro"))
print('Recall:', recall_score(Y_test, pred_y, average="macro"))
print('\n clasification report:\n', classification_report(Y_test, pred_y))


# #### Comparação modelos

# In[214]:


# Dicionário com os nossos modelos
models = {
    'decision_tree': tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 101),
    'random_forest': RandomForestClassifier(n_estimators = 50),
    'adaboost': AdaBoostClassifier(),
    'svm': svm.SVC(C=3,kernel='linear')
}

# Onde vamos salvar as acurácias
scores = {}

# Treinando e testando os modelos
for clf_name, clf in models.items():
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(Y_test, y_pred)
    
    scores[clf_name] = acc


# In[215]:


# Vamos usar um plot para ver o resultado

plt.figure(figsize=(10, 5))

sns.barplot(list(scores.values()), list(scores.keys()))
plt.xticks(np.arange(0, 0.85, 0.05));
plt.xlabel('Acurácia')
plt.ylabel('Modelo')
plt.title('Comparação modelos')


# #### PCA no treinamento

# In[216]:


pca = PCA(n_components=2)
pca.fit(X_train)
pca_train_origin = pca.transform(X_train)


# código exemplo para scatterplot (sendo pca_train o array com dados projetados, e rot_train rotulos discretos)
scatter = plt.scatter(pca_train_origin[:,0], pca_train_origin[:,1], c=Y_train, cmap="jet")
legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#ax.add_artist(legend1)
plt.title('Scatterplot com projeção PCA do conjunto de treinamento original (14 dimensões)')


# In[217]:


#### PCA no teste


# In[218]:


pca = PCA(n_components=2)
pca.fit(X_test)
pca_test_origin = pca.transform(X_test)


# código exemplo para scatterplot (sendo pca_train o array com dados projetados, e rot_train rotulos discretos)
scatter = plt.scatter(pca_test_origin[:,0], pca_test_origin[:,1], c=Y_test, cmap="jet")
legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#ax.add_artist(legend1)
plt.title('Scatterplot com projeção PCA do conjunto de teste original (14 dimensões)')


# ## Testando previsões
# 
# Na seção anterior, alguns classificadores foram treinados para categorizar os clientes. Até então, toda a análise se baseava nos dados dos primeiros 10 meses. Nesta seção, testo o modelo dos últimos dois meses do dataset, que foi armazenado no dataframe `set_test`:

# ### Tratativa dados - novos clientes Ecommerce

# In[219]:


nb_products_per_basket.drop(nb_products_per_basket.index, inplace=True)
df_cleaned.drop(df_cleaned.index, inplace=True)
basket_price.drop(basket_price.index, inplace=True)
transactions_per_user.drop(transactions_per_user.index, inplace=True)
df_combined.drop(df_combined.index, inplace=True)
df_combined_rfml.drop(df_combined_rfml.index, inplace=True)
rfml.drop(rfml.index, inplace=True)
#X.drop(X.index, inplace=True)


# In[220]:


df_ecommerce_testing_new_cli = set_teste.copy(deep = True)


# In[221]:


df_ecommerce_testing_new_cli.shape


# In[222]:


print(df_ecommerce_testing_new_cli['InvoiceDate'].min(), '->',  df_ecommerce_testing_new_cli['InvoiceDate'].max())


# Número de produtos adquiridos em cada transação:

# In[223]:


temp = df_ecommerce_testing_new_cli.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Número de produtos'})


# Pedidos cancelados
# 
# Em primeiro lugar, contamos número de transações correspondentes aos pedidos cancelados:

# In[224]:


nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x:int('C' in x))

#______________________________________________________________________________________________
n1 = nb_products_per_basket['order_canceled'].sum()
n2 = nb_products_per_basket.shape[0]
print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))


# Tratativas cancelamento

# In[225]:


df_cleaned = df_ecommerce_testing_new_cli.copy(deep = True)
df_cleaned['QuantityCanceled'] = 0

entry_to_remove = [] ; doubtfull_entry = []

for index, col in  df_ecommerce_testing_new_cli.iterrows():
    if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
    df_test = df_ecommerce_testing_new_cli[(df_ecommerce_testing_new_cli['CustomerID'] == col['CustomerID']) &
                         (df_ecommerce_testing_new_cli['StockCode']  == col['StockCode']) & 
                         (df_ecommerce_testing_new_cli['InvoiceDate'] < col['InvoiceDate']) & 
                         (df_ecommerce_testing_new_cli['Quantity']   > 0)].copy()
    #_________________________________
    # Cancelation WITHOUT counterpart
    if (df_test.shape[0] == 0): 
        doubtfull_entry.append(index)
    #________________________________
    # Cancelation WITH a counterpart
    elif (df_test.shape[0] == 1): 
        index_order = df_test.index[0]
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        entry_to_remove.append(index)        
    #______________________________________________________________
    # Various counterparts exist in orders: we delete the last one
    elif (df_test.shape[0] > 1): 
        df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
        for ind, val in df_test.iterrows():
            if val['Quantity'] < -col['Quantity']: continue
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index) 
            break            


# In[226]:


df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)
df_cleaned.drop(doubtfull_entry, axis = 0, inplace = True)
remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
print("nb of entries to delete: {}".format(remaining_entries.shape[0]))
remaining_entries[:5]


# In[227]:


df_cleaned.shape


# #### Basket Price 
# 
# Crio uma nova variável que indica o preço total de cada compra

# In[228]:


df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
df_cleaned.sort_values('CustomerID')[:5]


# Consolido todas as compras feitas durante um único pedido para recuperar o valor total do pedido:

# In[229]:


#___________________________________________
# soma de compras / usuário e pedido
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})
#_____________________
# data do pedido
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# seleção de entradas significativas:
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID')[:6]


# In[230]:


# Create snapshot date
snapshot_date = basket_price['InvoiceDate'].max() + timedelta(days=1)
print('snapshot_date: ' , snapshot_date)

# Grouping by CustomerID
data_process = basket_price.groupby(['CustomerID']).agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'Basket Price': 'sum'})
# Rename the columns 
data_process.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'Basket Price': 'MonetaryValue'}, inplace=True)


# In[231]:


data_process


# Adicionando Lenght

# In[232]:


#________________________________________________________________
# nb de visites et stats sur le montant du panier / utilisateurs
transactions_per_user=basket_price.groupby(by=['CustomerID']).agg({
    'Basket Price':'sum',
    'InvoiceDate':lambda x: (x.max() - x.min()).days})

transactions_per_user.rename(columns={'InvoiceDate': 'Lenght'}, inplace=True)
#for i in range(5):
#    col = 'categ_{}'.format(i)
#    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /\
#                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
#basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]


# In[233]:


last_date = basket_price['InvoiceDate'].max().date()
print('Data de referência - extração: ', last_date)



first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase      = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test  = first_registration.applymap(lambda x:(last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x:(last_date - x.date()).days)


transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop = False)['InvoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop = False)['InvoiceDate']


transactions_per_user[:5]


# In[234]:


transactions_per_user = remove_outliers_IQR(transactions_per_user, transactions_per_user.columns)


# In[235]:


data_process_noutli = remove_outliers_IQR(data_process, data_process.columns)


# Segmentação

# In[236]:


# --Calculate R and F groups--
# Create labels for Recency and Frequency
r_labels = range(4, 0, -1)
f_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(data_process_noutli['Recency'], q=4, labels=r_labels)
# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(data_process_noutli['Frequency'].rank(method='first'), q=4, labels=f_labels)
# Create new columns R and F 
data_process_noutli = data_process_noutli.assign(R = r_groups.values, F = f_groups.values)
data_process_noutli.head()


# In[237]:


# Create labels for MonetaryValue
m_labels = range(1, 5)
# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(data_process_noutli['MonetaryValue'], q=4, labels=m_labels)
# Create new column M
data_process_noutli = data_process_noutli.assign(M = m_groups.values)


# Join

# In[238]:


transactions_per_user.head()


# In[239]:


data_process_noutli.head()


# In[240]:


data_process_noutli.reset_index(inplace=True)
data_process_noutli.head()


# In[241]:


df_combined = data_process_noutli.set_index('CustomerID').join(transactions_per_user.set_index('CustomerID'))
df_combined_rfml = df_combined[['Recency','Frequency','MonetaryValue','Lenght','R','F','M']]


# In[242]:


df_combined_rfml


# In[243]:


# Create labels for Lenght
L_labels = range(1, 5)
# Assign these labels to three equal percentile groups 
L_groups = pd.qcut(df_combined_rfml['Lenght'].rank(method='first'), q=4, labels=L_labels)
# Create new column L
df_combined_rfml = df_combined_rfml.assign(L = L_groups.values)
df_combined_rfml.head(20)


# In[244]:


df_combined_rfml


# proximidade marca

# In[245]:


df_combined_rfml['Proximidade_Marca_Score'] = df_combined_rfml[['R','F']].sum(axis=1)
print(df_combined_rfml['Proximidade_Marca_Score'].head())


# In[246]:


# Concat RFM quartile values to create RFM Segments
def join_rfml(x): return str(x['R']) + str(x['F']) + str(x['M']) + str(x['L'])
df_combined_rfml['RFML_Segment_Concat'] = df_combined_rfml.apply(join_rfml, axis=1)
rfml = df_combined_rfml
rfml.head()


# In[247]:


# Calculate RFM_Score
rfml['RFML_Score'] = rfml[['R','F','M','L']].sum(axis=1)
print(rfml['RFML_Score'].head())


# In[248]:


rfml


# In[249]:


# Create a new variable RFM_Level
rfml['RFML_Level'] = rfml.apply(rfml_level, axis=1)
# Print the header with top 5 rows to the console
rfml.head()


# In[250]:


# Calculate average values for each RFM_Level, and return a size of each segment 
rfml_level_agg = rfml.groupby('RFML_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Lenght': 'mean',
    'MonetaryValue': ['mean', 'count']
    
}).round(1)
# Print the aggregated dataset
print(rfml_level_agg)


# In[251]:


rfml.reset_index(inplace=True)
rfml


# In[252]:



def labels(df):
    if df['Proximidade_Marca_Score'] >= 7:
        return 'Leal'   
    elif ((df['Proximidade_Marca_Score'] >= 5) and (df['Proximidade_Marca_Score'] < 7)):
        return 'Em processo de fidelização'    
    elif ((df['Proximidade_Marca_Score'] >= 4) and (df['Proximidade_Marca_Score'] < 5)):
        return 'Pouco relacionamento/ regular'
    else:
        return 'Clientes novos/ potenciais'
# Create a new variable RFM_Level
rfml['Proximidade_Marca'] = rfml.apply(labels, axis=1)
# Print the header with top 5 rows to the console
rfml.head(5)


# In[253]:


df_combined_rfml_final= rfml[['CustomerID','Recency','Frequency','MonetaryValue','Lenght','Proximidade_Marca_Score','RFML_Score']]


# In[254]:


df_combined_rfml_final


# #### Insight on product categories

# In[255]:


df_produits = pd.DataFrame(df_ecommerce_testing_new_cli['Description'].unique()).rename(columns = {0:'Description'})


# In[256]:


keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)


# In[257]:


list_products = []
for k,v in count_keywords.items():
    list_products.append([keywords_select[k],v])
list_products.sort(key = lambda x:x[1], reverse = True)


# In[258]:


liste = sorted(list_products, key = lambda x:x[1], reverse = True)
#_______________________________
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(7, 25))
y_axis = [i[1] for i in liste[:125]]
x_axis = [k for k,i in enumerate(liste[:125])]
x_label = [i[0] for i in liste[:125]]
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 13)
plt.yticks(x_axis, x_label)
plt.xlabel("Nb. of occurences", fontsize = 18, labelpad = 10)
ax.barh(x_axis, y_axis, align = 'center')
ax = plt.gca()
ax.invert_yaxis()
#_______________________________________________________________________________________
plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 25)
plt.show()


# In[259]:


list_products = []
for k,v in count_keywords.items():
    word = keywords_select[k]
    if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
    if len(word) < 3 or v < 13: continue
    if ('+' in word) or ('/' in word): continue
    list_products.append([word, v])
#______________________________________________________    
list_products.sort(key = lambda x:x[1], reverse = True)
print('palavras preservadas:', len(list_products))


# In[260]:


liste_produits = df_cleaned['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))


# In[261]:


threshold = [0, 1, 2, 3, 5, 10]
label_col = []
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(liste_produits):
    prix = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
    j = 0
    while prix > threshold[j]:
        j+=1
        if j == len(threshold): break
    X.loc[i, label_col[j-1]] = 1


# In[262]:


print("{:<8} {:<20} \n".format('faixa', 'número de produtos') + 20*'-')
for i in range(len(threshold)):
    if i == len(threshold)-1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i],threshold[i+1])    
    print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))


# In[263]:


X


# In[264]:


matrix = X.values
for n_clusters in range(2,10):
    kmeans = KMeans(n_clusters = n_clusters,n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


# In[265]:


n_clusters = 5  # MUDAR?
silhouette_avg = -1
while silhouette_avg < 0.145:
    kmeans = KMeans(n_clusters = n_clusters,n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    
    #km = kmodes.KModes(n_clusters = n_clusters, init='Huang', n_init=2, verbose=0)
    #clusters = km.fit_predict(matrix)
    #silhouette_avg = silhouette_score(matrix, clusters)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)


# In[266]:


#____________________________________
# define individual silouhette scores
sample_silhouette_values = silhouette_samples(matrix, clusters)
#__________________
# and do the graph
graph_component_silhouette(n_clusters, [-0.07, 0.33], len(X), sample_silhouette_values, clusters)


# In[267]:


liste = pd.DataFrame(liste_produits)
liste_words = [word for (word, occurence) in list_products]

occurence = [dict() for _ in range(n_clusters)]
    
for i in range(n_clusters):
    liste_cluster = liste.loc[clusters == i]
    for word in liste_words:
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']: continue
        occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))


# In[268]:


#________________________________________________________________________
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)
#________________________________________________________________________
def make_wordcloud(liste, increment):
    ax1 = fig.add_subplot(4,2,increment)
    words = dict()
    trunc_occurences = liste[0:150]
    for s in trunc_occurences:
        words[s[0]] = s[1]
    #________________________________________________________
    wordcloud = WordCloud(width=1000,height=400, background_color='lightgrey', 
                          max_words=1628,relative_scaling=1,
                          color_func = random_color_func,
                          normalize_plurals=False)
    wordcloud.generate_from_frequencies(words)
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    plt.title('cluster nº{}'.format(increment-1))
#________________________________________________________________________
fig = plt.figure(1, figsize=(14,14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]
for i in range(n_clusters):
    list_cluster_occurences = occurence[i]

    tone = color[i] # define the color of the words
    liste = []
    for key, value in list_cluster_occurences.items():
        liste.append([key, value])
    liste.sort(key = lambda x:x[1], reverse = True)
    make_wordcloud(liste, i+1)            


# Customer categories

# In[269]:


corresp = dict()
for key, val in zip (liste_produits, clusters):
    corresp[key] = val 
#__________________________________________________________________________
df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)


# In[270]:


for i in range(5):
    col = 'categ_{}'.format(i)        
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x:x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace = True)
#__________________________________________________________________________________________________
df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3','categ_4']][:5]


# In[271]:


#___________________________________________
# somme des achats / utilisateur & commande
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})
#____________________________________________________________
# pourcentage du prix de la commande / categorie de produit
for i in range(5):
    col = 'categ_{}'.format(i) 
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price.loc[:, col] = temp['categ_{}'.format(i)] 
#_____________________
# date de la commande
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
#______________________________________
# selection des entrées significatives:
basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending = True)[:5]


# In[272]:


#________________________________________________________________
# nb de visites et stats sur le montant du panier / utilisateurs
transactions_per_user=basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['min','max','mean','sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:,col] = basket_price.groupby(by=['CustomerID'])[col].sum() /                                            transactions_per_user['sum']*100

transactions_per_user.reset_index(drop = False, inplace = True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]


# In[273]:


df_combined = df_combined_rfml_final.set_index('CustomerID').join(transactions_per_user.set_index('CustomerID'))


# In[274]:


df_combined.reset_index(inplace=True)
df_combined.head()


# Fim do tratamento de dados, inicio da predição cluster e classificação dos novos clientes nos respectivos clusters.

# In[275]:


list_cols = ['Recency','Frequency','MonetaryValue','Lenght','Proximidade_Marca_Score','RFML_Score','min','max','mean','categ_0','categ_1','categ_2','categ_3','categ_4']

#_____________________________________________________________
selected_customers2 = df_combined.copy(deep = True)
matrix_test = np.array(selected_customers2[list_cols])
matrix_test.shape


# In[276]:


scaled_test_matrix = scaler.transform(matrix_test) 


# In[277]:


Y = kmeans_f.predict(scaled_test_matrix)


# In[278]:


# armazena os nomes das classes
cl = np.unique(Y)
# armazena o número de elementos em cada classe
ncl = np.zeros(len(cl))
for i in np.arange(0, len(cl)):
    a = Y == cl[i]
    ncl[i] = len(Y[a])
print(ncl)


# In[279]:


# número de classes
numbers = np.arange(0, len(cl))
plt.bar(numbers, ncl,  alpha=.75)
# mostra o nome das classes ao invés dos números
plt.xticks(numbers, cl)
plt.title('Número de elementos em cada classe')
plt.show(True)


# In[280]:


X = scaled_test_matrix


# In[281]:


oversamp = over_sampling.SMOTE() # sampling_strategy pode ser usado para casos binários
Xo, Yo = oversamp.fit_resample(X, Y)


# In[282]:


# armazena os nomes das classes
cl = np.unique(Yo)
# armazena o número de elementos em cada classe
ncl = np.zeros(len(cl))
for i in np.arange(0, len(cl)):
    a = Yo == cl[i]
    ncl[i] = len(Yo[a])
print(ncl)


# In[283]:


# número de classes
numbers = np.arange(0, len(cl))
plt.bar(numbers, ncl,  alpha=.75)
# mostra o nome das classes ao invés dos números
plt.xticks(numbers, cl)
plt.title('Número de elementos em cada classe')
plt.show(True)


# In[284]:


#Predict the response for test dataset
y_pred = clf.predict(Xo)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Yo, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(Yo, y_pred, average = 'weighted'))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(Yo, y_pred, average = 'weighted'))


# #### PCA

# In[285]:


pca = PCA(n_components=2)
pca.fit(Xo)
pca_train = pca.transform(Xo)


# In[286]:


# código exemplo para scatterplot (sendo pca_train o array com dados projetados, e rot_train rotulos discretos)
scatter = plt.scatter(pca_train[:,0], pca_train[:,1], c=y_pred, cmap="jet")
legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
#ax.add_artist(legend1)
plt.title('Scatterplot com projeção PCA do conjunto de treinamento original (14 dimensões)')


# FIM
