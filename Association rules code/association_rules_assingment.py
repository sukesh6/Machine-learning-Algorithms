import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
from sqlalchemy import create_engine
import pickle

df=pd.read_csv(r"C:/Users/sukes/Downloads/Data Set (2)/book.csv")

# Calculate the count of each item across all transactions
count=df.loc[:,:].sum()

#To know the most popular item in dataset
pop_item=count.sort_values(ascending=False).head(10)

#Converting the series of most popular item into dataframe
pop_item=pop_item.to_frame()

#resting the index of dataframe
pop_item=pop_item.reset_index()

#Renameing the index - item and 0 - count
pop_item=pop_item.rename(columns={'index':'items',0:'count'})
pop_item

#Data Visuallization
#Setting the runtime configuration for figuresize
plt.rcParams['figure.figsize']=(10,6)
plt.style.use('dark_background')

#ploting the horizantal bargraph
pop_item.plot.barh()
plt.title('most popular books')
plt.gca().invert_yaxis()

#Using Appriori algorithm
frequent_itemsets=apriori(df,min_support=0.09,max_len=5,use_colnames=True)
frequent_itemsets

#Sorting the values to know most support item
frequent_itemsets.sort_values('support',ascending=False,inplace=True)
frequent_itemsets

# Generate association rules from the frequent itemsets using the lift metric
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=1)
rules.head(10)
rules.sort_values('lift',ascending=False).head(10)

def to_list(i):
    return(sorted(list(i)))
ma_x=rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)

# Sort the merged list of items (transactions)
ma_x=ma_x.apply(sorted)

# Convert the merged list of transactions into a list of lists
rules_sets = list(ma_x)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules=[]

# Iterate over unique item sets and capture their indexes in the original list of rules_sets
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))
    
# Display the list of indexes of unique item sets
index_rules

rules_no_reduandancy=rules.iloc[index_rules,:]
rules_no_reduandancy 

rules10=rules_no_reduandancy.sort_values('lift',ascending=False).head(10)
rules10

#pLoting the scatterplot
rules10.plot(x='support',y='confidence',c=rules10.lift,kind='scatter',s=12,cmap=plt.cm.coolwarm)

# Convert 'antecedents' and 'consequents' columns to string type to facilitate string operations
rules10['antecedents'] = rules10['antecedents'].astype('string')
rules10['consequents'] = rules10['consequents'].astype('string')

# Remove the prefix "frozenset({" from the 'antecedents' column
rules10['antecedents'] = rules10['antecedents'].str.removeprefix("frozenset({")

# Remove the suffix "})" from the 'antecedents' column
rules10['antecedents'] = rules10['antecedents'].str.removesuffix("})")

# Remove the prefix "frozenset({" from the 'consequents' column
rules10['consequents'] = rules10['consequents'].str.removeprefix("frozenset({")

# Remove the suffix "})" from the 'consequents' column
rules10['consequents'] = rules10['consequents'].str.removesuffix("})")

