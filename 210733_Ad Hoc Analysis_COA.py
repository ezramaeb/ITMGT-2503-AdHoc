#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json 
import matplotlib.pyplot as plt


# ## General Pivot Table 

# In[2]:


# Generate the pivot table showing ALL data from the DataFrame

file = "./transaction-data-adhoc-analysis.json"
df = pd.read_json(file)

df


# # I. Breakdown of the count of each item sold per month

# ## DataFrame I: Each Item Sold per Month

# In[3]:


# Dictionary for the months of the General Pivot Table
months = {'/01/':'January',
         '/02/':'February',
         '/03/':'March',
         '/04/':'April',
         '/05/':'May',
         '/06/':'June'}

for i in months:
    df.loc[df["transaction_date"].str.contains(i),"month"] = months[i]

def products_receipt(month):
    per_month = df[df["month"]==month]

    per_month['transaction_items']
    new_list = []
    for index, row in per_month.iterrows():
        x = (row["transaction_items"].split(";"))
        for i in range(0,len(x)):
            new_list.append(x[i].split(','))

    item_df = pd.DataFrame(new_list,columns=['brand','Items','quantity'])
    item_df['quantity'] = item_df['quantity'].str.extract('(\d+)',expand=False).astype(int)

    monthly_receipt=item_df.groupby('Items').sum().squeeze()
    return monthly_receipt

products_sold_df = pd.DataFrame({i:products_receipt(i) for i in list(months.values())})
products_sold_df['Total Items'] = products_sold_df['January'] + products_sold_df['February'] + products_sold_df['March'] + products_sold_df['April'] + products_sold_df['May'] + products_sold_df['June'] 

products_sold_df


# ### Bar Graph for DataFrame I: Each Item Sold per Month

# In[4]:


new_sold_df = products_sold_df.transpose()
new_sold_df.index.name = 'Month'
new_sold_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='')

new_sold_df=new_sold_df[['Month','Beef Chicharon','Gummy Vitamins','Gummy Worms','Kimchi and Seaweed','Nutrional Milk','Orange Beans','Yummy Vegetables']].iloc[:6]

new_sold_df.plot(x='Month',
        kind='bar',
        stacked=False,
        title='Quantity of Each Item Sold Per Month',
           figsize=(15,10),
           color = ['#FF9AA2', '#FFB7B2', '#FFDAC1','#E2F0CB', '#B5EAD7', '#C7CEEA', '#FFD1DC']);


# ### Pie Graph for DataFrame I: Each Item Sold per Month

# In[5]:


products = products_sold_df.index
values = products_sold_df['Total Items']

plt.figure(figsize=(12,8))
plt.pie(values, labels=products, autopct='%1.1f%%',labeldistance=1.15,
        colors = ['#FF9AA2', '#FFB7B2', '#FFDAC1','#E2F0CB', '#B5EAD7', '#C7CEEA', '#FFD1DC']);
plt.title('Total Items Sold',fontsize=15)
plt.show();


# In[6]:


import plotly.express as px

new_sold_df = products_sold_df.transpose()
new_sold_df.index.name = 'Month'

new_sold_df=new_sold_df.iloc[:6]
new_sold_df

fig = px.line(new_sold_df, 
              title='Items Sold Per Month')
fig


# # II. Breakdown of the total sale value per item per month

# In[7]:


price_chart = df[["transaction_items","transaction_value"]].drop_duplicates(subset=['transaction_items']).loc[(df['transaction_items'].str.contains(";") == False) & (df['transaction_items'].str.contains("x1"))]

def new_index(name):
    index_name = name[name.index(",")+1:name.index(",",name.index(",")+1)]
    return index_name

price_chart['Items'] = price_chart['transaction_items'].apply(new_index)
cost_per_item = price_chart.set_index('Items')['transaction_value']

total_sales_df = products_sold_df.copy(deep = True)
total_sales_df['Cost per Item'] = cost_per_item

for i in list(products_sold_df.keys()):
    total_sales_df[i]=total_sales_df['Cost per Item']*products_sold_df[i]
    

total_sales_df=total_sales_df[['Cost per Item','January','February','March','April','May','June']]
total_sales_df['Total Sales per Item'] = total_sales_df['January'] + total_sales_df['February'] + total_sales_df['March'] + total_sales_df['April'] + total_sales_df['May'] + total_sales_df['June']

# Total Monthly Sales Row

total_row = total_sales_df.sum().drop('Cost per Item')
total_sales_df.loc['Total Monthly Sales']=total_row
total_sales_df.replace(np.nan, '', regex=True,inplace=True)
total_sales_df = total_sales_df.astype(int,errors='ignore')

total_sales_df


# ## DataFrame II: Total Monthly Sales

# In[8]:


monthly_sales_df = total_sales_df.iloc[0:8,1:7]
monthly_sales_df


# In[9]:


new_month = monthly_sales_df.transpose() 
plt.figure(figsize=(12,8))
plt.pie(new_month['Total Monthly Sales'], labels=new_month.index, autopct='%1.1f%%',labeldistance=1.1,
        colors = ['#FF9AA2', '#FFB7B2', '#FFDAC1','#E2F0CB', '#B5EAD7', '#C7CEEA', '#FFD1DC']);
plt.title('Total Monthly Sales',fontsize=15)
plt.show();


# ## DataFrame II: Total Sales per Item

# In[10]:


item_sales_df = total_sales_df.iloc[0:7,1:8]
item_sales_df


# In[11]:


products = item_sales_df.index
values = item_sales_df['Total Sales per Item']

plt.figure(figsize=(12,8))
plt.pie(values, labels=products, autopct='%1.1f%%',labeldistance=1.1,
        colors = ['#FF9AA2', '#FFB7B2', '#FFDAC1','#E2F0CB', '#B5EAD7', '#C7CEEA', '#FFD1DC']);
plt.title('Total Sales per Item',fontsize=15)
plt.show();


# # III. Customer Loyalty

# ### A. Repeater: Number of customers from the current month who also purchased in the previous month

# In[12]:


months_list = ['January','February','March','April','May','June']

# Gets the index based on the months_list 
def repeater(month):
    month_index = months_list.index(month)

    # Refers to January
    if month_index == 0:
        past_month=None
    else:
        past_month = months_list[month_index-1:month_index][0]

    # DataFrame referring to customers who purchased during the current month
    current_df = df[df["month"]==month].drop_duplicates(subset=['name'])
    current_customers = set(current_df['name'])

    # DataFrame referring to customers who purchased during the past month
    past_df = df[df["month"]==past_month].drop_duplicates(subset=['name'])
    past_customers = set(past_df['name'])

    # Compares the set of customers who purchased during the current month and past month
    return len(current_customers & past_customers)


# Creates a dictionary for the customers who are repeaters per month
repeater_dict = {i:repeater(i) for i in months_list}
repeater_series = pd.Series(repeater_dict) 
repeater_series


# ### B. Inactive: Number of customers who have purchase history but do not have a purchase for the current month

# In[13]:


months_list = ['January','February','March','April','May','June']

# Gets the index based on the months_list 
def inactive(month):
    month_index = months_list.index(month)
    past_months = months_list[:month_index]

    # DataFrame referring to customers who purchased during the current month
    current = df[df["month"]==month].drop_duplicates(subset=['name'])
    current_customers = list(current['name'])

    # DataFrame referring to customers who purchased during the past month
    past = df[df["month"].isin(past_months)].drop_duplicates(subset=['name'])
    past_customers = list(past['name'])

    for name in current_customers:
        if name in past_customers:
            past_customers.remove(name)
        else:
            continue

    return len(past_customers)

inactive_dict = {i:inactive(i) for i in months_list}
inactive_series = pd.Series(inactive_dict) 
inactive_series


# ### C. Engaged Customers: Customers who consistently purchased every single month  

# In[14]:


months_list = ['January','February','March','April','May','June']

def engaged(month):
    
    # Creates a set pertaining to all customers who purchased per month
    def customers_per_month(month):
        engaged = sorted(set(df[df["month"]==month].drop_duplicates(subset=['name'])['name']))
        return engaged

      # Creates a set pertaining to all customers who purchased per month
    current_dict = {i:customers_per_month(i) for i in list(months.values())}

    i = 0
    mon1=months_list[0]
    engaged = set(current_dict[mon1])

    # Creates & gets the list of current customers
    while i <= months_list.index(month):
        current_month = months_list[i]
        current_list = set(current_dict[current_month])
        
        # Compares the set of engaged and current customers
        engaged = engaged & current_list
        i = i+1
        if i > months_list.index(month):
            
            # Shows the total at the end of a certain month 
            return len(engaged)

engaged_dict={i:engaged(i) for i in months_list}
engaged_series=pd.Series(engaged_dict)
engaged_series


# ### DataFrame III: Repeater, Inactive, Engaged

# In[15]:


rie_df = pd.DataFrame({'Repeaters':repeater_series,'Inactive':inactive_series,'Engaged':engaged_series}).transpose()
rie_df


# In[16]:


new_df = rie_df.transpose()
new_df.index.name = 'Month'
new_df.reset_index(level=None, drop=False, inplace=True, col_level=0, col_fill='')

new_df=new_df[['Month','Repeaters','Inactive','Engaged']]

new_df.plot(x='Month',
        kind='bar',
        stacked=False,
        title='Grouped Bar Graph with dataframe',
           figsize=(15,10),
           color = ['#F18387','#FFCEBB', '#9DDFDA']);


# In[ ]:




