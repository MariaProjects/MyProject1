#!/usr/bin/env python
# coding: utf-8

# ## We have to user tab to have the next line in quoting blocks
# ## end the line with two spaces follow by Return 
# 

# >I hope this email finds you well. I'm reaching out to check on the status of your laptop after I had the opportunity to work on it last [Date]. Your satisfaction is our top priority, and I want to ensure that you're happy with the service provided.
# 
# >**Could you please take a moment to let me know if you have experienced any issues or difficulties with your laptop since the repair? 
#     >>If so, I'd be more than happy to assist you further and address any concerns you may have. 
#     >>>We take pride in our work and strive to deliver the best possible service to our customers.**
# 
# 1. If you have any feedback on our service, we would greatly appreciate it. Your input is valuable and helps us improve our offerings for future customers.
#     1. Please don't *hesitate to reach out to me directly at [Your Email] or give me a call at [Your Phone Number] if you need any assistance or have any questions*. I'll be more than happy to help.
#     2. Thank you for choosing us to take care of your laptop, and we look forward to serving you in the future.
#     
#     
# >test  
# test  
# test

# ___

# 2.1 Flower

# ![_][fly]
# 
# [fly]:https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Cimicifuga_ramosa_-_spikes.jpg/100px-Cimicifuga_ramosa_-_spikes.jpg "myfly"

# [Section title](#2.1-Flower)

# Monday|Tuesday|Wed
# -|-|-
# 1|3|7
# 4|6|7

# In[4]:


import numpy as np
import pandas as pd


# In[12]:


a = '1234567891011121314151617181920'
a[:10]


# In[16]:


a[:-10]


# In[17]:


a[5:10]


# In[18]:


a[::-1]


# In[20]:


b = 'monday is slow'
b.upper()


# In[21]:


b.lower()


# In[25]:


a.count('1')


# In[29]:


c = ['alt', 'tab', 'plus', 'minus']
c.remove('minus')
c


# ## List

# In[9]:


import numpy as np
import pandas as pd


# In[16]:


list1 = ['This are my numbers', 'and', 'this are not']
list2 = ['1', '2', '3', '4', '5']
list1.append('wait for answer')
list1


# In[17]:


list3 = list1 + list2
list3


# ## Tuple

# In[18]:


nature = ('mountens', 'sky', 'rivers', 'lakes', 'desert', 'beach')
nature


# In[19]:


string = ('This is my list')
len(string)


# ## We cannot append something to a tuple. We have to convet it to list then append then convet it back to tupel  
# >>nature.append('flowers')  
# >>>nature  

# In[34]:


nature_list = list(nature)
nature_list.append('flowers')
nature_list
nature_list = tuple(nature_list)
print(nature_list)


# In[22]:


string2 = tuple(string)
print(string2)


# In[24]:


string2.count('i')


# In[25]:


len(string2)


# In[35]:


*rest,a, b, c, d, e, f, g = string2
print(a, b, c, d, e, f, g)


# In[40]:


list3.append('5')
print(list3)


# In[42]:


list3.remove('5')
list3


# In[46]:


list3.remove('5')
list3


# In[47]:


list3.insert(4, 0)
list3


# In[49]:


list3.remove(0)


# In[50]:


list3.insert(4, '0')
list3


# In[52]:


list3[::-3]


# ## Dictionary 

# In[53]:


my_dictionary = {'sky':'blue', 'grass':'green', 'whater':'turcoz', 'roks':'gray', 'treen':'brown'}
my_dictionary


# In[55]:


my_dictionary['sky']


# In[56]:


my_dictionary.keys()


# In[57]:


my_dictionary.values()


# In[59]:


my_dictionary.items()


# ## Control flow statements

# In[60]:


sky = ['blue', 'gray', 'oprage', 'pink', 'purple']
for x in sky:
    print(x)


# In[62]:


for x in 'blue':
    print(x)


# In[64]:


sky = ['blue', 'gray', 'orange', 'pink', 'purple']
for x in sky:
    print(x)
    if x == 'orange':
        break


# In[66]:


sky = ['blue', 'gray', 'orange', 'pink', 'purple']
for x in sky:
    if x == 'orange':
        continue
    print(x)


# In[67]:


for x in range(7):
    print(x)
else:
    print('done')


# In[68]:


for x in range(5, 10):
    print(x)


# In[69]:


for x in range(1, 20, 2):
    print(x)


# In[74]:


A = ['Blue', 'Green', 'Brown']
B = ['Sky', 'Sea', 'Mounten']

for x in A:
    for y in B:
        print(x, y)
   


# In[81]:


number = 7
for i in range(number):
    print('*' * (i + 1))


# In[84]:


number = 7
for i in range(number):
    for j in range(i + 1):
        print('*', end="")
   


# In[85]:


number = 7
for i in range(number):
    for j in range(i + 1):
        print('*', end="")
    print('')


# In[86]:


number = 7
for i in range(number):
    for j in range(i - 1):
        print('*', end="")
    print('')


# In[123]:


j = 7
while j > 0:
    if j == 0:
       print('done')
    else:
        print(j)
    j -=1


# In[87]:


i = 1
while i < 6:
    print(i)
    i += 1
else:
    print('i is not longer less then 6')


# ## Arrays 
# > used for same type of data - a lot faster then list for calculation

# In[92]:


my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
my_array = np.array(my_list)
my_array


# #### Adding two arrays will add each element from one array to corresponding element of the other array

# In[93]:


second_array = ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x = my_array + second_array
x


# In[94]:


x.shape


# In[98]:


np.delete(x, [9])


# In[99]:


np.mean(x)


# In[100]:


np.max(x)


# In[101]:


d = np.arange(10, 25, 5)


# In[102]:


d


# In[105]:


e = np.full((4, 4), 5)
e


# In[106]:


f = np.eye(7)
f


# In[107]:


f.shape


# In[108]:


len(f)


# In[109]:


f.ndim


# In[110]:


f.astype(int)


# In[111]:


f.sum()


# In[112]:


f.max()


# In[114]:


f[0:5, 5]
# second 5 stends for column


# In[115]:


f[:: -1]


# ## Print

# In[116]:


print("""
Rutine:
\t-Eat
\t-Sleep\n\t-Repeat
""")


# In[117]:


"%f" % 3.1234567890


# In[118]:


"%.2f" % 3.1234567890


# In[120]:


"%020.2f" % 3.1234567890


# ## DATA Ingestion into Jupiter Notebook

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dfcsv = pd.read_csv('C:/Users/maria/Documents/Data science/Files for analysis/train.csv')
dfcsv


# In[3]:


# SAVE CSV Documents
dfcsv.to_csv('C:/Users/maria/Documents/Data science/Files for analysis/trainCopy.csv')


# In[9]:


dfexcel = pd.read_excel('C:/Users/maria/Documents/Data science/Files for analysis/titanic.xls')
dfexcel


# In[7]:


dfexcel.to_excel('C:/Users/maria/Documents/Data science/Files for analysis/titanicCopy.xls')


# ## Connect to Databese 

# In[10]:


import sqlite3


# In[12]:


conn = sqlite3.connect('C:/Users/maria/Documents/Data science/Database Analysis/Chinook_Sqlite.sqlite')
cur = conn.cursor()
cur.execute('select * from Album ;')
results = cur.fetchall()
print(results)
cur.close()


# In[15]:


cur = conn.cursor()
cur.execute('select artistId from Album ;')
results = cur.fetchall()
print(results)
cur.close()
conn.close()


# In[17]:


import sqlite3 
conn = sqlite3.connect("C:/Users/maria/Documents/Data science/Database Analysis/pythondb1.db")
cur = conn.cursor()
cur.execute('create table if not exists project_test (id integer PRIMARY KEY, name text NOT NULL, begine_date text, end_date text);')
cur.execute('create table if not exists task_test (id integer PRIMARY KEY, name NOT NULL, priotity integer, status_id integer NOT NULL, project_id integer NOT NULL, end_date text NOT NULL, FOREIGN KEY (project_id) REFERENCES projects(id));')
conn.commit()


# In[24]:


import sqlite3 
conn = sqlite3.connect("C:/Users/maria/Documents/Data science/Database Analysis/pythondb1.db")
cur = conn.cursor()
cur.execute("INSERT INTO tasks(name,priority,status_id,project_id,begin_date,end_date)VALUES('Analyze the requirements of the app', 1, 1, 1, '2015-01-01', '2015-01-02');")

cur.execute("INSERT INTO tasks(name,priority,status_id,project_id,begin_date,end_date)VALUES('Confirm with user about the top requirements', 1, 1, 1, '2015-01-03', '2015-01-05');")
conn.commit()


# In[22]:


cur.execute('insert into project_test(name, begine_date, end_date) Values ("my app", "2015-01-01", "2015-01-30");')
conn.commit()


# In[25]:


cur.execute('update project_test SET name = "Second app", begine_date = "2020-01-01", end_date = "2023-01-01" where id = 5;')
conn.commit()


# ## DATA INGESTION using website

# In[26]:


import bs4 as bs
import urllib.request
import pandas as pd
dfhtml = pd.read_html('https://www.census.gov/quickfacts/troycitymichigan')
dfhtml

for df in dfhtml:
    print(df)


# ## Import xml data

# In[33]:


from xml.dom import minidom
import pandas as pd
doc = minidom.parse('C:/Users/maria/Documents/Data science/Files for analysis/test.xml')

name = doc.getElementsByTagName("name")[0]
print(name.firstChild.data)

staffs = doc.getElementsByTagName("staff")
for staff in staffs:
    sid = staff.getAttribute("id")
    name = staff.getElementsByTagName("name")[0]
    expense = staff.getElementsByTagName("expense")[0]
    print("id:%s, name:%s, expences:%s" % (sid, name.firstChild.data, expense.firstChild.data) )
    


# ## Inserting data to SQLite

# In[34]:


import pandas as pd
import numpy as np
df = pd.read_csv('C:/Users/maria/Documents/Data science/Files for analysis/coviddata.csv')
df


# In[35]:


# Check for missing data
df.count()


# In[36]:


# pick only a single country 
save_df = df[df['countriesAndTerritories']=="United_States_of_America"]
save_df


# In[37]:


save_df.count()


# In[38]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///C:/Users/maria/Documents/Data science/Database Analysis/importdata.db', echo=True)
sqlite_connection = engine.connect()


# In[39]:


sqlite_table = "Mytest"
save_df.to_sql(sqlite_table, sqlite_connection, if_exists='fail')


# In[ ]:


sqlite_connection.close()


# ## Second way of importing data into SQL

# In[ ]:


import sqlite3
conn =sqlite3.connect('C:/Users/maria/Documents/Data science/Database Analysis/importdata.db')


# In[ ]:


sqlite_table = 'name_your_table'
file_df.to_sql(sqlite_table, conn, if_exists='fail')


# # Pandas
# ## Conditional statements

# In[43]:


# Find the value of a column when a condition is met
save_df['deaths_weekly'][save_df['notification_rate_per_100000_population_14-days']==save_df['notification_rate_per_100000_population_14-days'].min()]


# ## Replace NaN with specific value

# In[45]:


save_df.fillna(0, inplace=True)
save_df


# ## SEE STATISTICS of DataFrame

# In[48]:


save_df.describe()


# In[51]:


# Geta df where deaths > then 10000
d1 = save_df[save_df['deaths_weekly']>10000]
d1


# In[52]:


d1.shape


# ## Changing the Index

# In[56]:


# Pandas has its own idexing by default form 0 to row number
# We can reset the index by set_index() function

save_df.set_index('dateRep', inplace=True)
save_df


# In[58]:


# We can reset index back to how it was before with reset_index() function
save_df.reset_index()
save_df


# # NumPy 
# ## Create random generated array

# In[59]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


# A seed is used when we need to genereate the same data again and again
np.random.seed(1)
arr = np.random.randint(1, 7, (5, 8))
arr


# In[63]:


# Select column 4
arr[:, 4]


# In[65]:


# Select row 4
arr[4, :]


# In[66]:


arr / 2 


# In[67]:


arr ** 2


# In[85]:


arr2 = arr % 3
arr2


# In[74]:


arr


# In[94]:


arr3 = np.random.randint(1, 7, (5, 8))
arr3
firstdata = pd.DataFrame(arr3)
firstdata


# In[95]:


arr4 = np.random.randint(1, 7, (5, 8))
arr4
seconddata = pd.DataFrame(arr4)
seconddata


# In[82]:


e1 = arr.std()
e1


# In[72]:


# the results [0, 3, 0, 5, 0] it tells us where the max number is position inside the array (ex position 0 on first row)
arr.argmax(axis=1)


# In[75]:


arr.argmax(axis=0)


# In[79]:


# This function will add the numbers (ex array [a, b, c, d]: cumsum: [a, a+b, a+b+c, a+b+c+d])
arr.cumsum()


# In[80]:


np.arctan(arr)


# In[81]:


np.percentile(arr, q=.5, axis=1)


# In[83]:


# get the location where the argument is true
np.where(e1 < arr)


# ## Merging DataFrames

# #### Concat

# In[96]:


# Concat can be used only when we have the same number of index in both data frames
df4 = pd.concat([firstdata, seconddata], ignore_index=True, axis=1)
df4


# In[97]:


df5 = pd.concat([firstdata, seconddata], ignore_index=True, axis=0)
df5


# In[99]:


df = pd.concat([firstdata, seconddata], keys=['one', 'two'])
df


# #### Merging - Using iner or outer join

# In[100]:


# Inner join - In this method you will get an intersetion of two dataframes with merged column.
temp1={
      "date":['01-02-12','03-02-12','04-02-12','05-02-12'],
      "event":['sunny','cold','cold','rainy'],
      "temp":[14,16,15,10]
}
temp=pd.DataFrame(temp1)
temp


# In[101]:


ws={
      "date":['01-02-12','03-02-12','04-02-12','05-02-12'],
      "event":['sunny','cold','cold','rainy'],
      "wind-speed":[12,10,9,14],
}
wind_speed=pd.DataFrame(ws)
wind_speed


# In[102]:


df = pd.merge(temp, wind_speed, on=['date', 'event'])
df


# In[103]:


# Outer join - This is just like union of two dataframe.The value which dont exist will contain NaN.
temp1={
      "date":['01-02-12','03-02-12','04-02-12','05-02-12'],
      "event":['sunny','cold','hot','sunny'],
      "temp":[14,16,15,10]
}
temp=pd.DataFrame(temp1)
temp


# In[104]:


ws={
      "date":['01-02-12','03-02-12','04-02-12','05-02-12'],
      "event":['sunny','cold','cold','rainy'],
      "wind-speed":[12,10,9,14],
}
wind_speed=pd.DataFrame(ws)
wind_speed


# In[105]:


# We have to add an additonal argument for outer joint: how ='outer'
df = pd.merge(temp, wind_speed, on=['date', 'event'], how='outer')
df


# ## Reshaping dataframe

# In[11]:


import pandas as pd
import numpy as np


# In[3]:


mydis={'Day':['M', 'T', 'W', 'Th', 'F', 'S', 'Sun'],
      'Troy':[1, 2, 3, 4, 5, 6, 7],
      'Detrit':[10, 20, 30, 40, 50, 60, 70],
      'Warren':[11, 12, 13, 14, 15, 16, 17]}
df=pd.DataFrame(mydis)
df


# In[15]:


df2=pd.melt(df, id_vars=['Day'], var_name='City', value_name='Temp')
df2


# In[17]:


df2.pivot(index='Day', columns='City')


# In[18]:


df2.pivot(index='City', columns='Temp')


# In[35]:


d = pd.pivot_table(df2, index='Day', columns='City', aggfunc='count')


# In[31]:


e = pd.pivot_table(df2, index='Day', columns='City', aggfunc='sum')


# In[32]:


f = pd.pivot_table(df2, index='Day', columns='City', aggfunc='mean')


# In[33]:


h=pd.pivot_table(df2, index='Day', columns='City', aggfunc='max')


# In[34]:


j=pd.pivot_table(df2, index='Day', columns='City', aggfunc='min')


# In[29]:


# (tested in notebook 31) to find out how may times the word rain appears in the weather table:
# count_rain = (myfile['column_name']=='rain').sum()

# or we can use this method"
# count_rain = myfile['Column_name'].value_counts()


# In[30]:


# create a pivot table that five us the sum of humidity for different type of weather:
# a = pd.pivot_table(myfile, index= 'type of weather', values ='humidity', aggfunc = np.sum)


# ## Combining data base on GROUP

# In[6]:


dfnba = pd.read_csv("C:/Users/maria/Documents/Data science/Files for analysis/nba.csv") 
dfnba


# In[7]:


team_group = dfnba.groupby('Team')
team_group.first()


# In[8]:


team_group.get_group('Chicago Bulls')


# In[42]:


team_group2 = dfnba.groupby(['Team', 'Position'])
team_group2.first()


# ## GroupBy: Split, Apply and Combine

# In[44]:


# your operation will be applied to all of group and return the result of all groups
team_group['Age'].min()


# In[45]:


r = dfnba.groupby('Team')
r['Salary'].agg([np.min, np.max, np.sum, np.mean, np.std])


# ## Handling Missing Data

# ### There are four ways we can handle missing data  
#     1) fillna() - will replace the NAN values with the value we choose
#     2) dropna() - will remove the rows that have Null values
#     3) forward fill: ffill - will fill the Null value with mirroring the value in front of it
#     4) backward fill: bfill - will fill the Null value with the values under the nule value
#     
#     Find missing data for entire table: df.isnull().sum().sum() or df.isnull().any().any()
#     Find missing data for specific: df.isnull().sum().int
#                                     df.isnull().any().bool

# In[9]:


dfnba2 = dfnba.fillna(0)
dfnba2


# In[48]:


dfnba3 = dfnba.dropna()
dfnba3


# In[10]:


dfnba4 = dfnba.ffill()
dfnba4


# In[50]:


dfnba5 = dfnba.bfill()
dfnba5


# In[52]:


print(dfnba.isnull().sum())


# In[53]:


df4 = dfnba.dropna()
df4


# In[55]:


print(df4.isnull().sum())


# ### Replacing specific values with other values

# In[51]:


# find the nule values
s = dfnba.isnull()
s


# In[55]:


null_counts = s.sum()
null_counts


# In[53]:


dfnba.count()


# In[56]:


# we can replace some vaues that dosesn't make sense like outliner:
#df2 = df.replace(["5000", "2000"], np.NaN)


# In[65]:


# first try to replace teh NaN value in the last row - NOT SUCESSFULL
dfnba7 = dfnba.replace({'Name': 'Benjamin J', 'Team': 'Boston Celtics', 'Number':'28', 'Position':'SF', 'Age':'21'})
dfnba7


# In[1]:


# If data has some latin charactes or UTF-8 characters then we will get an error and the table will not be desplayed. 
# In order to get the table we need to use: encoding ='utf-8' or encoding='latine'


# In[11]:


# Remove the extra characters that are prsent in the table next to a number. 
dfnba8=dfnba.replace({'Number':'[A-Za-z]', 'Age':'[A-Za-z]', 'Salary':'\$' }, "", regex=True)
dfnba8


# ### If your dataset contains data which is repeating more than once or you want to change some set of string in to number then you have to apply list mapping
# 

# In[67]:


# f = df.replace(['poor', 'average', 'good', 'excelent'], [4, 6, 8, 10])


# In[1]:


import pandas as pd
import numpy as np


# In[14]:


dfnba.shape


# In[15]:


dfnba.columns


# In[17]:


dfnba.iloc[: 10]


# In[18]:


dfnba.iloc[:, 5]


# In[19]:


dfnba.iloc[:,::-1]


# In[20]:


dfnba.loc[0:50, 'Age':'Salary']


# In[21]:


dfnba.loc[::-1, :]


# In[24]:


# Print all data where age is grater then 30
dfnba.loc[dfnba['Age']>30]


# In[26]:


new_dfnba = dfnba.drop('College', axis=1)


# In[34]:


age = dfnba.groupby('Age')
age.first()


# In[35]:


age['Salary'].agg(np.mean)


# In[37]:


table = pd.pivot_table(dfnba, values='Salary', index=['Age'], columns='Team', aggfunc=np.max)
table


# In[38]:


dfnba.isna()


# In[39]:


# dfnba['Salary'].fillna(0)


# ## Data Analysis

# In[40]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[42]:


df = pd.read_csv("C:/Users/maria/Documents/Data science/Files for analysis/data.csv")
df.head(5)


# In[43]:


df.tail(5)


# In[45]:


df.dtypes


# In[46]:


df = df.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors', 'Vehicle Size'], axis=1)
df.head()


# In[49]:


df2 = df.rename(columns={'Engine HP':'Engine','Engine Cylinders':'Cylinders', 'Transmission Type':'Transmission', 'Driven_Wheels':'Wheels','highway MPG':'Hayway_MPG', 'city mpg':'city_MPG'})
df2.head(5)


# In[50]:


new_df2 = df[df.duplicated()]
new_df2


# In[51]:


new_df2.count()


# In[57]:


sns.boxplot(x=df2['MSRP'])


# In[58]:


sns.boxplot(x=df2['Cylinders'])


# In[59]:


Q1 =df2.quantile(0.25)
Q3 =df2.quantile(0.75)
IQR = Q3 -Q1
print(IQR)


# In[61]:


df2 = df2[~((df2<(Q1 -1.5 *IQR)) |(df2>(Q3 + 1.5 * IQR))).any(axis=1)]
df2.shape


# In[64]:


## Plot different features against one another (scatter), against frequency (histogram)
df2.Make.value_counts().nlargest().plot(kind='bar', figsize=(8,3))
plt.title('Number of cars by make')
plt.ylabel('Number of cars')
plt.xlabel('Make');


# In[65]:


plt.figure(figsize=(8,2))
c=df2.corr()
sns.heatmap(c, cmap='BrBG', annot=True)
c


# ## Scatterplot

# In[67]:


fig, ax=plt.subplots(figsize=(10,6))
ax.scatter(df2['Engine'], df2['MSRP'])
ax.set_xlabel('Engine')
ax.set_ylabel('MSRP')
plt.show()


# ## Data Visualization

# In[1]:


import matplotlib
print(matplotlib.__version__)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np


# In[3]:


xpoints = np.array([0, 10])
ypoints = np.array([200, 500])

plt.plot(xpoints, ypoints)

plt.show


# In[4]:


xpoints = np.array([0, 10])
plt.plot(xpoints)

plt.show


# In[6]:


# Example Draw a line in a diagram from position (1, 3) to position (8, 10)

xpoints = np.array([5, 8])
ypoints = np.array([5, 10])

plt.plot(xpoints, ypoints)

plt.show()


# In[8]:


# Example : Draw two points in the diagram, one at position (5, 5) and one in position (8, 10)
xpoints = np.array([5, 8])
ypoints = np.array([5, 10])

plt.plot(xpoints, ypoints, 'o')

plt.show()


# In[12]:


# You can use the keyword argument marker to emphasize each point with a specified marker

ypoints = np.array([3, 8, 3, 10])
plt.plot(ypoints, '*-.b')
plt.show()


# In[21]:


ypoints = np.array([3, 8, 3, 10])
plt.plot(ypoints, marker = '*', ms = 30, mfc = 'g', mec = 'r', ls = 'dotted', linewidth = '10')
plt.show()


# In[22]:


# You can plot as many lines as you like by simply adding more plt.plot() functions
y1 = np.array([1, 5, 1, 4])
y2 = np.array([3, 5, 4, 10])
y3 = np.array([5, 7, 6, 10])
plt.plot(y1)
plt.plot(y2)
plt.plot(y3)
plt.show()


# In[23]:


#With the subplots() function you can draw multiple plots in one figure
# The subplots() function takes three arguments that describes the layout of the figure.

# The layout is organized in rows and columns, which are represented by the first and second argument.

# The third argument represents the index of the current plot

# plt.subplot(1, 2, 1)
# the figure has 1 row, 2 columns, and this plot is the first plot

# plt.subplot(1, 2, 2)
# the figure has 1 row, 2 columns, and this plot is the second plot


# In[25]:


y1 = np.array([1, 5, 1, 4])
y2 = np.array([3, 5, 4, 10])
plt.subplot(1, 2, 1)
plt.plot(y1, y2)

y3 = np.array([3, 5, 4, 10])
y4 = np.array([5, 7, 6, 10])
plt.subplot(1, 2, 2)
plt.plot(y3, y4)
plt.show()


# In[26]:


# We want a figure with 2 rows and 1 column (meaning that the two plots will be displayed on top of each other)
y1 = np.array([1, 5, 1, 4])
y2 = np.array([3, 5, 4, 10])
plt.subplot(2, 1, 1)
plt.plot(y1, y2)

y3 = np.array([3, 5, 4, 10])
y4 = np.array([5, 7, 6, 10])
plt.subplot(2, 1, 2)
plt.plot(y3, y4)
plt.show()


# In[27]:


#With Pyplot, you can use the scatter() function to draw a scatter plot.

# The scatter() function plots one dot for each observation. It needs two arrays of the same length, one for the values 
# of the x-axis, and one for values on the y-axis

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

plt.scatter(x, y)
plt.show()


# In[29]:


# Compare scatter Plots
#day one, the age and speed of 13 cars:
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y, color = 'hotpink')

#day two, the age and speed of 15 cars:
x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y, color = 'blue')

plt.show()


# In[33]:


# With Pyplot, you can use the bar() function to draw bar graphs

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y, color = 'green', width = 0.5)
plt.show()


# In[38]:


x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.barh(x, y, color = 'blue')
plt.show()


# In[39]:


# It is a graph showing the number of observations within each given interval.

x = np.random.normal(50, 10, 25)
print(x)


# In[40]:


x = np.random.normal(50, 10, 25)
plt.hist(x)
plt.show()


# In[48]:


y = np.array([35, 25, 25, 15])
mylables = ['Python', 'Java', 'C++', 'SQL']
mycolors = ['hotpink', 'blue', 'g', 'red']
myexplode = [0.2, 0.1, 0, 0]
plt.pie(y, labels = mylables, explode = myexplode, colors = mycolors)
plt.legend()
plt.show() 


# ## Merge three data frames

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df_meal = pd.read_csv('C:/Users/maria/Documents/Data science/Files for analysis/meal_info.csv')
df_meal


# In[4]:


df_food = pd.read_csv('C:/Users/maria/Documents/Data science/Files for analysis/train_food.csv')
df_food


# In[5]:


df_center = pd.read_csv('C:/Users/maria/Documents/Data science/Files for analysis/fulfilment_center_info.csv')
df_center


# In[6]:


df = pd.merge(df_food, df_center, on='center_id')
df = pd.merge(df, df_meal, on = 'meal_id')
df


# In[13]:


tot_num_orders = pd.pivot_table(data=df, index='category', values='num_orders', aggfunc=np.sum)
tot_num_orders


# In[15]:


import matplotlib.pyplot as plt
plt.style.use('seaborn')

plt.bar(tot_num_orders.index, tot_num_orders['num_orders'])
# when names overlaps on top of each other we can use the xticks() function: plt.xticks(rotation=90)
plt.xticks(rotation=90)
plt.xlabel('Food Items')
plt.ylabel('Quantity Sold')
plt.title('Most Sold Items')
plt.show()


# In[17]:


# nd the unique meals - what was ordered by itself and not with other foods
#dictionary of meals per food items
item_count ={}
for i in range(tot_num_orders.index.nunique()):
    item_count[tot_num_orders.index[i]] = tot_num_orders.num_orders[i]/df_meal[df_meal['category']==tot_num_orders.index[i]].shape[0]

plt.bar([x for x in item_count.keys()], [x for x in item_count.values()], color='red')
plt.xticks(rotation = 90)
plt.xlabel('Food item')
plt.ylabel('No of meals')
plt.title('Meals per food items')
plt.show()


# ## Profiling 

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling as pp
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


df = pd.read_csv("C:/Users/maria/Documents/Data science/Files for analysis/train.csv")
df


# In[22]:


import ydata_profiling as pp
report = pp.ProfileReport(df)


# In[24]:


# This will create a html file that will give us a description of the data 
report.to_file('seemydata.html')


# In[26]:


# get a visual of how many people have survived titanic
sns.countplot(x='Survived', data=df)


# In[28]:


# Find how many man and how many womans survived
sns.countplot(x='Sex', data =df)


# In[29]:


# Find which class of pasangers survived more
sns.countplot(x='Pclass', data=df)


# In[30]:


# find out how much pasangers enbarc in each port
sns.countplot(x='Embarked', data=df)


# In[31]:


# Histogram is the same as countplot but is use for numeric data
sns.distplot(df['Age'])


# In[32]:


# see how many pasangers payd for a ticket
sns.displot(df['Fare'])


# In[33]:


# Visualize multiple variables in the same plot
sns.countplot(x='Survived', hue='Sex', data=df)


# In[34]:


# see if the class the passangers were part of has any impact on survival rate
sns.countplot(x='Survived', hue='Pclass', data=df)


# In[35]:


# See a quantitative and a categorical item in the same plot
sns.boxplot(data=df, x='Survived', y='Age')


# In[38]:


# were passengers that paid a higher fair was more lickly to survive? 
sns.barplot(data=df, x='Survived', y='Fare')


# In[39]:


#what specific age group was more likely to survive the Titanic
sns.boxplot(data=df, x='Age', y='Sex', hue='Survived')


# In[40]:


# See the diference between class, fare and survival rate
sns.barplot(data=df, x='Pclass', y='Fare', hue='Survived')


# In[42]:


numeric = df.iloc[:, [0, 1, 5, 9]]
sns.pairplot(numeric.dropna(), hue='Survived')


# In[43]:


corr = df.corr()
sns.heatmap(corr)


# In[ ]:




