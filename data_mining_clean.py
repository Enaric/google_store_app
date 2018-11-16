# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

d = pd.read_csv('googleplaystore.csv')
print(d.columns)

d = d.astype(str)
d.drop_duplicates(subset='App', inplace=True)
# d.dropna(axis=0, how='all')
# d.dropna(axis=0, how='any')
d = d[d['Android Ver'] != np.nan]
d = d[d['Android Ver'] != 'NaN']
d = d[d['Installs'] != 'Free']
d = d[d['Installs'] != 'Paid']
d = d[d['Size'] != 'Varies with device']
d = d[d['Size'] != '']

# d = d[d['Rating'] != np.NaN]
d = d[d['Rating'] != 'NaN']
d = d[d['Rating'] != 'nan']

print('Number of apps in the dataset : ', len(d))

d['Rating'] = d['Rating'].astype(float)

d['Installs'] = d['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
d['Installs'] = d['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
d['Installs'] = d['Installs'].apply(lambda x: int(x))

# d['Size'] = d['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
d['Size'] = d['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
d['Size'] = d['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
d['Size'] = d['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

d['Size'] = d['Size'].apply(lambda x: float(x))
d['Installs'] = d['Installs'].apply(lambda x: float(x))

d['Price'] = d['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
d['Price'] = d['Price'].apply(lambda x: float(x))

d['Reviews'] = d['Reviews'].apply(lambda x: int(x))


def get_index_value(name, x):
    a = d.groupby([name]).size().reset_index()
    for index, i in enumerate(a.values):
        if i[0] == str(x):
            return index
    return -1


d['Category Val'] = d['Category'].apply(lambda x: get_index_value('Category', x))
d['Type Val'] = d['Type'].apply(lambda x: get_index_value('Type', x))
d['Content Rating Val'] = d['Content Rating'].apply(lambda x: get_index_value('Content Rating', x))
d['Android Ver Val'] = d['Android Ver'].apply(lambda x: get_index_value('Android Ver', x))
#
dc = d[['Rating', 'Category Val', 'Rating', 'Reviews', 'Size', 'Installs', 'Type Val', 'Price', 'Content Rating Val', 'Android Ver Val']]
# print(dc)
# print(d[['Category', 'CategoryVal']])

dc.to_csv('googleplaystore_clean_val.csv')
