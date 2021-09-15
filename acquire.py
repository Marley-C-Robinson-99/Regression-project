###################################         IMPORTS         ###################################
import pandas as pd
import os
from env import host, user, password

from datetime import date
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
###################################         ACQUIRE         ###################################
# Function to establish connection with Codeups MySQL server, drawing username, password, and host from env.py file
def get_db_url(host = host, user = user, password = password, db = 'zillow'):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Function to acquire neccessary zillow data from Codeup's MySQL server
def acquire_zillow():
    filename = "raw_zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        SQL = '''
        SELECT bedroomcnt, 
            bathroomcnt, 
            calculatedfinishedsquarefeet, 
            taxvaluedollarcnt, 
            yearbuilt, 
            taxamount, 
            fips,
	        pr.transactiondate
        FROM properties_2017
        LEFT JOIN propertylandusetype 
	        USING(propertylandusetypeid)
        LEFT JOIN predictions_2017 as pr
	        USING(parcelid)
        WHERE propertylandusetypeid 
	        IN ('260', '261', '262', '263', '264', '273', '275', '276', '279')
	    AND transactiondate BETWEEN '2017-05-01' AND '2017-08-31'
        '''
        # read the SQL query into a dataframe
        df = pd.read_sql(SQL, get_db_url())

        # renaming cols
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',
                              'transactiondate':'sale_date'})


        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        # Return the dataframe to the calling code
        return df

###################################         BASIC SUMMARY         ###################################
def object_vals(df):
    '''
    This is a helper function for viewing the value_counts for object cols.
    '''
    for col, vals in df.iteritems():
        print(df[col].value_counts(dropna = False))
        print('----------------------')

def col_desc(df):
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    return stats_df

def null_cnts(df):
    for col in df.columns:
        print(f'{col}: {df[col].isna().sum()}')

def summarize_df(df):
    '''
    This function returns the shape, info, a preview, the value_counts of object columns
    and the summary stats for numeric columns.
    '''
    print(f'This dataframe has {df.shape[0]} rows and {df.shape[1]} columns.')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Info****')
    print(df.info())
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Null Counts****')
    null_cnts(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Value Counts****')
    object_vals(df)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('****Column Stats****')
    print(col_desc(df))