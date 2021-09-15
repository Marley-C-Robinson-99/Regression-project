###################################         IMPORTS         ###################################
import pandas as pd
import os
from acquire import acquire_zillow

from datetime import date
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###################################         PREPARE FUNCS         ###################################
def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def yearly_tax(df):
    ''' 
    Creates a rounded yearly_tax feature
    '''
    # Getting current year
    curr_year = int(f'{str(date.today())[:4]}')

    # Creating column
    df['yearly_tax'] = df.tax_value / (curr_year - df.year_built)

    df.yearly_tax = round(df.yearly_tax.astype(float), 0)

def month_sold(df):
    '''
    Creates a month sold feature
    '''
    # Converting date to string for splitting in month_sold function
    df['sale_date'] = df['sale_date'].astype(str)

    # Splitting the date to select month into a df
    month_sold = df.sale_date.str.split(pat ='-', expand = True)
    
    # Creating month col
    df['month_sold'] = month_sold[1] # grabs date col of df
    
    # Replaces month numbers with strings
    # df['month_sold'] = df['month_sold'].replace(['05', '06', '07', '08'], ['may', 'jun', 'jul', 'aug'])
    
    # Recasting sale_date as datetime int
    df['sale_date'] = df['sale_date'].astype('datetime64')
    df['month_sold'] = df['month_sold'].astype(int)
    return df

def impute_zillow(df):   
    # impute year built using mode
    imp = SimpleImputer(strategy='most_frequent')  # build imputer

    imp.fit(df[['year_built']]) # fit to train

    # transform the data
    df[['year_built']] = imp.transform(df[['year_built']])
    return df


###################################         PREPARE/SPLIT         ###################################


def prepare_zillow(df = acquire_zillow()):
    ''' Prepare zillow data for exploration'''
    
    # list of non-object cols
    cols = []
    for col, vals in df.iteritems():
        if df[f'{col}'].dtype != object:
            cols.append(col)

    # removing outliers
    df = remove_outliers(df, 3, cols)
    
    df['county'] = df['fips'].replace([6037.0, 6059.0, 6111.0], ['Los Angeles', 'Orange', 'Ventura'])
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)


    df = impute_zillow(df)

    # creating yearly_tax and month_sold
    yearly_tax(df)
    month_sold(df)

    # Dummy vars for month_sold
    # df = pd.get_dummies(data = df, columns = ['month_sold'])

    # dropping sale_date and yearly_tax to prevent data leakage
    df = df.drop(columns = ['sale_date', 'yearly_tax'])
    
    # tax_rate column
    tax_rate =  df.taxamount / df.tax_value

    df['tax_rate'] = tax_rate
    return df


def tvt_split(df = prepare_zillow(), target = 'tax_value', seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for X_y spit if one is provided), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 


    The function returns, in this order, train, validate and test dataframes. 
    Or, if target is provided, in this order, X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    if target == None:
        X = df[[col for col in df.drop(columns = ['taxamount', 'tax_rate']).columns if df[col].dtype != object]]
        train_validate, test = train_test_split(X, test_size=0.2, 
                                                random_state=seed)
        train, validate = train_test_split(train_validate, test_size=0.3, 
                                                random_state=seed)
        return train, validate, test
    else:
        X = df[[col for col in df.drop(columns = ['taxamount', 'tax_rate']).columns if df[col].dtype != object]].drop(columns = target)
        y = df[target]
        X_train_validate, X_test, y_train_validate, y_test  = train_test_split(X, y, 
                                    test_size=0.2, 
                                    random_state=seed)
        X_train, X_validate, y_train, y_validate  = train_test_split(X_train_validate, y_train_validate, 
                                    test_size=.3, 
                                    random_state=seed)
        return X_train, X_validate, X_test, y_train, y_validate, y_test
