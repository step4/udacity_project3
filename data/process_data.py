# -*- coding: utf-8 -*-
# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load disaster messages and disaster categories from CSV files. Categories will be converted in seperate columns with values 1 and 0.

    Args:
    messages_filepath: string. The filepath to the messages CSV file as string.
    categories_filepath: string. The filepath to the categories CSV file as string.

    Returns:
    df: Pandas dataframe. The pandas dataframe containing all messages and categories in each row.
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories,how='outer',on=['id'])
    
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';',expand=True)
                                                    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda category: category[:-2]))
                                                    
    # rename the columns of `categories`
    categories.columns = category_colnames
         
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
                                                    
    # drop the original categories column from `df`
    df=df.drop(columns=['categories'])
                                                    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    return df


def clean_data(df):
    """Clean the dataframe containing the disaster messages and categories. Missing values and duplicates are dropped.

    Args:
    df: Pandas dataframe. The pandas dataframe containing all messages and categories in each row WITH duplicates and missing values.

    Returns:
    df: Pandas dataframe. The pandas dataframe containing all messages and categories in each row WITHOUT duplicates and missing values.
    """

    # drop duplicates
    df=df.drop_duplicates()
    
    # drop missing values
    df = df.dropna()

    return df


def save_data(df, database_filename):
    """Saving the dataframe as a specific sql file.

    Args:
    df: Pandas dataframe. The cleaned pandas dataframe containing all messages and categories.
    database_filename: sting. The path where the sql file should be saved.
    """
    
    table_name=database_filename.split("/")[1].split('.')[0]
    
    engine = create_engine(f'sqlite:///{database_filename}')
    
    df.to_sql(table_name, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
