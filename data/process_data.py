import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data files and merge into a single dataframe.
    """

    # load messages dataframe
    messages = pd.read_csv(messages_filepath)

    # load categories dataframe
    categories = pd.read_csv(categories_filepath)

    # merge dataframe
    df = messages.merge(categories, on="id")

    return df

def clean_data(df):
    """
    Clean dataframe by rename colums, converting values to binary and
    dropping duplicates.
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # create category column name by spliting name and value from categories
    # ex> 'related-1' to  related
    category_colnames = row.apply(lambda x: x.split('-')[0])
    # rename categories dataframe
    categories.columns = category_colnames

    for column in category_colnames:
        # assign values for each category column names
        categories[column] = categories[column].apply(lambda x:x.split("-")[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # assumes that any value other than 0 is a 1 (force binary)
        categories[column] = np.where(categories[column] == 0, 0, 1)
    # drop the catagoies column from dataframe
    df=df.drop('categories',axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df=df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """
    Save dataframe to a database file.
    """
    path = ['sqlite:///', database_filename]
    engine = create_engine("".join(path))
    df.to_sql(name='DisasterResponse', con=engine, index=False, if_exists='replace')

def main():
    """
    Main function to execute load, clean and save files.
    """
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
