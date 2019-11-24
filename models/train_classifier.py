# import libraries
import sys
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt','averaged_perceptron_tagger','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """Load disaster messages and disaster categories from a sqlite file.

    Args:
    database_filepath: string. The filepath to the sqlite file as string.

    Returns:
    df: Pandas dataframe. The pandas dataframe containing all messages and categories in each row.
    """
    
    # load data from database
    table_name=database_filepath.split("/")[1].split('.')[0]
    engine = create_engine(f'sqlite:///{database_filepath}')
    df=pd.read_sql(f"SELECT * FROM {table_name}", engine)
    
    X = df.message.values
    
    df_categories =  df.drop(columns=['id','message','original','genre'])
    Y =df_categories.values
    
    category_names=list(df_categories.columns)
    
    return X,Y,category_names


def tokenize(text):
    """Tokenize a string of text. Following steps a done: remove punctuation, split sentence in words, remove stop words, lemmatize wors, stem words, convert word to lowercase and remove all leading/trailing whitespace of word.

    Args:
    text: string. Text to tokenize.

    Returns:
    clean_tokens: list of strings. Each list item contains a clean token after tokenization.
    """
             
    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer =WordNetLemmatizer()
    stemmer = PorterStemmer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # stem word
        clean_tok = stemmer.stem(clean_tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build a Multi-Output-Classifier with Random-Forest. Data will be transformed with a CountVectorizer and TfidfTransformer.

    Returns:
    cv: SKLearn classifier. The SKLearn pipeline with the transformations and classifier.
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
             
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
#         'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 10000),
        'tfidf__smooth_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth': [None, 5],
#         'clf__estimator__max_features': [500,'auto']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,verbose=10, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate a classifier model with input and label test data by reporting f1 score, precision and recall for each output category of the dataset.

    Args:
    model: SKLearn model. The SKLearn model fit by training data.
    X_test: Numpy Matrix. The input test data.
    Y_test: Numpy Matrix. The label test data.
    category_names: List of strings. The list of category names for readable output.

    """
    
    Y_pred = model.predict(X_test)
    
    numberOfColumns = Y_pred.shape[1]

    for colIndex in range(0,numberOfColumns):
        y_true = Y_test[:,colIndex]
        y_pred = Y_pred[:,colIndex]
        print(f"category: {category_names[colIndex]}\n",classification_report(y_true, y_pred ,labels=[0,1]))


def save_model(model, model_filepath):
    """Save the trained SKLearn model in a pickle file.

    Args:
    model: SKLearn model. The SKLearn model fit by training data.
    model_filepath: string. The filepath where to save the pickle file.

    """
    model_pickle= pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()