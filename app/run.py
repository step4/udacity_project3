import json
import plotly
import pandas as pd
import re

import nltk
nltk.download(['punkt','averaged_perceptron_tagger','stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# extract data needed for visuals
df_categories =  df.drop(columns=['id','message','original','genre'])
categories_count=list(df_categories.sum(axis=0))
categories_names = list(df_categories.columns)

df_messages = df['message']
token_count = list(df_messages.apply(lambda x: len(tokenize(x))))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_count
                )
            ],

            'layout': {
                'autosize':False,
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':45
                },
                'margin':{
                    'l':75,
                    'r':50,
                    'b':200,
                    't':50,
                    'pad':4
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=token_count
                )
            ],

            'layout': {
                'autosize':False,
                'title': 'Distribution of message length',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message length",
                    'tickangle':45
                },
                'margin':{
                    'l':75,
                    'r':50,
                    'b':200,
                    't':50,
                    'pad':4
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()