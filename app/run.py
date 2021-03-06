import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cat_list = []
    for col in df.drop(['id', 'original', 'genre'], axis=1).columns.tolist():
        if col != 'message':
            col_count = df.groupby(col)['message'].count().reset_index()
            if len(col_count[col_count[col] == 1]) > 0:
                col_count = col_count[col_count[col] == 1]['message'].iloc[0]
                cat_list.append((col, col_count))
            else:
                cat_list.append((col, 0))

    cat_names_many = [i[0] for i in cat_list if i[1] > 5000]
    cat_counts_many = [i[1] for i in cat_list if i[1] > 5000]
    trace1 = Scatter(
        x=cat_names_many,
        y=cat_counts_many,
        mode='markers',
        marker=dict(
            size=10, color='#ff9f43', symbol=18
        )
    )

    cat_names_few = [i[0] for i in cat_list if i[1] <= 5000]
    cat_counts_few = [i[1] for i in cat_list if i[1] <= 5000]
    trace2 = Scatter(
        x=cat_names_few,
        y=cat_counts_few,
        mode='markers',
        marker=dict(
            size=10, color='blue'
        )
    )

    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [trace1, trace2],

            'layout': {
                'font': {
                    'size': 10
                },
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
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
