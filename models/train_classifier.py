import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """
    Load data from the database created with process_data.py
    """
    path = ['sqlite:///', database_filepath]
    engine = create_engine("".join(path))
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns.values)

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize, lemmatize and clean text messages.
    """
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailling while space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens    




def build_model():
    """
    Build model with pipe line and GridSearchCV with parameters
    """
    classifier = MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1)

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
        ('clf', classifier)
    ])
    parameters = parameters = {
    'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model outputs using classification_report.
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred)
    Y_pred_df.columns = category_names

    for category in category_names:
        print('\n#### {} ####\n{}\n'.format(category, classification_report(Y_test[category], Y_pred_df[category])))





def save_model(model, model_filepath):
    """
     Save model to a pickle file.
     """
    pickle.dump(model, open(model_filepath, 'wb'))
    


def main():
    """
    Main function to load data and buil/train/evaluate/save model
    """
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