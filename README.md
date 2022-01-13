# Disaster response project

### Table of contents

1. [Project motivation](#motivation)
2. [File descriptions](#files)
3. [Instructions](#instructions)

### Project motivation<a name="motivation"></a>

 I used a dataset containing real messages that were sent during disaster events to classify them (using machine learning) so that they can be sent to the appropriate disaster relief agencies.

### File descriptions <a name="files"></a>

CSV files contain disaster response messages ('data/disaster_messages.csv') and their classification into certain categories ('data/disaster_categories.csv').

Scripts prepare ('data/process_data.py') and model data ('models/train_classifier.py'). Their output is used on an web app ('app/run.py').

### Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001 (local machine).
![alt tag](https://github.com/colorxyz/Disaster-Response-Pipelines/tree/main/image/screenshot.png)
