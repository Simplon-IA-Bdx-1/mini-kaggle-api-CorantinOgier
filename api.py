import pandas as pd
import json, csv, webbrowser
from flask import Flask, request, jsonify, render_template


app = Flask(__name__, instance_relative_config=True)


@app.route('/submit',methods=['POST'])
def predict():

    
     
    data = pd.read_csv(request.files.get('file'))
    
    from sklearn.metrics import roc_auc_score
    y_true = data['SeriousDlqin2yrs'] 
    y_scores = data['Predictions'] 
    auc = roc_auc_score(y_true, y_scores)

    
    
        
    dictionnaire = {
        'type': 'Score',
         'AUC': auc
        }
    return jsonify(dictionnaire)


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000' 
    webbrowser.open_new(url)
    app.run(debug=True)