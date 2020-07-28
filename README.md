# Projet mini API Kaggle

### Objectif
Créez une API à laquelle on peut envoyer des prédictions faites sur un dataset de test donné, via un "submission file" ayant les mêmes spécifications que sur Kaggle, et qui renvoie le score sur le dataset en question. 

On se basera sur le challenge GMSC. Vous pouvez créer cette API avec Flask, ou un autre outil de l'écosystème Python. Pour obtenir un dataset de test sur lequel les output sont connues, vous pouvez partager le dataset de train en 2. Enregistrez le dataset de test dans un fichier test2.csv, et fournissez un fichier contenant des prédictions sur ce dataset, qu'on nommera test2-predictions.csv.

Les commandes suivantes seront exécutées afin de tester votre travail:

```
pip install -r requirements.txt

export FLASK_APP=api.py
flask run

curl --request POST \
  --url 'http://localhost:5000/submit' \
  --header 'accept: multipart/form-data' \
  -F 'file=@test2-predictions.csv'
```

Livrables:
* requirements.txt contenant les dépendances Python
* api.py contenant le code de l'API
* split.py contenant le code qui partage le dataset de train en 2 et permet de créer test2.csv
* test2.csv
* predict.py contenant le code qui génère test2-predictions.csv
* test2-predictions.csv