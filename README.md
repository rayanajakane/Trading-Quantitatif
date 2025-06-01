# Machine Learning et Data Science appliqués au Trading Quantitatif
 
Ce projet a été réalisé dans le cadre du cours **IND8123 - Technologies Financières pour Ingénieurs** à Polytechnique Montréal.

L’objectif principal de ce projet est de se mettre dans la peau d’un trader quantitatif et d’utiliser différents **indicateurs techniques** ainsi que des **modèles de machine learning** pour prédire l’évolution de titres boursiers (*Google, Microsoft, Apple et Tesla*).

Le projet met en pratique :
- L’analyse d’indicateurs techniques classiques (RSI, bandes de Bollinger, Chaikin Money Flow, etc.)
- La création de règles de trading algorithmique simples
- L’entraînement et la comparaison de plusieurs modèles d’apprentissage automatique pour la prévision des rendements boursiers à court terme

---

## Données utilisées

- **Rendements boursiers** de Google, Microsoft, Apple et Tesla
- Indice de volatilité VXN (NASDAQ)
- Indicateurs de crédit
- Indicateurs de la pente des taux d’intérêt à court terme

---

## Objectifs du projet

1. **Tester des indicateurs techniques** sur plusieurs actions
   - Déterminer si certains signaux peuvent générer des profits à 1 jour ou 5 jours
2. **Développer des règles de trading**
   - Exemples : RSI, bandes de Bollinger, Ultimate Oscillator, etc.
3. **Comparer la performance des indicateurs** sur des titres boursiers différents
4. **Mettre en place des modèles de prévision** pour le marché
   - Logit (régression logistique)
   - Arbre de classification
   - Random Forest
   - Gradient Boosting
5. **Appliquer le trading basé sur les modèles entraînés** pour différents titres boursiers
6. **Comparer la performance** des modèles et des règles de trading classiques

---

## Structure du notebook

1. **Prétraitement des données**  
   Nettoyage, transformation, création des indicateurs techniques.
2. **Évaluation des règles de trading**  
   Application de différentes stratégies simples et analyse des résultats sur Google.
3. **Comparaison des indicateurs**  
   Analyse comparative entre les performances des indicateurs techniques sur Apple et Google.
4. **Développement d’une règle de trading sur l’indicateur [Ultimate Oscillator](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ultimate-oscillator)**
5. **Machine Learning**
   - Séparation en jeux d’entraînement/test
   - Entraînement et évaluation des 4 modèles (matrices de confusion, métriques de performance)
6. **Trading basé sur les modèles entraînés**  
   Application des signaux de trading générés par les modèles sur les différentes actions.
7. **Analyse des résultats**  
   Comparaison entre les approches et discussion sur la pertinence des méthodes quantitatives dans un contexte réel de trading.

---

## Technologies et librairies

- [`NumPy`](https://numpy.org/), [`Pandas`](https://pandas.pydata.org/) – Manipulation et analyse des données
- [`Matplotlib`](https://matplotlib.org/), [`Seaborn`](https://seaborn.pydata.org/) – Visualisation des données et des résultats
- [`Technical Analysis Library`](https://technical-analysis-library-in-python.readthedocs.io/en/latest/) – Calcul des indicateurs techniques (momentum, tendance, volatilité, volume)
- [`scikit-learn`](https://scikit-learn.org/stable/)
  - [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) – Normalisation et mise à l’échelle des données
  - [`sklearn.experimental.enable_halving_search_cv`](https://scikit-learn.org/stable/modules/generated/sklearn.experimental.enable_halving_search_cv.html) – Recherche de grille avec réduction adaptative
  - [`sklearn.model_selection`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) – Découpage des jeux de données, validation croisée, recherche d’hyperparamètres
    - `train_test_split`, `TimeSeriesSplit`, `GridSearchCV`, `HalvingGridSearchCV`, `StratifiedKFold`
  - [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) – Régression logistique (Logit)
  - [`sklearn.tree.DecisionTreeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) – Arbre de classification
  - [`sklearn.tree.plot_tree`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html) – Visualisation d’arbre
  - [`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) – Forêt aléatoire
  - [`sklearn.ensemble.GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) – Gradient Boosting
  - [`sklearn.metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) – Évaluation des modèles
    - `confusion_matrix`, `classification_report`, `roc_curve`, `auc`, `roc_auc_score`, `accuracy_score`
- [`yfinance`](https://pypi.org/project/yfinance/) – Téléchargement des données financières
- [`pandas_datareader`](https://pandas-datareader.readthedocs.io/en/latest/) – Accès à des données financières externes

---

## Exemple de résultats

- **Analyse des indicateurs techniques**  
  Certains indicateurs (comme le RSI ou les bandes de Bollinger) ont généré des signaux rentables sur certains titres et périodes, mais leur efficacité varie selon le titre et le contexte de marché.
- **Performance des modèles de machine learning**  
  Les modèles Random Forest et Gradient Boosting ont montré de meilleures performances de classification que les règles de trading simples, notamment pour la prévision sur 5 jours. Cependant, l’écart de performance reste modéré en raison de la complexité et de la volatilité des marchés financiers.
- **Comparaison Apple vs Google**  
  Les indicateurs techniques n’ont pas le même impact selon le titre analysé : un indicateur pertinent sur Apple peut être inefficace sur Google, ce qui met en avant l’importance du tuning et de l’analyse spécifique à chaque titre.
- **Trading basé sur les modèles**  
  L’utilisation des signaux issus des modèles de machine learning permet d’automatiser la prise de décision et d’améliorer la réactivité du trading, mais l’avantage peut être limité par l’overfitting et la non-stationnarité des marchés.

## Exemples d’outputs visuels

### Graphique de performance de trading selon l'indicateur RSI Credit sur 5 jours
![image](https://github.com/user-attachments/assets/4e0aa2ac-1534-4eab-9af6-8f32c893d240)

### Graphique de performance de trading par Régression Logistique
![image](https://github.com/user-attachments/assets/2c4727bb-4ab5-4061-abb4-9d369b092150)

### Exemple de matrice de confusion
![image](https://github.com/user-attachments/assets/b13fb1a1-97d7-469f-bb86-c4556310339c)

### Exemple d'arbre de décision
![image](https://github.com/user-attachments/assets/39fd012b-a334-4885-a8c3-4fbb581233b1)

---

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour les détails.

---
