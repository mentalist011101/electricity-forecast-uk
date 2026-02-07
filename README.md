# UK Electricity Load Forecasting  
<!-- IMAGE PLACEHOLDER -->
![Project overview](assets/forecast_overview.png)

## 1. Contexte et motivation

La prévision de la consommation électrique est un problème central pour la planification énergétique, la gestion des réseaux et la détection d’événements anormaux.  
Ce projet vise à construire un **pipeline complet de forecasting en séries temporelles**, basé sur des données réelles du Royaume-Uni, en mettant l’accent sur :

- la rigueur méthodologique,
- l’interprétabilité des modèles,
- et la capacité d’industrialisation (temps réel, monitoring, visualisation).

Le projet s’inscrit dans la continuité de la certification **Time Series – Kaggle Learn**, mais dépasse volontairement le cadre d’un notebook pédagogique.

---

## 2. Données

- **Pays** : United Kingdom  
- **Source** : ENTSO-E Transparency Platform (via Kaggle)  
- **Variable cible** : consommation électrique (`load`)  
- **Fréquence initiale** : 30 minutes  
- **Période couverte** : 2014-12-31 → 2020-09-30  

### Prétraitements principaux
- conversion des timestamps en `datetime`
- vérification de la régularité temporelle
- gestion des journées incomplètes
- agrégation **journalière** par somme
- analyse et traitement des valeurs manquantes

Le choix de l’agrégation journalière est motivé par un compromis entre stabilité statistique, interprétabilité et horizon de prévision.

---

## 3. Problématique formulée

> Prévoir la consommation électrique journalière du Royaume-Uni à horizon glissant multi-steps (1 à 7 jours), tout en permettant la détection d’anomalies et une exploitation temps réel.

Contraintes imposées :
- modèles de **Machine Learning interprétables**
- pas de fuite temporelle
- pipeline défendable académiquement et industriellement

---

## 4. Approche méthodologique

### 4.1 Décomposition de la série
La série est décomposée à l’aide de **STL (Seasonal-Trend decomposition using Loess)** :

- Tendance (trend)
- Saisonnalité hebdomadaire
- Résidu

La décomposition permet :
- d’isoler la dynamique structurelle,
- de rendre le problème de ML plus stationnaire,
- de clarifier la détection d’anomalies.

### 4.2 Vérifications statistiques
- Analyse visuelle (trend, seasonal, residual)
- Autocorrélation des résidus
- Test de stationnarité (ADF) sur le résidu

Le modèle ML n’est entraîné **que sur le résidu**, une fois la structure expliquée.

---

## 5. Modélisation

### 5.1 Features utilisées
- retards (lags) : 1, 7, 14 jours
- statistiques glissantes (moyenne, écart-type)
- variables calendaires (week-end, jour de l’année)

### 5.2 Stratégie de prévision
- **Direct Multi-Horizon Forecasting**
- un modèle par horizon (1 à 7 jours)
- pas de récursivité

### 5.3 Modèle
- Régression linéaire régularisée (Ridge)
- choix motivé par :
  - l’interprétabilité,
  - la stabilité,
  - la facilité d’industrialisation

---

## 6. Reconstruction de la prévision

La prévision finale est obtenue par :


Cette approche permet de conserver une lecture claire de chaque composante du signal.

---

## 7. Détection d’anomalies

Les anomalies sont détectées **exclusivement sur le résidu**, à l’aide d’un z-score glissant :

- fenêtre temporelle fixe
- seuil explicite
- méthode robuste et auditable

Cette définition permet de distinguer :
- variations normales,
- chocs ponctuels,
- ruptures de comportement inattendues.

---

## 8. Évaluation

- Métrique principale : **MAE**
- Comparaison avec un baseline naïf saisonnier
- Analyse de l’évolution de l’erreur avec l’horizon

Les résultats montrent :
- une amélioration nette par rapport au baseline,
- une augmentation progressive de l’erreur avec l’horizon, conforme à la théorie.

---

## 9. Visualisation et déploiement

Deux interfaces sont proposées :

- **Streamlit** : dashboard structuré (forecast, STL, anomalies)
- **Gradio** : démonstration rapide et interactive

L’architecture du projet est pensée pour un passage vers un environnement de production.

---


## 10. Limites et perspectives

Limites actuelles :
- absence de variables exogènes (météo, prix, événements)
- agrégation journalière (perte d’information intra-journalière)

Perspectives :
- ajout de données météorologiques
- comparaison avec SARIMAX
- extension mensuelle long terme
- déploiement temps réel avec monitoring automatique

---

## 11. Auteur

Projet réalisé par **Luciano Fokouo**,  
étudiant en Data Science, avec un intérêt particulier pour les séries temporelles, l’énergie et l’industrialisation des modèles ML.


