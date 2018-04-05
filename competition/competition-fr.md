# Compréhension des communications

### Compétition d'apprentissage automatique


## Mise en contexte

Les pingouins se sont écrasés dans une métropole. Après un peu de travail, ils
ont réussi à se connecter à un réseau et à intercepter des communications.
Puisqu'ils sont des oiseaux, il ne sont pas capables de comprendre grand chose à
travers tous ces gazouillis. Malgré tout, ils remarquent quatre catégories de
communication:

- Media
- Businessman
- Politician
- Showbiz

Ils aimeraient concevoir un système pour classifier automatiquement
l'information dans ces catégories. Alors, les pingouins ont commencés à
étiqueter manuellement les données...


![pingouin](./img/pingouin.jpg)


## Description de la compétition

Vous avez à votre disposition des communications interceptées en format csv. Le
fichier `train.csv` a deux colonnes. La première colonne contient la
communication et la deuxième colonne contient l'étiquette de la communication
(un integer).

La correspondance entre la catégorie textuelle et le numéro de classe se trouve
dans le fichier `meta.csv`.

Votre mission est de créer un classificateur qui reconnaît la provenance d'une
communication. Plus précisément, vous devez implémenter un classificateur dans
le fichier `classifier.py`. On vous fournit un script `accuracy_tester.py` pour
facilement vérifier la performance de votre classificateur sur un dataset en
format csv.


> **NOTE** Le constructeur de la classe Classifier doit s'occuper de charger les
> paramètres **pré-entraînés** de votre modèle. La structure de votre code pour
> l'entraînement est à votre discrétion.

> **NOTE** Bien sûr, nous testerons la performance de votre classificateur sur
> des données qui ne sont pas à votre disposition. Ces données seront exactement
> dans le même format que les données fournies.


## Librairies et logiciels disponibles

- Numpy
- Jupyter
- Matplotlib
- Scipy
- PyTorch (si vous êtes ce genre de personne...)

Les librairies de machine learning telles que SciKitLearn ou StanfordNLP sont interdites. Le
but de la compétition est d'implémenter manuellement un algorithme de
classification de texte.

##  Questions et pistes de solution
Pour ceux qui ont compris les indices dans la description de la compétition,
vous devriez résoudre le problème aisément. Pour les autres, voici quelques
pistes de solution.

### Bag-of-words
Une approche simple est d'imaginer que la séquence des mots des communications
n'est pas importante; uniquement la distributions des mots eux-mêmes est
importante. On peut utiliser la fréquence d'apparition des mots dans les classes
pour entraîner un modèles de classification Naïve Bayes. Voici quelques formules
vous permettant d'implémenter un algorithme basique:

##### Objectif
L'objectif est de trouver la classe qui a plus haute probabilité selon les mots
`t_i` de la séquence. Formellement,

![argmax](./img/argmax.png)

Pour une plus grande stabilité numérique, il est généralement préférable
d'utiliser le logarithme des probabilités.

![argmax log](./img/argmax_log.png)


##### A priori sur la classe

![prior](./img/prior.png)


##### Probabilité conditionnelle d'un mot

![word prob](./img/word_prob.png)


##### Lissage de Laplace

Pour éviter de multiplier par des probabilité qui valent 0, il est suggéré de
lisser les distributions.

![laplace](./img/laplace.png)


### Modèle n-gram
À la place de regarder les mots en isolation, il est possible de séparer la
communication en bi-gramme. Par exemple, la séquence

Je suis un pingouin venu de l'espace.


devient


(Je suis) (suis un) (un pingouin) (pingouin venu) (venu de) (de l'espace.)


Ces bigrammes peuvent ensuite être utilisés dans le modèle décrit précédemment.


### Autres idées et questions

- Séparez le jeu de données pour avoir un set de validation.
- Ajoutez des règles manuelles avec des Regex.
- Comment gérer les mots inconnus lors de l'évaluation?
- Devez-vous faire un prétraitement sur les mots?
- Devez-vous considérez tous les mots de le modèle ou devez-vous en filtrer?


## Remise

### Instructions
Dans votre repo, décrivez brièvement le fonctionnement de votre algo dans un README.md.

### Correction automatisée
**Il ne faut pas changer la structure du projet**. Un script automatisé
s'occuper de la correction.

