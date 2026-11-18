# Projet-final-M2
Etude comparative de quatre architectures simplifiées : Facial Emotion Recognition
# Reconnaissance des émotions faciales
Projet final du cours **Réseaux de neurones : vision par ordinateur** .
## 1. Présentation du projet
Ce projet s'inscrit dans le thème **« Reconnaissance des émotions »** proposé dans le sujet. Il s'appuie sur deux articles de référence :
- **Modi et al.** : reconnaissance des émotions faciales avec des réseaux convolutifs (CNN) ;
- **Xue et al.** : Vision Transformer avec *pooling attentionnel* pour la reconnaissance des expressions faciales.

Les quatre modèles étudiés sont :
1. Un réseau convolutionnel classique (CNN Baseline).
2. Un CNN intégrant un mécanisme d'attention spatiale (CNN Attention).
3. Un Vision Transformer simplifié (Tiny ViT).
4. Un modèle hybride CNN–APP–Transformer, inspiré du principe d'Attentive Patch Pooling proposé par Xue et al.

On a construit ces modèles simplifiés pour comprendre leur fonctionnement et comparer leurs apports soumis aux mêmes conditions (mêmes données, même prétraitement; même protocole d'entraînement.
On ne cherche pas à égaler l'état de l'art des articles mais à en comprendre les principes généraux.

## 2. Objectifs
- Construire un modèle CNN de référence.
- Étudier l'effet d'une attention spatiale intégrée à un CNN.
- Évaluer un Vision Transformer simplifié entraîné directement sur FER-2013.
- Étudier une architecture hybride utilisant une sélection attentive des patches avant leur traitement par un Transformer.
- Comparer les quatre modèles à l'aide de métriques de classification.
- Analyser les avantages et les limites de chaque approche.
  
## 3. Données
Le projet utilise le dataset FER-2013, disponible sur Kaggle :
https://www.kaggle.com/datasets/msambare/fer2013

Le dataset contient des images faciales en niveaux de gris, de résolution 48 × 48 pixels, réparties en sept catégories : 'angry', 'disgust', 'fear', 'happy'; 'neutral', 'sad', 'suprise'.

Le dataset présente un déséquilibre important entre les catégories. Par exemple, la classe Disgust est moins représentée que Happy.
Pour limiter l'influence de ce déséquilibre, des poids de classes sont calculés à partir des effectifs de l'ensemble d'entraînement.
## 4. Méthode
### 4.1. Préparation des données et protocole expérimental
Pour réaliser cette expérience, nous utilisons le dataset FER-2013. Chaque image appartient à l'une des sept catégories d'émotions.

Les images sont chargées avec `image_dataset_from_directory`, puis regroupées en lots de 32 images. Les valeurs des pixels sont normalisées entre 0 et 1 afin de faciliter l'apprentissage des modèles.

Le dossier d'entraînement est divisé en deux parties : 80 % des images sont utilisées pour entraîner les modèles et 20 % pour vérifier leur capacité à reconnaître des images qu'ils n'ont pas vues pendant l'entraînement. Le dossier de test est conservé séparément et utilisé uniquement pour l'évaluation finale.

Pour rendre les modèles moins dépendants des images exactes du jeu d'entraînement, nous appliquons également des transformations aléatoires : retournement horizontal, légère rotation et zoom. Ces transformations augmentent la diversité des exemples présentés aux modèles.

Le dataset FER-2013 contient davantage d'images pour certaines émotions que pour d'autres. Pour limiter cet effet, nous utilisons une pondération des classes. Les émotions les moins représentées reçoivent un poids plus important pendant l'entraînement. 

L'objectif est de comparer les quatre architectures dans un cadre expérimental aussi proche que possible.
Les modèles sont entraînés séparément avec l'optimiseur Adam, un taux d'apprentissage initial de 0,0005 et la fonction de perte `sparse_categorical_crossentropy`. L'entraînement est limité à 10 époques au maximum.

On utilise pour adapter l'entraînement : `EarlyStopping` arrête celui-ci lorsque les performances de validation ne s'améliorent plus, tandis que `ReduceLROnPlateau` réduit le taux d'apprentissage lorsque la perte de validation cesse de diminuer.

### 4.2. Choix et construction des architectures
L'expérience repose sur quatre modèles. Leur comparaison permet d'étudier progressivement différentes façons d'extraire et d'utiliser les informations contenues dans les images faciales.

#### 4.2.1. CNN classique : modèle de référence
Un réseau de neurones convolutionnel, ou CNN (*Convolutional Neural Network*), est particulièrement adapté au traitement des images. Il apprend à reconnaître des caractéristiques visuelles grâce à des filtres appliqués sur différentes régions de l'image.

Pour les expressions faciales, ces caractéristiques peuvent être les contours des yeux,  la forme de la bouche, les sourcils ou les plis du visage. En combinant ces informations, le réseau apprend à distinguer les différentes émotions.

Nous avons choisi le CNN classique comme premier modèle parce qu'il constitue une base de comparaison. Il permet d'évaluer les performances obtenues avec une extraction conventionnelle des caractéristiques, avant d'introduire des mécanismes d'attention ou des Transformers.

Son architecture comprend plusieurs couches de convolution, de normalisation, d'activation et de réduction de dimension. Les caractéristiques extraites sont ensuite regroupées par Global Average Pooling, puis transmises à des couches de classification qui prédisent l'une des sept émotions.

#### 4.2.2. CNN avec attention spatiale
Le deuxième modèle reprend l'architecture du CNN de référence et lui ajoute un mécanisme d'attention spatiale.

L'idée est simple : toutes les zones d'une image ne sont pas nécessairement aussi utiles pour reconnaître une émotion. Certaines caractéristiques peuvent être plus importantes que d'autres pour la classification.

Le mécanisme d'attention apprend donc à attribuer un poids à différentes positions dans les cartes de caractéristiques produites par le CNN. Pour cela, une convolution 1 × 1 suivie d'une fonction sigmoïde génère une carte d'attention. Cette carte est ensuite multipliée par les caractéristiques extraites.

Une connexion résiduelle permet enfin d'ajouter les caractéristiques pondérées aux caractéristiques initiales. Le réseau conserve ainsi les informations d'origine tout en renforçant celles mises en évidence par l'attention.


#### 4.2.3. Vision Transformer simplifié (Tiny ViT)
Le troisième modèle utilise une approche différente : le Vision Transformer, ou ViT.

Alors que le CNN extrait progressivement des caractéristiques à l'aide de filtres convolutionnels, le Vision Transformer divise directement l'image en petites régions appelées *patches*. Il apprend ensuite les relations entre ces différentes régions grâce au mécanisme de *Self-Attention*.

Dans notre expérience, chaque image de 48 × 48 pixels est divisée en 64 patches de 6 × 6 pixels. Chaque patch est transformé en un vecteur de dimension 64. Un encodage positionnel est ajouté pour indiquer au modèle la position de chaque patch dans l'image.

La séquence obtenue est ensuite traitée par deux blocs Transformer. Chaque bloc utilise quatre têtes d'attention, des connexions résiduelles, une normalisation et un réseau de neurones intermédiaire (*MLP*) de dimension 128.

Enfin, les représentations des patches sont regroupées par *Global Average Pooling* avant la classification.

Nous avons choisi une version simplifiée du Vision Transformer afin de pouvoir l'entraîner directement sur FER-2013 avec des ressources informatiques limitées.

#### 4.2.4. Modèle hybride CNN–APP–Transformer

Le quatrième modèle est inspiré des travaux de Xue et al., intitulés *Vision Transformer with Attentive Pooling for Robust Facial Expression Recognition*.

Dans cet article, les auteurs proposent APViT, une architecture qui combine l'extraction de caractéristiques par CNN avec des mécanismes de sélection attentive avant et pendant le traitement par Transformer.

L'idée principale est que toutes les informations extraites d'un visage ne sont pas forcément utiles à la reconnaissance d'une émotion. Certaines régions peuvent être moins discriminantes ou contenir des informations perturbatrices. Il peut donc être intéressant de sélectionner les caractéristiques les plus pertinentes avant de les transmettre au Transformer.

L'article présente deux mécanismes :
- **Attentive Patch Pooling (APP)** : sélection des patches les plus informatifs à partir des caractéristiques extraites par le CNN.
- **Attentive Token Pooling (ATP)** : sélection des tokens les plus utiles au cours du traitement par Transformer.

Dans notre projet, nous reprenons le principe d'APP dans une version simplifiée, adaptée à FER-2013 et aux ressources matérielles disponibles.

Le fonctionnement du modèle se déroule en plusieurs étapes :

1. **Extraction des caractéristiques :** un petit CNN analyse l'image et produit une carte de caractéristiques de dimension 6 × 6 × 64.
2. **Création des patches :** cette carte est transformée en une séquence de 36 patches, chacun représenté par un vecteur de 64 valeurs.
3. **Évaluation de l'importance :** pour chaque patch, un score est calculé en additionnant les valeurs absolues de ses caractéristiques. Ce score sert à estimer son importance.
4. **Sélection des patches :** les 27 patches ayant les scores les plus élevés sont conservés. Cela représente 75 % des patches initiaux.
5. **Ajout d'un token de classification :** un token spécial, appelé `[CLS]`, est ajouté à la séquence. Il servira à construire la représentation utilisée pour prédire l'émotion.
6. **Encodage positionnel :** des informations de position sont ajoutées aux tokens sélectionnés.
7. **Traitement par Transformer :** la séquence est analysée par deux blocs Transformer utilisant quatre têtes d'attention.
8. **Classification :** la représentation du token `[CLS]` est transmise aux couches finales pour prédire l'une des sept émotions.

Cette architecture cherche donc à combiner deux avantages : la capacité du CNN à extraire des caractéristiques locales du visage et la capacité du Transformer à apprendre les relations entre les caractéristiques sélectionnées.

**Différence avec l'article original :** notre modèle ne constitue pas une reproduction complète d'APViT. Nous utilisons un CNN léger entraîné depuis zéro, nous implémentons le principe APP, mais nous n'ajoutons pas le module ATP. Le Transformer est également simplifié. Il s'agit donc d'une adaptation expérimentale destinée à étudier l'intérêt de la sélection attentive des patches, et non à reproduire les performances rapportées par Xue et al.

### 4.3. Logique de l'expérience et comparaison des modèles

Les quatre modèles ont été choisis pour étudier progressivement différentes stratégies de reconnaissance des expressions faciales :

- Le CNN classique sert de référence pour mesurer les performances d'une architecture convolutionnelle.
- Le CNN avec attention permet de vérifier si une pondération spatiale des caractéristiques améliore cette référence.
- Le Tiny ViT permet d'étudier une approche fondée directement sur les relations entre les patches de l'image.
- Le modèle CNN–APP–Transformer permet d'évaluer l'intérêt de sélectionner les patches les plus informatifs avant leur traitement par un Transformer.

Cette comparaison permet ainsi de mieux comprendre si les mécanismes d'attention et la sélection des caractéristiques apportent un avantage dans la reconnaissance des expressions faciales.

Les modèles sont entraînés séparément avec les mêmes données d'entraînement et de validation. Ils sont ensuite évalués sur le même jeu de test.

### 4.4. Critères d'évaluation

Les performances sont comparées à l'aide de trois métriques : l'accuracy, le macro F1-score et le weighted F1-score.

**Accuracy**

L'accuracy représente la proportion d'images correctement classées parmi toutes les images du jeu de test. Elle donne une indication générale de la capacité du modèle à reconnaître les émotions.

**Macro F1-score**

Le macro F1-score correspond à la moyenne des F1-scores calculés séparément pour les sept émotions, sans tenir compte de leur nombre d'images. Chaque émotion possède donc la même importance dans le calcul.

Cette métrique est particulièrement utile pour FER-2013, car le dataset présente un déséquilibre entre les classes. Elle permet de vérifier si un modèle reconnaît correctement les différentes émotions et pas seulement les catégories les plus représentées.

**Weighted F1-score**

Le weighted F1-score correspond également à une moyenne des F1-scores par émotion, mais chaque classe est pondérée selon son nombre d'images dans le jeu de test.

Il permet donc d'évaluer les performances globales en tenant compte de la distribution des classes.

L'utilisation de ces trois métriques permet d'obtenir une comparaison plus complète : l'accuracy mesure la proportion globale de bonnes prédictions, tandis que les deux F1-scores apportent des informations complémentaires sur la qualité de la classification des différentes émotions.

## 5. Installation
Prérequis : **Python 3.10 à 3.13**  et `pip`.

```bash
git clone https://github.com/<utilisateur>/<depot>.git
cd <depot>

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

Un GPU n'est pas obligatoire 

## 6. Exécution
1. Télécharger FER-2013 (version avec dossiers `train/` et `test/`) depuis Kaggle et placer l'archive à la racine du dépôt .
2. Ouvrir le notebook :
   ```bash
   jupyter notebook Facial_emotion_recognition_modele.ipynb
   ```
3. Dans la 2ᵉ cellule, adapter la variable `chemin_zip` au nom réel de l'archive (par défaut `archive(1).zip`).
4. Exécuter les cellules dans l'ordre (*Run All*). Le notebook :
   - extrait l'archive dans `images_extraites/` et explore les données ;
   - construit les jeux `train` / `val` / `test` et calcule les poids de classes ;
   - définit et entraîne les quatre modèles ;
   - évalue chacun sur le jeu de test et trace les comparaisons ;
   - sauvegarde les modèles : `cnn_baseline.keras`, `cnn_attention.keras`, `tiny_vit.keras`, `app_vit.keras`.

Il est également possible d'exécuter le notebook sur **Google Colab**.

## 7. Résultats

### 7.1. Résultats expérimentaux

Après l'entraînement des quatre modèles, nous avons évalué leurs performances sur le dataset FER-2013. Les résultats obtenus sont présentés dans le tableau suivant.

| Modèle | Accuracy | Macro F1 | Weighted F1 |
|--------|---------:|---------:|------------:|
| CNN Baseline | 0,432 | 0,335 | 0,402 |
| CNN Attention | 0,457 | 0,345 | 0,412 |
| Tiny ViT | 0,379 | 0,294 | 0,353 |
| **CNN-APP-Transformer** | **0,483** | **0,392** | **0,452** |

Le CNN-APP-Transformer obtient les meilleurs résultats sur les trois métriques globales, avec une accuracy de 48,3 % et un Macro F1 de 0,392. Il est suivi par le CNN avec attention spatiale, puis par le CNN de base.

Le Tiny ViT obtient les résultats les plus faibles. Il atteint une accuracy de 37,9 %, ce qui montre qu'il rencontre davantage de difficultés à reconnaître les expressions faciales dans notre configuration expérimentale.

Ces résultats montrent que les performances varient selon l'architecture utilisée. En particulier, le modèle hybride qui combine un CNN et un Transformer obtient les meilleurs résultats parmi les quatre modèles testés.

Nous avons également comparé les résultats pour chaque émotion afin de mieux comprendre les points forts et les difficultés des modèles.

**F1-score pour chaque émotion :**

| Émotion | CNN Attention | CNN Baseline | CNN-APP-Transformer | Tiny ViT |
|---------|--------------:|-------------:|--------------------:|---------:|
| angry | 0,285 | 0,290 | **0,342** | 0,278 |
| disgust | 0,018 | 0,000 | **0,103** | 0,000 |
| fear | 0,117 | 0,166 | **0,186** | 0,119 |
| happy | 0,650 | 0,668 | **0,679** | 0,557 |
| neutral | 0,459 | 0,436 | **0,462** | 0,310 |
| sad | 0,248 | 0,159 | **0,329** | 0,271 |
| surprise | 0,641 | 0,624 | **0,642** | 0,525 |

Le CNN-APP-Transformer obtient le meilleur F1-score pour les sept émotions. Son avantage est particulièrement visible pour 'disgust' et 'sad', deux classes qui sont difficiles à reconnaître pour les autres modèles.

L'émotion 'happy' obtient de bons résultats avec tous les modèles, tandis que *surprise* est également assez bien reconnue. À l'inverse, 'disgust' reste la classe la plus difficile, même si le CNN-APP-Transformer obtient un meilleur score que les autres modèles.

### 7.2. Analyse des résultats

La comparaison des quatre modèles permet de faire plusieurs observations.

**Le CNN-APP-Transformer obtient les meilleurs résultats globaux.**

Avec une accuracy de 48,3 %, un Macro F1 de 0,392 et un Weighted F1 de 0,452, il arrive en première position sur les trois métriques.

Il obtient également le meilleur F1-score pour chacune des sept émotions. Cette observation est intéressante, car elle montre que son avantage ne concerne pas seulement les classes les plus représentées dans le jeu de données.

Ces résultats suggèrent que l'association d'un réseau convolutionnel et d'une sélection attentive des patches peut être utile pour la reconnaissance des expressions faciales. Le CNN extrait d'abord des caractéristiques locales du visage, puis le mécanisme APP sélectionne les patches considérés comme les plus importants avant leur traitement par le Transformer.

Cependant, ces résultats ne prouvent pas que cette architecture est toujours supérieure aux autres. Ils montrent seulement qu'elle est la plus performante parmi les modèles et dans les conditions de notre expérience.

**Le CNN avec attention spatiale améliore légèrement le CNN de base.**

Le CNN avec attention spatiale atteint une accuracy de 45,7 %, contre 43,2 % pour le CNN de base. Son Macro F1 passe de 0,335 à 0,345.

Cette amélioration reste limitée, mais elle suggère que l'attention spatiale peut aider le modèle à mieux exploiter certaines zones du visage.

Cependant, les résultats par émotion montrent que cette amélioration n'est pas présente pour toutes les classes. Par exemple, le CNN de base obtient un meilleur score pour angry, fear et happy. L'attention spatiale apporte donc un gain global modeste, mais son effet varie selon l'émotion à reconnaître.

**Le Tiny ViT obtient les résultats les plus faibles.**

Le Tiny ViT atteint une accuracy de 37,9 % et un Macro F1 de 0,294. Il est moins performant que les trois autres modèles dans notre expérience.
Le modèle est entraîné à partir de zéro, sans pré-entraînement, sur des images de petite taille (48 × 48 pixels) et avec un nombre limité d'epoch.

Contrairement aux CNN, les Transformers ne disposent pas naturellement des mêmes mécanismes pour détecter les caractéristiques locales d'une image. Ils peuvent avoir besoin de davantage de données ou d'un pré-entraînement pour apprendre des représentations efficaces.

Cela ne signifie pas que les Transformers sont moins performants que les CNN en général. Cela indique plutôt que celui de notre expérience n'a pas obtenu des résultats aussi satisfaisants.

**La combinaison d'un CNN et d'un Transformer semble intéressante.**

Le CNN-APP-Transformer obtient une accuracy de 48,3 %, contre 37,9 % pour le Tiny ViT. Cette différence montre que l'ajout d'une étape d'extraction de caractéristiques par un CNN peut être bénéfique avant le traitement des patches par un Transformer.

Le CNN aide à extraire les informations locales du visage. Ensuite, le mécanisme APP sélectionne une partie des patches afin de privilégier ceux qui semblent les plus importants pour la reconnaissance.

Dans notre expérience, cette combinaison donne de meilleurs résultats que l'utilisation du Tiny ViT seul. Il faut toutefois rappeler que notre modèle est une adaptation simplifiée du mécanisme APP présenté par Xue et al. Il ne reproduit pas toute l'architecture APViT.

**Certaines émotions restent difficiles à reconnaître.**

Les résultats par émotion montrent que 'happy' et 'surprise' sont relativement bien reconnues par les quatre modèles. En revanche, 'disgust' reste particulièrement difficile.

Le CNN-APP-Transformer améliore le F1-score de 'disgust' à 0,103, alors que les autres modèles obtiennent des scores très faibles, voire nuls. Malgré cette amélioration, le résultat reste faible.

Cette difficulté peut notamment être liée au faible nombre d'images disponibles pour cette émotion. Le déséquilibre des classes peut empêcher les modèles d'apprendre suffisamment de caractéristiques pour reconnaître correctement les émotions les moins représentées.

La classe "fear" reste également difficile, avec des F1-scores compris entre 0,117 et 0,186. Certaines expressions faciales peuvent être visuellement proches, ce qui rend leur distinction plus compliquée.

Enfin, le Macro F1 est inférieur à l'accuracy pour tous les modèles. Cela montre que les performances sont moins bonnes lorsque chaque émotion reçoit la même importance dans le calcul du score. L'accuracy seule ne suffit donc pas pour évaluer correctement les modèles sur un dataset déséquilibré.

### 7.3. Limites de l'expérience

Plusieurs limites doivent être prises en compte pour interpréter les résultats.

- **La résolution des images :** les images FER-2013 ont une taille de 48 × 48 pixels. Certains détails du visage peuvent donc être difficiles à distinguer.
- **Le déséquilibre des classes :** certaines émotions, notamment 'disgust', sont beaucoup moins représentées que d'autres.
- **La proximité entre les expressions :** certaines émotions présentent des caractéristiques faciales similaires, ce qui peut provoquer des confusions.
- **L'absence de pré-entraînement :** les quatre modèles sont entraînés à partir de zéro, sans utiliser de poids appris sur un autre jeu de données.
- **Les ressources matérielles :** les expériences sont réalisées localement, sans GPU. Cela limite le temps et les possibilités d'entraînement.
- **La durée d'entraînement :** le nombre d'epoch reste limité. Les modèles n'ont donc peut-être pas atteint leur meilleur niveau de performance.
- **La simplification du CNN-APP-Transformer :** notre quatrième modèle reprend le principe de sélection attentive des patches, mais ne reproduit pas tous les éléments de l'architecture APViT proposée par Xue et al.

### 7.4. Améliorations possibles

Plusieurs pistes pourraient être étudiées pour améliorer les résultats.

- **Augmenter le nombre d'epoch :** entraîner les modèles plus longtemps pourrait leur permettre d'apprendre des caractéristiques plus précises, tout en surveillant le risque de surapprentissage.
- **Ajuster les hyperparamètres :** tester différents taux d'apprentissage, tailles de batch et paramètres d'architecture pourrait améliorer les performances.
- **Utiliser un modèle pré-entraîné :** partir d'un réseau déjà entraîné sur un grand jeu d'images pourrait faciliter l'apprentissage, notamment pour le Transformer.
- **Répéter les expériences :** entraîner chaque modèle plusieurs fois avec différentes initialisations permettrait de vérifier si les différences observées sont stables.
- **Mieux traiter les classes minoritaires :** tester d'autres méthodes de pondération ou d'augmentation des données pourrait améliorer la reconnaissance des émotions les moins représentées.

### 7.5. Bilan des résultats

Dans cette expérience, le CNN-APP-Transformer obtient les meilleures performances globales et les meilleurs F1-scores pour les sept émotions. Ces résultats suggèrent que l'association d'une extraction de caractéristiques par CNN et d'une sélection attentive des patches avant le Transformer est une piste intéressante pour la reconnaissance des expressions faciales.

Le CNN avec attention spatiale améliore légèrement les résultats du CNN de base, mais cet avantage reste limité et varie selon les émotions. Le Tiny ViT obtient les résultats les plus faibles dans notre configuration expérimentale.

La principale conclusion est donc que **le CNN-APP-Transformer est le modèle le plus performant parmi les quatre architectures testées dans cette expérience**. Toutefois, des entraînements plus longs, plusieurs répétitions et une implémentation plus complète du mécanisme APP seraient nécessaires pour confirmer la stabilité de cet avantage.

