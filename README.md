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
Pour réaliser cette expérience, nous utilisons le dataset FER-2013, composé d'images de visages en niveaux de gris de taille 48 × 48 pixels. Chaque image appartient à l'une des sept catégories d'émotions.

Les images sont chargées avec `image_dataset_from_directory`, puis regroupées en lots de 32 images. Les valeurs des pixels sont normalisées entre 0 et 1 afin de faciliter l'apprentissage des modèles.

Le dossier d'entraînement est divisé en deux parties : 80 % des images sont utilisées pour entraîner les modèles et 20 % pour vérifier leur capacité à reconnaître des images qu'ils n'ont pas vues pendant l'entraînement. Le dossier de test est conservé séparément et utilisé uniquement pour l'évaluation finale.

Pour rendre les modèles moins dépendants des images exactes du jeu d'entraînement, nous appliquons également des transformations aléatoires : retournement horizontal, légère rotation et zoom. Ces transformations augmentent la diversité des exemples présentés aux modèles.

Le dataset FER-2013 contient davantage d'images pour certaines émotions que pour d'autres. Pour limiter cet effet, nous utilisons une pondération des classes. Les émotions les moins représentées reçoivent un poids plus important pendant l'entraînement. La pondération est adoucie à l'aide de la racine carrée de l'inverse de la fréquence, afin d'éviter d'attribuer des poids excessifs aux classes très rares.

L'objectif est ensuite de comparer les quatre architectures dans un cadre expérimental aussi proche que possible. Les modèles sont entraînés séparément avec l'optimiseur Adam, un taux d'apprentissage initial de 0,0005 et la fonction de perte `sparse_categorical_crossentropy`. L'entraînement est limité à 10 époques au maximum.

Deux mécanismes permettent d'adapter l'entraînement : `EarlyStopping` arrête celui-ci lorsque les performances de validation ne s'améliorent plus, tandis que `ReduceLROnPlateau` réduit le taux d'apprentissage lorsque la perte de validation cesse de diminuer.

### 4.2. Choix et construction des architectures
L'expérience repose sur quatre modèles. Leur comparaison permet d'étudier progressivement différentes façons d'extraire et d'utiliser les informations contenues dans les images faciales.

#### 4.2.1. CNN classique : modèle de référence
Un réseau de neurones convolutionnel, ou CNN (*Convolutional Neural Network*), est particulièrement adapté au traitement des images. Il apprend à reconnaître des caractéristiques visuelles grâce à des filtres appliqués sur différentes régions de l'image.

Dans le cas des expressions faciales, ces caractéristiques peuvent correspondre aux contours des yeux, à la forme de la bouche, aux sourcils ou encore aux plis du visage. En combinant ces informations, le réseau apprend à distinguer les différentes émotions.

Nous avons choisi le CNN classique comme premier modèle parce qu'il constitue une base de comparaison. Il permet d'évaluer les performances obtenues avec une extraction conventionnelle des caractéristiques, avant d'introduire des mécanismes d'attention ou des Transformers.

Son architecture comprend plusieurs couches de convolution, de normalisation, d'activation et de réduction de dimension. Les caractéristiques extraites sont ensuite regroupées par *Global Average Pooling*, puis transmises à des couches de classification qui prédisent l'une des sept émotions.

#### 4.2.2. CNN avec attention spatiale
Le deuxième modèle reprend l'architecture du CNN de référence et lui ajoute un mécanisme d'attention spatiale.

L'idée est simple : toutes les zones d'une image ne sont pas nécessairement aussi utiles pour reconnaître une émotion. Certaines caractéristiques peuvent être plus importantes que d'autres pour la classification.

Le mécanisme d'attention apprend donc à attribuer un poids à différentes positions dans les cartes de caractéristiques produites par le CNN. Pour cela, une convolution 1 × 1 suivie d'une fonction sigmoïde génère une carte d'attention. Cette carte est ensuite multipliée par les caractéristiques extraites.

Une connexion résiduelle permet enfin d'ajouter les caractéristiques pondérées aux caractéristiques initiales. Le réseau conserve ainsi les informations d'origine tout en renforçant celles mises en évidence par l'attention.

Ce deuxième modèle permet de répondre à une première question : **le fait d'accorder davantage d'importance à certaines régions des caractéristiques améliore-t-il la reconnaissance des émotions par rapport à un CNN classique ?**

#### 4.2.3. Vision Transformer simplifié (Tiny ViT)
Le troisième modèle utilise une approche différente : le Vision Transformer, ou ViT.

Alors que le CNN extrait progressivement des caractéristiques à l'aide de filtres convolutionnels, le Vision Transformer divise directement l'image en petites régions appelées *patches*. Il apprend ensuite les relations entre ces différentes régions grâce au mécanisme de *Self-Attention*.

Dans notre expérience, chaque image de 48 × 48 pixels est divisée en 64 patches de 6 × 6 pixels. Chaque patch est transformé en un vecteur de dimension 64. Un encodage positionnel est ajouté pour indiquer au modèle la position de chaque patch dans l'image.

La séquence obtenue est ensuite traitée par deux blocs Transformer. Chaque bloc utilise quatre têtes d'attention, des connexions résiduelles, une normalisation et un réseau de neurones intermédiaire (*MLP*) de dimension 128.

Enfin, les représentations des patches sont regroupées par *Global Average Pooling* avant la classification.

Nous avons choisi une version simplifiée du Vision Transformer afin de pouvoir l'entraîner directement sur FER-2013 avec des ressources informatiques limitées.

Ce troisième modèle permet de répondre à une deuxième question : **un Transformer qui apprend les relations entre les différentes régions du visage peut-il obtenir de meilleurs résultats qu'un CNN ?**

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
   jupyter notebook facial_emotion_recognition-quatre_modele.ipynb
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
#### Résultats expérimentaux

Les résultats obtenus sur le jeu de test sont les suivants :

| Modèle | Accuracy | Macro F1 | Weighted F1 |
|--------|---------:|---------:|------------:|
| CNN Baseline | 0,485 | 0,389 | 0,460 |
| **CNN Attention** | **0,516** | **0,426** | **0,485** |
| Tiny ViT | 0,360 | 0,273 | 0,328 |
| CNN-APP-Transformer | 0,451 | 0,371 | 0,437 |

**F1-score par émotion :**

| Émotion | CNN Baseline | CNN Attention | Tiny ViT | CNN-APP-Transformer |
|---------|-------------:|--------------:|---------:|--------------------:|
| angry    | 0,35 | **0,42** | 0,23 | 0,34 |
| disgust  | 0,05 | **0,19** | 0,00 | 0,03 |
| fear     | 0,17 | 0,10 | 0,11 | **0,20** |
| happy    | 0,72 | **0,75** | 0,52 | 0,67 |
| neutral  | 0,47 | **0,51** | 0,28 | 0,36 |
| sad      | 0,31 | **0,37** | 0,29 | 0,35 |
| surprise | **0,65** | **0,65** | 0,49 | **0,65** |


Le CNN-APP-Transformer obtient les meilleurs résultats sur les trois métriques globales dans cette expérience.
Le CNN avec attention spatiale n'améliore pas les performances globales par rapport au CNN de base.
Le Tiny ViT présente les résultats les plus faibles des quatre modèles dans cette configuration expérimentale.
Ces résultats suggèrent que la combinaison d'une extraction convolutionnelle locale et d'une sélection attentive des patches peut être intéressante pour cette tâche.

Ils ne démontrent cependant pas que cette approche est systématiquement supérieure aux autres architectures.

### Analyse
- **Le CNN avec attention spatiale est le meilleur modèle** sur toutes les métriques globales (+3 points d'accuracy et +4 points de macro-F1 face au CNN de base). Cette amélioration est modeste et provient d'une seule exécution par modèle : elle doit être interprétée comme une tendance, pas comme une différence statistiquement établie.
- **Le Tiny ViT est nettement le moins performant.** Ce résultat est cohérent avec la littérature : les Transformers manquent de biais inductif (localité, invariance par translation) et demandent beaucoup de données ou un pré-entraînement. Avec ~23 000 images de 48 × 48, 10 époques et aucun pré-entraînement, il apprend peu.
- **L'ajout de la convolution avant le Transformer aide** : le CNN-APP-Transformer gagne environ 9 points d'accuracy par rapport au Tiny ViT, ce qui confirme l'intérêt d'un backbone convolutif pour extraire des caractéristiques locales. Il reste toutefois en dessous des deux CNN, ce qui peut s'expliquer par notre version simplifiée de la sélection de patchs et par un entraînement très court.
- **Les classes rares et ambiguës restent difficiles pour tous les modèles.** `disgust` (111 images de test) est quasiment jamais reconnue malgré la pondération des classes, et `fear` obtient un rappel très faible (5 à 15 %) : elle est souvent confondue avec `sad`, `angry` ou `neutral`. À l'inverse, `happy` et `surprise`, visuellement distinctives, atteignent les meilleurs scores.
- **Biais vers `neutral` pour les CNN** : la classe `neutral` a un rappel élevé (≈ 0,70) mais une précision faible (≈ 0,35 à 0,40), ce qui indique que les modèles la prédisent par défaut en cas d'incertitude.
- Le **macro-F1** est nettement plus bas que l'accuracy pour tous les modèles : l'accuracy seule masque l'échec sur les classes minoritaires.
### Limites
Plusieurs limites doivent être prises en compte :

- Les images FER-2013 ont une résolution relativement faible.
- Le dataset présente un déséquilibre entre les classes.
- Certaines expressions sont visuellement proches.
- Les modèles sont entraînés depuis zéro, sans pré-entraînement externe.
- Les ressources matérielles sont limitées et l'entraînement est effectué localement sans GPU.
- Le Tiny ViT est volontairement réduit et entraîné pendant un nombre limité d'époques.
- Le quatrième modèle est une adaptation simplifiée d'APP et ne reproduit pas l'ensemble de l'architecture APViT.
- La sélection des patches dans notre APP ne conserve pas explicitement leurs coordonnées spatiales d'origine dans l'encodage positionnel.
- Les résultats proviennent d'une seule configuration expérimentale et ne constituent pas une comparaison statistique sur plusieurs répétitions.

### Améliorations
- Augmenter le nombre d'époques et ajuster les hyperparamètres ;
- remplacer le critère d'importance des patchs par un score d'attention appris, comme dans l'article de Xue et al. ;
- utiliser un pré-entraînement ou des architectures plus profondes ;
