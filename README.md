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
### 4.1 Prétraitement et protocole

- Chargement avec `image_dataset_from_directory` (niveaux de gris, 48 × 48, batch de 32).
- Normalisation des pixels dans [0, 1] (`Rescaling(1/255)`).
- **Séparation 80 % / 20 %** du dossier `train/` en entraînement et validation (`seed=42`). Le dossier `test/` n'est utilisé **que pour l'évaluation finale**.
- **Augmentation de données** (dans le modèle) : flip horizontal, rotation (±5 %), zoom (8 %).
- **Pondération des classes** adoucie (racine carrée de l'inverse de la fréquence, puis normalisée) pour compenser le déséquilibre sans poids extrêmes (`disgust` ≈ 2,4 ; `happy` ≈ 0,6).
- Optimiseur Adam (lr = 5·10⁻⁴), perte `sparse_categorical_crossentropy`, **10 epochs maximum**, `EarlyStopping` (patience 3, restauration des meilleurs poids) et `ReduceLROnPlateau` (facteur 0,5, patience 2).

### 4.2 Fondements théoriques et choix des architectures

#### Réseau de neurones convolutionnel (CNN)

Un CNN (*Convolutional Neural Network*) est un réseau de neurones conçu pour traiter des données structurées sous forme de grilles, notamment les images.

Les couches convolutionnelles utilisent des filtres qui apprennent progressivement à détecter des caractéristiques visuelles :

- contours et textures ;
- formes locales ;
- motifs faciaux ;
- caractéristiques plus complexes associées aux expressions.

Le CNN est choisi comme modèle de référence car les convolutions sont adaptées à l'extraction de caractéristiques spatiales locales dans les images.

#### CNN avec attention spatiale

Le mécanisme d'attention permet au réseau de pondérer différemment les caractéristiques extraites.
Dans notre implémentation, une convolution 1 × 1 suivie d'une fonction sigmoïde produit une carte d'attention spatiale.
Cette carte est multipliée par les caractéristiques du CNN :
\[
F_{att}=F \odot A
\]
où :

- \(F\) représente les cartes de caractéristiques ;
- \(A\) représente la carte d'attention ;
- \(\odot\) représente la multiplication élément par élément.

Une connexion résiduelle est ensuite utilisée :
\[
F_{out}=F+F_{att}
\]

L'objectif est de conserver les caractéristiques initiales tout en renforçant celles auxquelles le mécanisme d'attention attribue un poids élevé.

Cette architecture permet d'étudier si une pondération spatiale améliore les performances du CNN classique.

####  Vision Transformer simplifié (Tiny ViT)

Le Vision Transformer applique le mécanisme Transformer aux images.
Contrairement à un CNN classique, il divise l'image en petites régions appelées *patches*.
Dans notre implémentation :
- Taille de l'image : 48 × 48 pixels.
- Taille d'un patch : 6 × 6 pixels.
- Nombre de patches : 64.
- Dimension de projection : 64.
- Nombre de têtes d'attention : 4.
- Nombre de blocs Transformer : 2.
- Dimension du MLP : 128.

Chaque patch est transformé en vecteur numérique. Un encodage positionnel est ajouté afin de conserver l'information sur la position des patches.

Les vecteurs sont ensuite traités par des blocs Transformer composés de :
- Multi-Head Self-Attention ;
- connexions résiduelles ;
- normalisation ;
- réseau MLP.

Une opération de Global Average Pooling est utilisée pour agréger les représentations avant la classification finale.
Le Tiny ViT est choisi pour comparer une approche Transformer directe avec les architectures convolutionnelles.

### 5.4 Modèle hybride CNN–APP–Transformer
Cette architecture est inspirée de l'article de Xue et al. :
*Vision Transformer with Attentive Pooling for Robust Facial Expression Recognition.*
L'article présente APViT, une architecture qui utilise deux mécanismes de regroupement attentif :
- Attentive Patch Pooling (APP) ;
- Attentive Token Pooling (ATP).

L'objectif est de sélectionner les caractéristiques les plus discriminantes et de réduire l'influence des informations moins pertinentes, notamment les zones bruitées ou occluses.

#### Adaptation réalisée dans ce projet
Notre quatrième modèle reprend le principe d'Attentive Patch Pooling, mais utilise une architecture simplifiée adaptée à FER-2013 et aux ressources matérielles disponibles.

Son fonctionnement est le suivant :

**Étape 1 — Extraction des caractéristiques par CNN**
Un petit backbone CNN transforme l'image en une carte de caractéristiques de dimension 6 × 6 × 64.
**Étape 2 — Transformation en patches**
La carte de caractéristiques est réorganisée en une séquence de 36 patches, chacun représenté par un vecteur de dimension 64.
**Étape 3 — Calcul du score d'importance**
Pour chaque patch, le modèle calcule un score à partir de la somme des valeurs absolues de ses activations sur les canaux :
\[
s_i=\sum_c |f_{i,c}|
\]

**Étape 4 — Sélection attentive**
Les 27 patches ayant les scores les plus élevés sont conservés parmi les 36 patches disponibles.
Cette sélection correspond à une proportion de conservation de 75 %.
**Étape 5 — Ajout du token de classification**
Un token [CLS] entraînable est ajouté à la séquence afin de produire une représentation globale pour la classification.
**Étape 6 — Encodage positionnel**
Un encodage positionnel est ajouté à la séquence de tokens.
Dans cette implémentation, les positions encodées correspondent aux rangs des patches après leur sélection, et non à leurs coordonnées spatiales originales.
**Étape 7 — Traitement par Transformer**
La séquence est traitée par deux blocs Transformer avec quatre têtes d'attention.
**Étape 8 — Classification**
La représentation du token [CLS] est utilisée par des couches denses pour prédire l'une des sept émotions.
#### Différence avec l'architecture originale de Xue et al.
Le modèle développé dans ce projet n'est pas une reproduction complète d'APViT.
Il s'agit d'une adaptation expérimentale :
- utilisation d'un backbone CNN léger entraîné depuis zéro ;
- implémentation du principe APP ;
- absence du module ATP ;
- architecture Transformer réduite ;
- adaptation à la résolution 48 × 48 et aux sept classes de FER-2013.
Le but est d'étudier l'intérêt de la sélection attentive des patches dans un modèle hybride CNN-Transformer, et non de reproduire les résultats de l'article original.

#### Protocole expérimental

Les quatre modèles sont entraînés sur le même dataset et évalués sur le dataset.
Les paramètres communs comprennent :

- Taille d'entrée : 48 × 48 × 1.
- Batch size : 32.
- Nombre maximal d'époques : 10.
- Optimiseur : Adam.
- Fonction de perte : Sparse Categorical Crossentropy.
- Métrique suivie pendant l'entraînement : Accuracy.
- Pondération des classes pendant l'entraînement.
- Data augmentation.
- Early Stopping.
- ReduceLROnPlateau.

Les modèles utilisent une valeur initiale de learning rate de 0.0005.
Les callbacks sont configurés avec :
- EarlyStopping : patience de 3 époques, restauration des meilleurs poids selon `val_loss`.
- ReduceLROnPlateau : réduction du learning rate par un facteur de 0.5 après 2 époques sans amélioration, avec un minimum de 0.00001.
Les quatre modèles sont entraînés séparément.

#### Métriques d'évaluation

Trois métriques principales sont utilisées.
### Accuracy
L'accuracy représente la proportion de prédictions correctes parmi toutes les images testées.

### Macro F1-score
Le macro F1-score calcule le F1-score de chaque classe puis effectue une moyenne non pondérée.
Cette métrique est particulièrement utile lorsque les classes sont déséquilibrées, car chaque émotion contribue de manière égale au résultat final.

### Weighted F1-score
Le weighted F1-score calcule une moyenne des F1-scores par classe pondérée par le nombre d'images de chaque classe.
Il tient donc compte de la distribution réelle des classes.

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
