#!/usr/bin/env python
# coding: utf-8

# # Data
# - movie_ratings_500_id.pkl contains the interactions between users and movies
# - movie_metadata.pkl contains detailed information about movies, e.g. genres, actors and directors of the movies.
# 
# # Goal
# 
# - Construct your own recommender systems
# - Compare the performances of at least one of the baselines
# 
# 

# # Baselines
# 
# ## User-Based Collaborative Filtering
# This approach predicts $\hat{r}_{(u,i)}$ by leveraging the ratings given to $i$ by $u$'s similar users. Formally, it is written as:
# 
# \begin{equation}
# \hat{r}_{(u,i)} = \frac{\sum\limits_{v \in \mathcal{N}_i(u)}sim_{(u,v)}r_{vi}}{\sum\limits_{v \in \mathbf{N}_i(u)}|sim_{(u,v)}|}
# \end{equation}
# where $sim_{(u,v)}$ is the similarity between user $u$ and $v$. Usually, $sim_{(u,v)}$ can be computed by Pearson Correlation or Cosine Similarity.
# 
# ## Item-Based Collaborative Filtering
# This approach exploits the ratings given to similar items by the target user. The idea is formalized as follows:
# 
# \begin{equation}
# \hat{r}_{(u,i)} = \frac{\sum\limits_{j \in \mathcal{N}_u(i)}sim_{(i,j)}r_{ui}}{\sum\limits_{j \in \mathbf{N}_u(i)}|sim_{(i,j)}|}
# \end{equation}
# where $sim_{(i,j)}$ is the similarity between item $i$ and $j$. Usually, $sim_{(i,j)}$ can be computed by Pearson Correlation or Cosine Similarity.
# 
# ## Vanilla MF (You may use the package Surprise if you do not want to write the training function by your self)
# Vanilla MF is the inner product of vectors that represent users and items. Each user is represented by a vector $\textbf{p}_u \in \mathbb{R}^d$, each item is represented by a vector $\textbf{q}_i \in \mathbb{R}^d$, and $\hat{r}_{(u,i)}$ is computed by the inner product of $\textbf{p}_u $ and $\textbf{q}_i$. The core idea of Vanilla MF is depicted in the followng figure and follows the idea of SVD as we have seen during the TD.
# 
# ![picture](https://drive.google.com/uc?export=view&id=1EAG31Qw9Ti6hB7VqdONUlijWd4rXVobC)
# 
# \begin{equation}
# \hat{r}_{(u,i)} = \textbf{p}_u{\textbf{q}_i}^T
# \end{equation}
# 
# ## Some variants of SVD
# 
# 
# 
# -  SVD with bias: $\hat{r_{ui}} = \mu + b_u + b_i + {q_i}^Tp_u$
# - SVD ++: $\hat{r_{ui}} = \mu + b_u + b_i + {q_i}^T(p_u + |I_u|^{\frac{-1}{2}}\sum\limits_{j \in I_u}y_j)$
# 
# ## Factorization machine (FM)
# 
# FM takes into account user-item interactions and other features, such as users' contexts and items' attributes. It captures the second-order interactions of the vectors representing these features , thereby enriching FM's expressiveness. However, interactions involving less relevant features may introduce noise, as all interactions share the same weight. e.g. You may use FM to consider the features of items.
# 
# \begin{equation}
# \hat{y}_{FM}(\textbf{X}) = w_0 + \sum\limits_{j =1}^nw_jx_j + \sum\limits_{j=1}^n\sum\limits_{k=j+1}^n\textbf{v}_j^T\textbf{v}_kx_jx_k
# \end{equation}
# 
# where $\textbf{X} \in \mathbb{R}^n$ is the feature vector, $n$ denotes the number of features, $w_0$ is the global bias, $w_j$ is the bias of the $j$-th feature and $\textbf{v}_j^T\textbf{v}_k$ denotes the bias of interaction between $j$-th feature and $k$-th feature, $\textbf{v}_j \in \mathbb{R}^d$ is the vector representing $j$-th feature.
# 
# ## MLP
# 
# You may also represent users and items by vectors and them feed them into a MLP to make prediction.
# 
# ## Metrics
# 
# - \begin{equation}
# RMSE = \sqrt{\frac{1}{|\mathcal{T}|}\sum\limits_{(u,i)\in\mathcal{T}}{(\hat{r}_{(u,i)}-r_{ui})}^2}
# \end{equation}
# 
# - \begin{equation}
# MAE = \frac{1}{|\mathcal{T}|}\sum\limits_{(u,i)\in\mathcal{T}}{|\hat{r}_{(u,i)}-r_{ui}|}
# \end{equation}
# -  Bonnus: you may also consider NDCG and HR under the top-k setting
# 

# # Requirements
# - Minimizing the RMSE and MAE
# - Try to compare different methods that you have adopted and interpret the results that you have obtained
# - Construct a recommender system that returns the top 10 movies that the users have not watched in the past
# - Before January 7th

# In[7]:


import pickle
import pandas as pd


# ## Récupération des données

# In[5]:


with open('movie_metadata.pkl', 'rb') as file:
    metadata = pickle.load(file)


# In[8]:


metadata = pd.DataFrame(metadata)


# In[9]:


metadata


# In[10]:


with open('movie_ratings_500_id.pkl', 'rb') as file:
    grades = pickle.load(file)


# ## Mise en forme des données pour les première baselines

# In[15]:


# Récupérer la liste de tous les utilisateurs

users = []

for film, grade in grades.items():
    for g in grade:
        user = g['user_id']
        if user not in users:
            users.append(user)

users = set(users)


# In[320]:


# Obtenir un dictionnaire tel que clés = user, valeurs = dict avec clés = films, valeurs = notes

notes = {}
cpt = 1
for user in users:
    print(cpt)
    fn = {} # Pour chaque user on crée un dic avec en clé les films et en valeur la note attribuée à ce film par ce user
    for film in metadata.columns:
        for f, g in grades.items():
            if film == f:
                for v in g:
                    if v['user_id'] == user:
                        fn[film] = v['user_rating']
    notes[user] = fn
    cpt+=1

notes


# In[25]:


for user in notes.keys():
    for film in metadata.columns:
        if film not in notes[user].keys():
            notes[user][film] = None

notes


# # Baselines

# ## Implémentation du User_based collaborative filtering

# In[33]:


# User-based collaborative filtering

# Trouver les users les plus proches de u

# Indice de corrélation de Pearson
import math
def pearson(person1, person2):
    sum_xy=0
    sum_x=0
    sum_y=0
    sum_x2=0
    sum_y2=0
    n=0
    for key in notes[person1].keys():
          if (key in notes[person2].keys()) & (notes[person2][key] != None) & (notes[person1][key] != None):
            n += 1
            x=int(notes[person1][key])
            y=int(notes[person2][key])
            sum_xy +=x*y
            sum_x += x
            sum_y += y
            sum_x2 += x**2
            sum_y2 += y**2
    # ajouter le caq n = 0 (aucun film en commun)
    if n == 0:
        denominator = 0
    else:
        denominator = math.sqrt(sum_x2 - (sum_x**2) / n) * math.sqrt(sum_y2 - (sum_y**2) / n)
    if denominator == 0:
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) /n ) / denominator


# In[41]:


# Création d'un dictionnaire avec pour clés les users, et pour valeur un dictionnaire avec pour clé les autres users
# et commme valeur le coefficient de pearson entre les deux users s'il est supérieur à 0
# pour avoir un ensemble de voisins cohérent

# Les voisins seront les users qui ont un pearson supérieur à 0

dic_pearson = {}
cpt = 1
for user1 in users:
    print(cpt)
    dic_pearson[user1] = []
    for user2 in users:
        p = pearson(user1, user2)
        if (user2 != user1) and (p > 0):
            dic_pearson[user1].append(user2)
    if (cpt > 10): # on arrête à 10 provisoirement car trop long à charger
        break
    cpt+=1

dic_pearson


# In[55]:


# Calcul du r^

r_chap = {}

for user in dic_pearson.keys():
    r_chap[user] = {}
    # Certains users n'ont pas de voisins (aucun film vu en commun), d'où les dic vides dans r_chap
    for neig in dic_pearson[user]:
        for film in notes[user]:
            if ((notes[neig][film]) != None) and ((notes[user][film]) == None):
                r_chap[user][film] = (pearson(user, neig)*int(notes[neig][film]))/abs(pearson(user, neig))

r_chap


# In[290]:


# On prend les 10 films avec les meilleures notes de r^ en guise de recommandation pour chaque user

reco1 = {}

for user in r_chap.keys():
    reco1[user] = []
    r_chap[user] = dict(sorted(r_chap[user].items(), key=lambda item:item[1], reverse=True))
    cpt = 0
    for film in r_chap[user].keys():
        if cpt < 10:
            reco1[user].append(film)
            cpt += 1

reco1

# Mettre les titres


# In[68]:


# Remplacer rui par ruj sinon pas possible

# Mesure de similarité adaptée au dataframe des films (données non quantitatives)

# Besoin de connaître le nombre total d'acteurs et de genres

acteurs = []
genres = []

for film in metadata.columns:
    for acteur in metadata[film]['actors']:
        if acteur not in acteurs:
            acteurs.append(acteur)
    for gen in metadata[film]['genre']:
        if gen not in genres:
            genres.append(gen)

# On met des coef pour les ressemblances selon les lignes
# Total = 1, réalisateur = 0.4, genre = 0.35, acteurs = 0.25

coef_real = 0.4
coef_genre = 0.35
coef_act = 0.25


# In[69]:


def simFilms(film1, film2):
    dir = 0
    if (metadata[film1]['director'] == metadata[film2]['director']):
        dir += 1
    act_cpt = 0
    for act in metadata[film1]['actors']:
        if (act in metadata[film2]['actors']):
            act_cpt += 1
    total_act = (len(metadata[film1]['actors']) + len(metadata[film2]['actors']))/2
    score_act = act_cpt/total_act
    genr = 0
    for gen in metadata[film1]['genre']:
        if (gen in metadata[film2]['genre']):
            genr += 1
    total_genre = (len(metadata[film1]['genre']) + len(metadata[film2]['genre']))/2
    score_genre = genr/total_act
    
    score_total = dir*coef_real + score_act*coef_act + score_genre*coef_genre
    return score_total


# In[75]:


# Création d'un dictionnaire avec pour clés les films, et pour valeur un dictionnaire avec pour clé les autres films
# et commme valeur le score de similarité entre les deux films s'il est supérieur à 0
# pour avoir un ensemble de voisins cohérent

# Les voisins seront les users qui ont un score de similarité supérieur à 0


dic_sim = {}
cpt = 1
for film1 in metadata.columns:
    print(cpt)
    dic_sim[film1] = []
    for film2 in metadata.columns:
        sim = simFilms(film1, film2)
        if (film2 != film1) and (sim > 0):
            dic_sim[film1].append(film2)
    cpt += 1

dic_sim


# ## Implémentation du Item-based collaborative filtering

# In[86]:


# Calcul du r^

r_chap_item = {}
cpt = 1
for user in users:
    print(cpt)
    r_chap_item[user] = {}
    for film in dic_sim.keys():
        r_chap_item[film] = {}
        sum_num = 0
        sum_den = 0
        r = 0
        # Certains users n'ont pas de voisins (aucun film vu en commun), d'où les dic vides dans r_chap
        for neig in dic_sim[film]:
            if (notes[user][neig] != None):
                sum_num += simFilms(film, neig)*int(notes[user][neig])
                sum_den += abs(simFilms(film, neig))
            else:
                sum_num += 0
                sum_den += 0
        if (sum_den != 0):
            r_chap_item[user][film] = sum_num/sum_den
        else:
            r_chap_item[user][film] = None
    cpt += 1

r_chap_item


# In[94]:


# On prend les 10 films avec les meilleures notes de r^ en guise de recommandation pour chaque user

reco = {}

for user in r_chap_item.keys():
    reco[user] = []
    r_chap_item1[user] = {k: v for k, v in r_chap_item[user].items() if v != None}
    r_chap_item1[user] = dict(sorted(r_chap_item1[user].items(), key=lambda item:item[1], reverse=True))
    cpt = 0
    for film in r_chap_item1[user].keys():
        if cpt < 10:
            reco[user].append(film)
            cpt += 1

reco

# Mettre les titres


# In[98]:


cpt_pareil = 0

for user in users:
    if (user in reco1.keys()) and (user in reco.keys()):
        for liste1 in reco1.values():
            for liste2 in reco.values():
                for film in liste1:
                    if (film in liste2):
                        cpt_pareil +=1

total_pareil = 100*369680
pareil = cpt_pareil/total_pareil
pareil


# # Implémentation de Vanilla MF

# In[318]:


pip install surprise


# In[322]:


from surprise import Dataset
from surprise import SVD
from surprise import Reader
from surprise.model_selection import cross_validate
# Construction des données avec le format attendu par Surprise
ratings_list = []
for user, film_ratings in notes.items():
    for film, rating in film_ratings.items():
        if rating != None:
            ratings_list.append({'user': user, 'item': film, 'rating': float(rating)})

#df avec les ratings
ratings_df = pd.DataFrame(ratings_list)

# chartgement sous le format attendu par Surprise
reader = Reader(rating_scale=(0, 5))  # Spécifier l'échelle de notation
data = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)

# MF (SVD) avec Surprise
model = SVD()

# Évaluation avec une validation croisée
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[323]:


trainset = data.build_full_trainset()
model.fit(trainset)


# In[324]:


user_id = 2473404
item_id = 'tt0323807'
predicted_rating = model.predict(user_id, item_id).est
print(f"La prédiction de notation pour l'utilisateur {user_id} sur le film {item_id} est : {predicted_rating}")


# # Implémentation MLP

# In[332]:


pip install tensorflow


# In[333]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[334]:


# user and movie df
notes_df = pd.DataFrame([(user, movie, rating) for user, movies in notes.items() for movie, rating in movies.items()],
                         columns=['user_id', 'movie_id', 'rating'])


# Convert ratings to numeric values
notes_df['rating'] = pd.to_numeric(notes_df['rating'], errors='coerce')\

# user and movie indices
user_index = {user: idx for idx, user in enumerate(users)}
movie_index = {movie: idx for idx, movie in enumerate(notes_df['movie_id'].unique())}

# Map user and movie IDs to indices
notes_df['user_index'] = notes_df['user_id'].map(user_index)
notes_df['movie_index'] = notes_df['movie_id'].map(movie_index)

# Filter out rows with missing ratings
filtered_notes_df = notes_df.dropna(subset=['rating'])

# Standardize the ratings
scaler = StandardScaler()
filtered_notes_df['rating'] = scaler.fit_transform(filtered_notes_df['rating'].values.reshape(-1, 1))


# Split the data into training and testing sets
train_data, test_data = train_test_split(filtered_notes_df, test_size=0.2, random_state=42)

# Define the model
embedding_dim = 50

user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')

user_embedding = Embedding(input_dim=len(user_index), output_dim=embedding_dim, input_length=1)(user_input)
movie_embedding = Embedding(input_dim=len(movie_index), output_dim=embedding_dim, input_length=1)(movie_input)

user_flatten = Flatten()(user_embedding)
movie_flatten = Flatten()(movie_embedding)

concat = Concatenate()([user_flatten, movie_flatten])
dense1 = Dense(64, activation='relu')(concat)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(1, activation='linear')(dense2)

model = Model(inputs=[user_input, movie_input], outputs=output)


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Prepare the input data
X_train = [train_data['user_index'].values, train_data['movie_index'].values]
y_train = train_data['rating'].values

X_test = [test_data['user_index'].values, test_data['movie_index'].values]
y_test = test_data['rating'].values

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# In[335]:


# Evaluate the model
predictions = model.predict(X_test)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(predictions)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
mae = mean_absolute_error(y_test_original, predictions)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Compare predictions with actual ratings
results = pd.DataFrame({
    'User ID': test_data['user_id'],
    'Movie ID': test_data['movie_id'],
    'Actual Rating': scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(),
    'Predicted Rating': predictions.flatten()
})

print(results)


# # Implémentation du modèle Funk SVD : notre modèle de recommandation

# In[301]:


# Remise en forme des données pour adapter au modèle de SVD

# Avoir un df avec le user_id, le item_id et le rating en colonne
# Actuellement on a un dictionnaire où la clé est le item_id

for film in grades.keys():
    ratings = pd.DataFrame(columns=['user_id','item_id','rating', 'timestamp'])
    path = "ratings/" + film + ".csv"
    for grading in grades[film]:
        ratings = ratings._append({"user_id": grading['user_id'], "item_id": film[2:10], "rating": grading['user_rating'], 'timestamp': grading['user_rating_date']}, ignore_index=True)
    ratings.to_csv(path, index=False)


# On crée un fichier csv par film (enregistrés dans un dossier "ratings" qui lui est au même niveau que le notebook) et on les concatène par la suite, tout faire d'un coup aurait pris trop de temps

# In[302]:


# Concaténation des csv (on a un csv de notation par film)

notations = pd.DataFrame(columns=['user_id','item_id','rating', 'timestamp'])

import os

for file in os.listdir('ratings'):
    df = pd.read_csv('ratings/' + file)
    notations = pd.concat([notations, df])


# Par la suite on doit modifier les ID des users et des items pour qu'ils soient tous au format numérique et qu'ils aillent de 1 à 36968 pour les users et de 1 à 528 pour les films (afin de coller au modèle que l'on implémente)

# In[303]:


# Remplacer les user ID par des valeurs de 1 à 36968
ids = {}
cpt = 1
for user in notations['user_id']:
    if user not in ids.keys():
        ids[user] = cpt
        cpt += 1


# In[304]:


# Remplacer les item ID par des valeurs de 1 à 36968
idsf = {}
cpt = 1
for item in notations['item_id']:
    if item not in idsf.keys():
        idsf[item] = cpt
        cpt += 1
print(idsf)


# In[255]:


user_list = []

for user in notations['user_id']:
    user_list.append(ids[user])


# In[256]:


item_list = []

for item in notations['item_id']:
    item_list.append(idsf[item])


# In[257]:


# Remplacer la colonne par sa nouvelle valeur
notations.insert(0, "user_id2", user_list, allow_duplicates=False)


# In[258]:


notations.insert(2, "item_id2", item_list, allow_duplicates=False)


# In[260]:


notations.pop('user_id')
notations.pop('item_id')


# In[262]:


notations.columns = ['user_id', 'item_id', 'rating', 'date']


# In[263]:


# Export du csv global pour appeler avec data_path après
notations.to_csv("notations.csv", index=False, sep='\t', header=True)


# In[264]:


data_path:str = 'notations.csv'


# Maintenant que les données sont au bon format, on peut implémenter le modèle de SVD vu en cours

# In[280]:


# Modèle SVD

import pickle
import numpy as np
import matplotlib.pyplot as plt

class Funk_SVD:
    def __init__(self, path, USER_NUM, ITEM_NUM, FACTOR, EPOCHS:int=10, THETA:float=1e-4, ALPHA:float=0.002, BETA:float=0.02):
        super(Funk_SVD, self).__init__()
        self.path = path
        self.USER_NUM = USER_NUM
        self.ITEM_NUM = ITEM_NUM
        self.FACTOR = FACTOR
        self.EPOCHS = EPOCHS
        self.THETA = THETA
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.init_model()

    def load_data(self, flag='train', sep='\t', random_state=0, size=0.8):
        np.random.seed(random_state)
        with open(self.path, 'r') as f:
            for index, row in enumerate(f):
                if index == 0:
                    continue
                rand_num = np.random.rand()
                if flag == 'train':
                    if rand_num < size:
                        u, i, r, t = row.strip('\r\n').split(sep)
                        yield (int(u) - 1, int(i) - 1, float(r))
                else:
                    if rand_num >= size:
                        u, i, r, t = row.strip('\r\n').split(sep)
                        yield (int(u) - 1, int(i) - 1, float(r))

    def init_model(self):
        '''
        Initialize matrices P and Q with np.random.rand
        '''
        self.P = np.random.rand(self.USER_NUM, self.FACTOR)/(self.FACTOR**0.5)
        self.Q = np.random.rand(self.ITEM_NUM, self.FACTOR)/(self.FACTOR**0.5)

    def train(self):
        '''
        Train the model
        epochs - number of iterations
        theta - threshold of iterations
        alpha - learning rate
        beta - parameter of regularization term
        '''
        epochs, theta, alpha, beta = self.EPOCHS, self.THETA, self.ALPHA, self.BETA
        old_e = 0.0
        self.cost_of_epoch = []
        for epoch in range(epochs):
            print(f"Current epoch is {epoch + 1}")
            current_e = 0.0
            train_data = self.load_data(flag='train')
            for index, d in enumerate(train_data):
                u, i, r = d
                pr = np.dot(self.P[u], self.Q[i])
                err = r - pr
                current_e += pow(err,2) # loss term
                self.P[u] += alpha * (err*self.Q[i]-beta*self.P[u])
                self.Q[i] += alpha * (err*self.P[u]-beta*self.Q[i])
                current_e += (beta/2)*(sum(pow(self.P[u],2)))+sum(pow(self.Q[i],2)) # regularization
            self.cost_of_epoch.append(current_e)
            print(f'Cost is {current_e}')
            if abs(current_e - old_e) < theta:
                break
            old_e = current_e
            alpha *= 0.9

    def predict_rating(self, user_id, item_id):
        '''
        Predict rating for the target user of the target item

        user - the number of the user (user_id=xuhao-1)
        item - the number of the item (item_id=xuhao-1)
        '''
        return np.dot(self.P[user_id], self.Q[item_id])

    def recommend_list(self, user, k=10):
        '''
        Recommend the top n movies for the target user. For rating prediction, recommend items with a rating above 4/5 of the maximum rating.
        '''
        user_id = user-1
        user_items = {}
        for item_id in range(self.ITEM_NUM):
            user_had_look = self.user_had_look_in_train()
            if item_id in user_had_look[user]:
                continue
            pr = self.predict_rating(user_id,item_id)
            user_items[item_id] = pr
            items = sorted(user_items.items(), key=lambda x:x[1],reverse=True)[:k]
        return items

    def user_had_look_in_train(self):
        '''
        Write a function that returns the movies that users have already watched in the past
        '''
        user_had_look = {}
        train_data = self.load_data(flag='train')
        for index, d in enumerate(train_data):
            u,i,r = d
            user_had_look.setdefault(u,{})
            user_had_look[u][i] = r
        return user_had_look

    def test_rmse(self):
        '''
        Test the model and return the RMSE value
        '''
        rmse = 0
        num = 0
        test_data = self.load_data(flag='test')
        for index, d in enumerate(test_data):
            num = index+1
            u,i,r = d
            pr = np.dot(self.P[u], self.Q[i])
            rmse += pow((r-pr),2)
        rmse = (rmse/num)**0.5
        return rmse
    
    def test_mae(self):
        '''
        Test the model and return the MAE value
        '''
        mae = 0
        num = 0
        test_data = self.load_data(flag='test')
        for index, d in enumerate(test_data):
            num = index+1
            u,i,r = d
            pr = np.dot(self.P[u], self.Q[i])
            mae += abs(r-pr)
        mae = (mae/num)
        return mae

    def show(self):
        '''
        Create a function that plots the loss after each iteration
        '''
        nums=range(len(self.cost_of_epoch))
        plt.plot(nums, self.cost_of_epoch,label='cost value')
        plt.xlabel('# of epoch')
        plt.ylabel('cost')
        plt.legend()
        plt.show()

    def save_model(self):
        '''
        Save the model to pickle (P, Q, and RMSE)
        '''
        data_dict = {'P': self.P, 'Q': self.Q}
        f = open('funk-svd.pkl', 'wb')
        pickle.dump(data_dict, f)
        pass

    def read_model(self):
        '''
        Reload the model from the local disk
        '''
        f = open('funk-svd.pkl', 'rb')
        model = pickle.load(f)
        self.P = model['P']
        self.Q = model['Q']
        pass


# In[284]:


mf=Funk_SVD(data_path,36968,528,50,EPOCHS=100)#path,user_num,item_num,factor
mf.train()
mf.save_model()
mf.show()
rmse=mf.test_rmse()
mae=mf.test_mae()
print("rmse:",rmse)
print("mae:",mae)


# On remarque que la courbe de coût est croissante après deux itérations. Cependant, plus on augmente le nombre d'itérations, plus on diminue les RMSE et MAE.

# # Comparaison de user-based collaborative filtering avec SVD

# On fixe un user au hasard, ici le 16402, pour voir si les recommandations de la baseline user-based collaborative filtering et SVD ont des films en commun

# In[293]:


# On prend un utilisateur ayant un id présent dans la partie de la recommandation par user-ased (puisqu'on en a qu'une partie)
id = ids[2473404]
id


# In[294]:


mf=Funk_SVD(data_path,36968,528,50,EPOCHS=100)#path,user_num,item_num,factor
mf.train()
reco_svd = mf.recommend_list(16402)

reco_svd


# In[314]:


reco_svd2 = []

for f in reco_svd:
    reco_svd2.append(f[0])

reco_svd2


# In[311]:


# Récupération des recommandations user-based pour le user 2473404

reco_ub = reco1['2473404']
reco_ub_newids = []

for film in reco_ub:
    reco_ub_newids.append(idsf[int(film[3:10])])

reco_ub_newids


# In[315]:


# Comparaison

cpt_identq = 0

for film in reco_ub_newids:
    if film in reco_svd2:
        cpt_identq+=1

cpt_identq


# # Comparaison de item-based collaborative filtering avec SVD

# In[313]:


# Récupération des recommandations item-based pour le user 2473404

reco_ib = reco['2473404']
reco_ib_newids = []

for film in reco_ib:
    reco_ib_newids.append(idsf[int(film[3:10])])

reco_ib_newids


# In[316]:


# Comparaison

cpt_identq2 = 0

for film in reco_ib_newids:
    if film in reco_svd2:
        cpt_identq2+=1

cpt_identq2


# Un film est recommandé par SVD et user-based collaborative filtering. SVD et item-based collaborative filtering n'en ont pas en commun.

# ## Comparaison de Vanilla MF avec SVD

# In[328]:


# Établir la liste des meilleurs films pour notre user selon Vanilla MF

user_id = 2473404
item_id = 'tt0323807'
predicted_rating = model.predict(user_id, item_id).est

notes_vanilla = {}

for film in metadata.columns:
    item_id = film
    notes_vanilla[film] = model.predict(user_id, item_id).est

notes_vanilla = dict(sorted(notes_vanilla.items(), key=lambda item:item[1], reverse=True))

reco_vanilla = []
cpt = 0
for k in notes_vanilla.keys():
    if (cpt < 10):
        reco_vanilla.append(k)
        cpt+=1

reco_vanilla2 = []

for film in reco_vanilla:
    reco_vanilla2.append(idsf[int(film[3:10])])

reco_vanilla2


# In[329]:


# Comparaison

cpt_identq3 = 0

for film in reco_vanilla2:
    if film in reco_svd2:
        cpt_identq3+=1

cpt_identq3


# Un film est recommandé par SVD et Vanilla MF

# # Comparaison SVD et MLP

# In[342]:


reco_mlp = list(results[results['User ID'] == str(user_id)]['Movie ID'])
reco_mlp2 = []
for film in reco_mlp:
    reco_mlp2.append(idsf[int(film[3:10])])
reco_mlp2


# In[343]:


# Comparaison

cpt_identq4 = 0

for film in reco_svd2:
    if film in reco_mlp2:
        cpt_identq4 += 1

cpt_identq4


# Un film recommandé par MLP est également recommandé par SVD, ce qui est positif sachant que dans le test set de MLP, seulement deux films sont attribués au user

# In[ ]:




