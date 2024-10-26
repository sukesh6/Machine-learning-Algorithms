import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

game=pd.read_csv(r"C:/Users/sukes/OneDrive/Desktop/360/ds Assingments/3d.Recommendation Engine_ProbStatement/Dataset/Datasets_Recommendation Engine/game.csv")

from sqlalchemy import create_engine
from urllib.parse import quote

pw=quote('**********')
user='root'
db='datascience'

engine=create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
game.to_sql('game_tbl',con=engine,if_exists='replace',chunksize=1000,index=False)
sql='select * from game_tbl'
game=pd.read_sql_query(sql,con=engine)

game.isnull().sum()
game.duplicated().sum()
game.info()
game.describe()

# Create a TfidfVectorizer object with English stop words
tfidf=TfidfVectorizer(stop_words='english')

# Transform the 'genre' column into a TF-IDF matrix
tfidf_matrix=tfidf.fit(game.game)

# Save the TF-IDF matrix using joblib
joblib.dump(tfidf_matrix,'game')

# Load the saved TF-IDF matrix using joblib
game1=joblib.load('matrix')

tfidf_matrix=game1.transform(game.game)
tfidf_matrix.shape

# Calculate the cosine similarity matrix between the TF-IDF vectors
cosine_sim_matrix=cosine_similarity(tfidf_matrix,tfidf_matrix)
cosine_sim = cosine_similarity(tfidf_matrix)

game_index=pd.Series(game.index,index=game['game']).drop_duplicates()
game_id=game_index['Grand Theft Auto IV']
game_id

topN=5
def get_recommendations(Name, topN):
    game_id = game_index[Name]  
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id].flatten()))  
    cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)  
    cosine_scores_N = cosine_scores[1:topN + 1]  
    game_idx = [i[0] for i in cosine_scores_N]  
    '''game_scores = [i[1] for i in cosine_scores_N]  
    game_similar_show = pd.DataFrame(columns=["name", "Score"])  
    game_similar_show["name"] = game.loc[game_idx, "name"]   
    game_similar_show["Score"] = game_scores  '''

    #game_similar_show.reset_index(inplace=True)  

    #return (game_similar_show.iloc[1:, ]) 
    return game['game'].iloc[game_idx]
rec = get_recommendations("Grand Theft Auto IV", topN=5)
print(rec)