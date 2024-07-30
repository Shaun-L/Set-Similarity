#Function: accepts 2 lists, loop thorugh ach word in each list and compare in the other. take the cosine similarity
import numpy as np
from operator import itemgetter
from sentence_transformers import SentenceTransformer, util

list2 = ["apple", "orange", "book", "aple", "seattle", "seatac", "manhattan"]
list1 = ["aple", "fruit", "New york", "washington"]
list3 = ["happy", "sad", "california", "east coast", "confusing", "beach", "pineapple"]
list4 = ["great", "laptop", "flll"]
theta = 0.5

model = SentenceTransformer("all-MiniLM-L6-v2")  # Load the SentenceTransformer model

def get_word_embedding(token):
    return model.encode(token)

def cosine_similarity(vec1, vec2):
  cos_sim = util.cos_sim (vec1, vec2)
  return cos_sim

def name_similarity(name1, name2):
    embedding1 = get_word_embedding(name1)
    embedding2 = get_word_embedding(name2)
    if np.linalg.norm(embedding1) != 0 and np.linalg.norm(embedding2) != 0:
        similarity = cosine_similarity(embedding1, embedding2)  
    else:
        print("invalid embeddings")
        return 0
    return similarity

def set_similarity(list1, list2):

  if len(list1) > len(list2): #This makes sure that list1 is shorter or equal to list2
    longer = list1
    shorter = list2
  else:
    longer = list2
    shorter = list1

  #creates all tuples of all combination between the two sets formatted like: (word1, word2, cosineSimilarity)
  sim_scores = [] 
  for i in shorter:
    temp = ("", "", 0)
    score = 0
    for j in longer:
      if name_similarity(i,j).item() > score:
        score = name_similarity(i, j).item()
        temp = (i, j, score)
    sim_scores.append(temp)
  
  return sim_scores


print(set_similarity(list1, list2))





