import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
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
    cos_sim = util.cos_sim(vec1, vec2)
    return cos_sim

def name_similarity(name1, name2):
    embedding1 = get_word_embedding(name1)
    embedding2 = get_word_embedding(name2)
    if np.linalg.norm(embedding1) != 0 and np.linalg.norm(embedding2) != 0:
        similarity = cosine_similarity(embedding1, embedding2).item()
    else:
        print("invalid embeddings")
        return 0.0
    return similarity

def set_similarity(list1, list2):
    if len(list1) > len(list2):  # This makes sure that list1 is shorter or equal to list2
        longer = list1
        shorter = list2
    else:
        longer = list2
        shorter = list1

    # Creates all tuples of all combinations between the two sets formatted like: (word1, word2, cosineSimilarity)
    sim_scores = []
    total_score = 0
    for i in shorter:
        temp = ("", "", 0)
        score = 0
        for j in longer:
            sim = name_similarity(i, j)
            if sim > score:
                score = sim
                temp = (i, j, score)
        sim_scores.append(temp)
        total_score += score
        #print(total_score)
    
    avg_sim_score = total_score / len(sim_scores) if sim_scores else 0
    print("The average similarity is", avg_sim_score)
    
    return avg_sim_score

# Example usage:
print(set_similarity(list3, list1))
