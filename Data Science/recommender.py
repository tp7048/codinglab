# -*- coding: utf-8 -*-
"""
Created on Sun May 25 10:03:32 2025

@author: jinu
"""

import sys
import math
from collections import defaultdict

# calculate similarity between two matrices
def cosine_similarity(user_ratings1, user_ratings2):
    common_items = set(user_ratings1.keys()) & set(user_ratings2.keys())
    if not common_items:
        return 0

    sum1_2 = sum(user_ratings1[i] * user_ratings2[i] for i in common_items)
    sum1_1 = math.sqrt(sum([user_ratings1[i] ** 2 for i in common_items]))
    sum2_2 = math.sqrt(sum([user_ratings2[i] ** 2 for i in common_items]))

    if sum1_1 == 0 or sum2_2 == 0:
        return 0

    return sum1_2 / (sum1_1 * sum2_2)

# predict ratings
def predict_rating(user_id, item_id, user_ratings, similarity_matrix, k=5):
    similarities = []
    for other_user_id, ratings in user_ratings.items():
        if other_user_id != user_id and item_id in ratings:
            sim = similarity_matrix[user_id].get(other_user_id, 0)
            if sim > 0:
                similarities.append((sim, ratings[item_id]))

    # default value
    if not similarities:
        return 3.0

    similarities.sort(reverse=True)
    similarities = similarities[:k]

    numerator = sum(sim * rating for sim, rating in similarities)
    denominator = sum(abs(sim) for sim, _ in similarities)
    return numerator / denominator if denominator != 0 else 3.0

# read file and save
def load_data(filename):
    data = defaultdict(dict)
    with open(filename, 'r') as f:
        for line in f:
            user, item, rating, _ = line.strip().split('\t')
            data[int(user)][int(item)] = float(rating)
    return data

# create similarity matrix
def compute_user_similarity(user_ratings):
    users = list(user_ratings.keys())
    similarity_matrix = defaultdict(dict)

    for i, user1 in enumerate(users):
        for j in range(i + 1, len(users)):
            user2 = users[j]
            sim = cosine_similarity(user_ratings[user1], user_ratings[user2])
            similarity_matrix[user1][user2] = sim
            similarity_matrix[user2][user1] = sim

    return similarity_matrix

# for predict with test file
def run(train_file, test_file):
    train_data = load_data(train_file)
    test_data = []

    with open(test_file, 'r') as f:
        for line in f:
            user, item, rating, _ = line.strip().split('\t')
            test_data.append((int(user), int(item), float(rating)))

    similarity_matrix = compute_user_similarity(train_data)

    predictions = []
    for user, item, _ in test_data:
        pred_rating = predict_rating(user, item, train_data, similarity_matrix)
        predictions.append((user, item, round(pred_rating, 2)))

    output_file = train_file.replace('.base', '.base_prediction.txt')
    with open(output_file, 'w') as f:
        for user, item, rating in predictions:
            f.write(f"{user}\t{item}\t{rating}\n")

    print("Prediction complete")

# main
def main():
    if len(sys.argv) != 3:
        print("Usage Error: python recommender.py [train_file] [test_file]")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    run(train_file, test_file)

if __name__ == "__main__":
    main()
