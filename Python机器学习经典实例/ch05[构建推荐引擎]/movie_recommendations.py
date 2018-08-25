import json
import numpy as np

from euclidean_score import euclidean_socre
from pearson_score import pearson_socre
from find_similar_users import find_similar_users

# 为用户生成电影推荐
def generate_recommendations(dataset, user):
    if user not in dataset:
        raise TypeError('User ' + user + ' not present in the dataset')

    total_scores = {}
    similarity_sums = {}

    # 计算该用户与数据库中其他用户的皮尔逊相关系数
    for u in [x for x in dataset if x != user]:
        similarity_scores = pearson_socre(dataset, user, u)

        if similarity_scores <= 0:
            continue

        # 找到还没被该用户评价的电影
        for item in [x for x in dataset[u] if x not in dataset[user] or dataset[user][x] == 0]:
            total_scores.update({item: dataset[u][item] * similarity_scores})
            similarity_sums.update({item: similarity_scores})

    # 如果该用户看过数据库中的所有电影，则不推荐
    if len(total_scores) == 0:
        return ['No recommendations possible']

    # 生成一个电影评分标准列表
    movie_ranks = np.array([[total / similarity_sums[item], item] for item, total in total_scores.items()])

    # 提取推荐的电影
    recommendations = [movie for _, movie in movie_ranks]

    return recommendations

if __name__ == "__main__":
    data_file = 'movie_ratings.json'

    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user = 'Michael Henry'
    print("\nRecommendations for " + user + ":")
    movies = generate_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i + 1) + '. ' + movie)

    user = 'John Carson'
    print("\nRecommendations for " + user + ":")
    movies = generate_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i + 1) + '. ' + movie)