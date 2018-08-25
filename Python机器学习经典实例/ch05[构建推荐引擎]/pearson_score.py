import json
import numpy as np

# 计算user1和user2的皮尔逊相关系数
def pearson_socre(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')

    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')

    # 提取两个用户均评过分的电影
    rated_by_both = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    num_ratings = len(rated_by_both)
    # 如果两个用户都没评过分，得分为0
    if num_ratings == 0:
        return 0

    # 计算相同评分的电影平方值之和
    user1_sum = np.sum([dataset[user1][item] for item in rated_by_both])
    user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])

    # 计算所有相同评分的电影的平方和
    user1_square_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both])
    user2_square_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])

    # 计算数据集乘积之和
    product_sum = np.sum([dataset[user1][item]*dataset[user2][item] for item in  rated_by_both])

    # 计算皮尔逊相关系数
    Sxy = product_sum - (user1_sum * user1_sum / num_ratings)
    Sxx = user1_square_sum - np.square(user1_sum) / num_ratings
    Syy = user2_square_sum - np.square(user2_sum) / num_ratings

    if Sxx * Sxy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)

if __name__ == "__main__":
    data_file = 'movie_ratings.json'

    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user1 = 'John Carson'
    user2 = 'Michelle Peterson'

    print('Euclidean socre: ')
    print(pearson_socre(data, user1, user2))
