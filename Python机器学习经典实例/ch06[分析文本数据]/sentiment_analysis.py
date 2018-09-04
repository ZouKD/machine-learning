"""
分析句子的情感
    情感分析是NLP最受欢迎的应用之一。情感分析是指确定一段给定的文本是积极还是消极的
过程。有一些场景中，我们还会将‘中性’作为第三选项。情感分析常用于发现人们对一个特定
主题的看法。情感分析用于分析很多场景中的情绪，如营销活动、社交媒体、电子商务客户等。
"""
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def extract_features(word_list):
    return dict([(word, True) for word in word_list])

if __name__ == "__main__":
    # 加载积极与消极评论
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')

    # 将这些评论数据分成积极评论和消极评论
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Positive') for f in positive_fileids]

    features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Negative') for f in negative_fileids]

    # 分成训练数据集（80%）和测试数据集（20%）
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    # 提取特征
    features_train = features_positive[:threshold_positive] + \
        features_negative[:threshold_negative]

    features_test = features_positive[threshold_positive:] + \
        features_negative[threshold_negative:]
    print("Number of training datapoints: ", len(features_train))
    print("Number of test datapoints: ", len(features_test))

    # 训练朴素贝叶斯分类器
    classifizer = NaiveBayesClassifier.train(features_train)
    print("Accuracy of the classifizer: ", nltk.classify.util.accuracy(classifizer,
                                                                       features_test))
    print("Top 10 most informative words: ")
    for item in classifizer.most_informative_features()[:10]:
        print(item[0])

    # 输入一些简单的评论
    input_reviews = [
        "It is an amazing movie",
        "This if a dull moive. I wolud never recommend it to anyone.",
        "The cinematography is pretty great in this movie",
        "The direction was terrible and the story was all over the place"
    ]

    print('Predictions: ')
    for review in input_reviews:
        print("Reviews: ", review)
        probdist = classifizer.prob_classify(extract_features(review.split()))
        pre_sentiment = probdist.max()

        print("Predicted sentiment: ", pre_sentiment)
        print("Probability: ", round(probdist.prob(pre_sentiment), 2))


