"""
识别性别
"""
import random
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy

# 提取输入单词的特征
def gender_features(word, num_letters=2):
    return {'feature': word[-num_letters:].lower()}

if __name__ == "__main__":
    # 提取标记名称
    labeled_names = ([(name, 'male') for name in names.words('male.txt')]) + \
                    [(name, 'female') for name in names.words('female.txt')]

    # 设置随机生成数的种子值，并混合搅乱训练数据
    random.seed(7)
    random.shuffle(labeled_names)

    input_names = ['Leonardo', 'Amy', 'Sam']

    # 搜索参数空间
    for i in range(1, 5):
        print('Number of letters: ', i)
        featuresets = [(gender_features(n, i), gender) for (n, gender) in labeled_names]

        # 将数据分为训练数据集和测试数据集
        train_set, test_set = featuresets[500:], featuresets[:500]

        # 用朴素贝叶斯分类器做分类
        classifier = NaiveBayesClassifier.train(train_set)

        # 打印分类器的准确性
        print('Accuracy==>', str(100 * nltk_accuracy(classifier, test_set)) + str('%'))

        # 为新输入预测输出结果
        for name in input_names:
            print(name , '==>', classifier.classify(gender_features(name, i)))

