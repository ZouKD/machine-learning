"""
创建文本分类器
    文本分类器的目的是将文本文档分类为不同的类，这是NLP中非常重要的分析手段。这里将
使用一种技术，它基于一种叫tf-idf的统计数据，它表示词频--逆文档频率（term frequency--
inverse document frequency）。这个统计工具有助于理解一个单词在一组文档中对某一文档
的重要性。它可以作为特征向量来做分类文档。
"""
from sklearn.datasets import fetch_20newsgroups

# 选择一个类列表，并且用词典映射的方式定义。这些类型是加载的新闻组的数据集的一部分：
category_map = {'misc.forsale': 'Sales', 'rec.motorcycles': 'Motorcycles',
                'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography',
                'sci.space': 'Space'}

# 基于刚刚定义的类型加载训练数据
training_data = fetch_20newsgroups(subset='train', categories=category_map.keys(),
                                   shuffle=True, random_state=7)

# 特征提取
from sklearn.feature_extraction.text import CountVectorizer

# 用训练数据提取特征
vectorizer = CountVectorizer()
X_train_termcounts = vectorizer.fit_transform(training_data.data)
print("Dinmensions of training data: ", X_train_termcounts.shape)

# 训练分类器
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

# 定义一些随机输入的句子
input_data = [
    "The curveballs of right headed pitchers tend to curve to the left",
    "Casar cipher is an ancient from of encryption",
    "This two-wheeler is really good on slippery roads"
]

# 定义tf-idf变换器对象，并训练
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)

# 多项式朴素贝叶斯分类器
classifier = MultinomialNB().fit(X_train_tfidf, training_data.target)


# 用词频统计转换输入数据
X_input_termcounts = vectorizer.transform(input_data)

# 用tf-idf变换器变换输入数据
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

# 用训练过的分类器预测这些输入句子的输出类型
predicted_categories = classifier.predict(X_train_tfidf)

# 打印输出
for sentence, category in zip(input_data, predicted_categories):
    print('\nInput: ', sentence, '\nPredicted category: ',
          category_map[training_data.target_names[category]])