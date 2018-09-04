"""
用词形还原的方法还原文本的基本形式
    词形还原的目的也是将单词转换成其原型，但它是一个更结构化的方法。
"""
from nltk.stem import WordNetLemmatizer

# 定义一组单词来进行词形还原
words = ['table', 'probably', 'wolves', 'playing', 'is', 'dog', 'the',
         'beaches', 'grouned', 'dreamt', 'envision']

# 对比不同的词形还原器
lemmatizers = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']

# 基于WordNet词形还原器创建一个对象
lemmatizer_wordnet = WordNetLemmatizer()

formatted_row = '{:>24}' * (len(lemmatizers) + 1)
print('\n', formatted_row.format('WORD', *lemmatizers), '\n')

# 迭代列表中的单词，并用词形还原器进行词形还原
for word in words:
    lemmatized_words = [lemmatizer_wordnet.lemmatize(word, pos='n'),
                        lemmatizer_wordnet.lemmatize(word, pos='v')]
    print(formatted_row.format(word, *lemmatized_words))
