"""
创建词袋模型
    如果需要处理包含数百万单词的文档，需要将其转化成某种数值表示形式，以便让机器用这些
数据来学习算法。这些算法需要数值数据，以便可以对这些数据进行分析，并输出有用的信息。
这里需要用到词袋（bag_of_words）。词袋是从所有文档中的所有单词学习词汇的模型。学习之后，
词袋通过构建文档中所有单词的直方图来对每篇文档进行建模。
"""
import numpy as np
from nltk.corpus import brown
from chunking import splitter

if __name__ == "__main__":
    # 布朗语料库读取数据
    data = ' '.join(brown.words()[:10000])

    # 每块包含的单词数据
    num_words = 2000

    chunks = []
    counter = 0

    text_chunks = splitter(data, num_words)

    for text in text_chunks:
        chunk = {'index': counter, 'text': text}
        chunks.append(chunk)
        counter += 1

    # 提取文档-词矩阵
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df=5, max_df=.95)
    doc_term_matrix = vectorizer.fit_transform([chunk['text'] for chunk in chunks])

    # 从vectorizer对象中提取词汇，并打印
    vocab = np.array(vectorizer.get_feature_names())
    print('Vocabulary: ')
    print(vocab)

    # 打印文档-词矩阵
    print('Document term martrix: ')
    chunk_names = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3', 'Chunk-4']

    formatted_row = '{:>12}' * (len(chunk_names) + 1)
    print('\n', formatted_row.format('Word', *chunk_names, '\n'))

    for word, item in zip(vocab, doc_term_matrix.T):
        # 'item'是压缩的稀疏矩阵（csr_matrix）数据结构
        output = [str(x) for x in item.data]
        print(formatted_row.format(word, *output))
