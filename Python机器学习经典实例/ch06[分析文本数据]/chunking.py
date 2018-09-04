"""
用分块的方法划分文本
    分块是指基于任意随即条件将输入文本分割成块。与标记解析不同的是，分块没有条件约束，
分块的结果不需要实际意义。分块在文本分析中经常使用。当处理非常大的文本时，就需要将文本
进行分块，以便进行下一步分析。在本文件中，将输入文本分成若干块，每块都包含固定数目的
单词。
"""
import numpy as np
from nltk.corpus import brown

# 将文本分割成块
def splitter(data, num_words):
    words = data.split(' ')
    output = []

    cur_count = 0
    cur_words = []

    for word in words:
        cur_words.append(word)
        cur_count += 1
        if cur_count == num_words:
            output.append(' '.join(cur_words))
            cur_words = []
            cur_count = 0
    output.append(' '.join(cur_words))

    return output

if __name__ == "__main__":
    # 从布朗语料库加载数据
    data = ' '.join(brown.words()[:10000])

    # 每块包含的单词数目
    num_words = 1700

    chunks = []
    counter = 0

    text_chunks = splitter(data, num_words)
    print('Number of text chunks = ', len(text_chunks))
