# 示例文本
text = '''Are you curious about tokenization? Let's see how it works! We need to analyze a couple sentences with punctuations to see it in action'''

# 对句子进行解析
from nltk.tokenize import sent_tokenize
sent_tokenize_list = sent_tokenize(text) # 对输入文本运行句子解析器，提取出标记

print('Sentence tokenizer')
print(sent_tokenize_list)

'''
单词解析在NLP中是非常常用的。NLTK附带了几个不同的单词解析器。先从最基本的单词解析器开始：
'''
# 建立一个新的单词解析器
from nltk.tokenize import word_tokenize

print('Word tokenizer: ')
print(word_tokenize(text))


# 如果需要将标点符号保留到不同的句子标记中，可以用WordPunct标记解析器：
# 创建一个新的WordPunct标记解析器
from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()
print('Word punct tokenizer: ')
print(word_punct_tokenizer.tokenize(text))