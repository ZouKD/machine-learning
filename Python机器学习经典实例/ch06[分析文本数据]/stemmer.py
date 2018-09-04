from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

# 定义一些单词进行词干提取
words = ['table', 'probably', 'wolves', 'playing', 'is', 'dog', 'the', 'beaches',
         'grounded', 'dreamt', 'envision']

# 定义一个稍后会用到的词干提取器到列表
# 对比不同的词干提取算法
stemmers = ['PORTER', 'LANCASTER', 'SNOWBALL']

# 初始化3个词干提取器对象
stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmer_snowball = SnowballStemmer('english')

# 为了以整齐的表格形式将输出数据打印出来，需要设定其正确的格式：
formated_row = '{:>16}' * (len(stemmers) + 1)
print('\n', formated_row.format('WORD', *stemmers), '\n')

# 迭代列表中的单词，并用3个词干提取器进行词干提取：
for word in words:
    stemmed_words = [stemmer_porter.stem(word), stemmer_lancaster.stem(word),
                     stemmer_snowball.stem(word)]
    print(formated_row.format(word, *stemmed_words))
