# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 18:32:39 2018

@author: Administrator
"""

from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# 生成样本数据
X, y = samples_generator.make_classification(n_informative=4, n_features=20,
                                             n_redundant=0, random_state=5)

# 特征选择器
# 选择k个最好的特征
selector_k_best = SelectKBest(f_regression, k=10)

# 随机森林分类器
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)

# 构建机器学习流水线
pipeline_classifier = Pipeline([('selector', selector_k_best), ('rf', classifier)]) # 将特征选择器命名为selector，随机森林分类器命名为rf
pipeline_classifier.set_params(selector__k=6, rf__n_estimators=25) # 更新参数

# 训练分类器
pipeline_classifier.fit(X, y)

# 输出预测结果
prediction = pipeline_classifier.predict(X)
print("Prediction: \n",prediction)

# 打印模型的得分
print("Score:", pipeline_classifier.score(X, y))

# 打印被分类器选择的特征
features_state = pipeline_classifier.named_steps['selector'].get_support()
selected_features = []
for count, item in enumerate(features_state):
    if item:
        selected_features.append(count)
print("Selected features (0-index): ", ', '.join([str(x) for x in selected_features]))








