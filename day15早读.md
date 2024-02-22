

# 早读背诵面试题：

## 2.请介绍下你知道的文本表征的方法(词向量)。

基于 one-hot、tf-idf、textrank；
主题模型：LSA（SVD）、pLSA、LDA；
基于词向量的固定表征：Word2vec、FastText、GloVe；
基于词向量的动态表征：ELMo、GPT、BERT



## 3.如何生成句向量？

doc2vec
bert
词向量拼接、平均、tf-idf加权平均



## 4.如何计算文本似度

基于字符：最小编辑距离
基于向量：转为词向量或句向量后利用余弦距离或欧氏距离计算
有监督方法：构建分类器



## 5.样本不平衡的解决方法？

过采样
欠采样
文本增强



## 6.过拟合有哪些表现，怎么解决？

一般 training accuracy 特别高，但是testing accuracy 特别低时，预示可能出现了过拟合
如何解决过拟合：
增加数据量
数据增强
加入L1，L2正则
Dropout
Batch Normalization
early stopping



## 7.用过 jiaba 分词吗，了解原理吗

基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图(DAG)
采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
对于未登录词，采用了基于汉字成词能力的HMM 模型，使用了Viterbi 算法



## 8.了解命名实体识别吗？通常用什么方法，各自有什么特点

CRF。需要人工编写特征，训练速度较快
BiLSTM_CRF。自动抽取特征，训练速度较慢，需要大量标注数据
BERT_CRF。自动抽取特征，训练速度较慢，需要大量标注数据



## 9.了解HMM和CRF吗？

均属于概率图模型，常见于序列标注任务。CRF表现通常好于HMM
HMM属于生成式模型，CRF属于判别式模型



## 10.了解RNN吗，LSTM呢，LSTM相对RNN有什么特点

RNN， 循环神经网络（Recurrent Neural Network）是一种用于处理序列数据的神经网络。相比一般的神经网络来说，他能够处理序列变化的数据
LSTM，长短期记忆（Long short-term memory）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现



## 11.会用正则表达式吗？re.match() 和 re.search() 有什么区别？

前者匹配string 开头，成功返回Match object, 失败返回None，只匹配一个
后者在string 中进行搜索，成功返回Match object, 失败返回None, 只匹配一个