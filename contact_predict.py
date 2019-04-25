# coding: utf-8
from __future__ import print_function


import tensorflow.contrib.keras as kr


from  rnn_model import  TRNNConfig,TextRNN


import os


import numpy as np
import tensorflow as tf
from sklearn import metrics


from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, open_file,native_content


try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
test_dir = os.path.join(base_dir, 'cnews.test.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
save_dir1 = 'checkpoints/textrnn'
save_path1 = os.path.join(save_dir1, 'best_validation')  # 最佳验证结果保存路径
g1 = tf.Graph() # 加载到Session 1的graph
g2 = tf.Graph() # 加载到Session 2的graph

sess1 = tf.Session(graph=g1) # Session1
sess2 = tf.Session(graph=g2) # Session2
class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = sess1
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        pred_matrix = self.session.run(self.model.pred_matrix, feed_dict=feed_dict)
        # return self.categories[y_pred_cls[0]],pred_matrix
        return y_pred_cls, pred_matrix

class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)

        self.session = sess2
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path1)  # 读取保存的模型


    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        pred_matrix = self.session.run(self.model.pred_matrix, feed_dict=feed_dict)
        # return self.categories[y_pred_cls[0]],pred_matrix
        return y_pred_cls, pred_matrix

def read_file2(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(native_content(content))
                    #contents.append(list((native_content(content))))
                    labels.append(native_content(label))
            except:
                pass
    return contents, labels

if __name__ == '__main__':
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    #print(x_test)
    #print(y_test)

    with sess1.as_default():
        with g1.as_default():
            cnn_model = CnnModel()

    with sess2.as_default():  # 1
        with g2.as_default():
            rnn_model = RnnModel()
    lables=[]
    x_test, y_test = read_file2(test_dir)
    data_id, label_id = [], []
    for i in range(len(y_test)):
        label_id.append(cat_to_id[y_test[i]])

    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    y_test_cls = np.argmax(y_pad, 1)
    trueLables=[]
    a=0.5
    # ************************************************************
    y_pred_labels = np.zeros(shape=len(x_test), dtype=np.int32)
    # ************************************************************
    for i in range(len(x_test)):
        item = x_test[i]
        lableItem = y_test[i]
        temp, result_cnn = cnn_model.predict(item)
        temp1, result_rnn  = rnn_model.predict(item)
        #print(result_cnn)
        #print(result_rnn)
        #result =a* result_cnn + (1-a)*result_rnn
        result =  result_cnn
        print(result)
        labelId = np.argmax(result)
       # print(labelId)
        lable = categories[labelId]
        trueLables.append(lable==lableItem)
        print(lable+"==="+lableItem)

        lables.append(lable)
        # **************************************************
        y_pred_labels[i] = labelId
        # **************************************************
    # 评估
   # print(lables)
    print(trueLables)
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_labels, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")

    cm = metrics.confusion_matrix(y_test_cls, y_pred_labels)
    print(cm)


