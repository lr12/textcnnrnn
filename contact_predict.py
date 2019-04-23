# coding: utf-8

from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr


from  rnn_model import  TRNNConfig,TextRNN


import os


import numpy as np
import tensorflow as tf
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab


try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
save_dir = 'checkpoints/textrnn'
save_path1 = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
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
        return cnn_model.predict(i),pred_matrix

class RnnModel:
    def __init__(self):
        self.config = TRNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextRNN(self.config)

        self.session = tf.Session()
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
        return self.categories[y_pred_cls[0]],pred_matrix

def test():
    cnn_model = CnnModel()

if __name__ == '__main__':

    cnn_model = CnnModel()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)

    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, 5000)
    rnn_model = RnnModel()
    lables=[]
    for i in range(len(x_test)):
        item = x_test[i]
        y = y_test[i]
        _,result_cnn=cnn_model.predict(item)
        _, result_rnn  = rnn_model.predict(item)
        result = result_cnn + result_rnn
        labelId = np.argmax(result)
        lable = categories[labelId]
        lables.append(lable)
    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test, lables, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")

    cm = metrics.confusion_matrix(y_test, lables)
    print(cm)



