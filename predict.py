# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab
from  rnn_model import  TRNNConfig,TextRNN
try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
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
        return self.categories[y_pred_cls[0]]

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
        return self.categories[y_pred_cls[0]],pred_matrix
if __name__ == '__main__':
    with sess1.as_default():
        with g1.as_default():
              cnn_model = CnnModel()
    test_demo = ['本院经审查认为，因合同纠纷提起的诉讼，由被告住所地或者合同履行地人民法院管辖。经查，被告马静、徐建、夏慈训、折云峰、任重兮、程炳来的住所地为均不在天津市河西区。根据《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》，借贷双方就合同履行地未约定或者约定不明确，事后未达成补充协议，按照合同有关条款或者交易习惯仍不能确定的，以接受货币一方所在地为合同履行地。根据法律规定，公民的经常居住地是指公民离开住所地至起诉时已连续居住一年以上的地方，但公民住院就医的地方除外。而原告的户籍系2018年11月8日由天津市河东区沙柳北路冠云东里5号楼4门603号迁来，原告作为接受货币一方，其在天津市河西区居住不满一年，故天津市河西区不应认为是其经常居住地。故本院对该案没有管辖权，应移送至有管辖权的天津市静海区人民法院审理。依照《中华人民共和国民事诉讼法》第二十四条、第三十六条，《最高人民法院关于适用<中华人民共和国民事诉讼法>的解释》第四条、第二十一条，《最高人民法院关于审理民间借贷案件适用法律若干问题的规定》第三条之规定，裁定如下：',
                 '本院认为，涉案房屋的水、电系供水、供电单位提供，故只有供水、供电单位有权按照国家规定中止供水、供电。现被告通过代售水、电的便利条件，以停售涉案房屋水、电的方式，达到其收取涉案房屋物业费的目的，既不符合法律规定，也侵害了原告正常使用涉案房屋水、电的权利，存在过错。故此，被告应停止侵害，使得原告能够正常购买水、电，从而保证涉案房屋正常的用水、用电。关于涉案房屋欠付的物业服务费，被告应通过合法途径依法主张权利。综上所述，原告要求被告开通其所有房产的水、电使用的诉讼请求，本院判定被告向原告正常出售水、电，以保证涉案房屋正常用水、用电。依照《中华人民共和国侵权责任法》第六条、第十五条第一款第（一）项规定，判决如下：']
    for i in test_demo:

        print(cnn_model.predict(i))
    with sess2.as_default():  # 1
            with g2.as_default():
                 rnn_model = RnnModel()
    for i in test_demo:
        print(rnn_model.predict(i))
