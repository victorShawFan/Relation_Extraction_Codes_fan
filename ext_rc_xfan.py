# _*_ coding: utf-8 _*_
'''这里要完成的是一个多标签文本分类的任务，称其为rc'''
import json
import os
import random
import re

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
from tqdm import tqdm

import bert_tools as bt
import ljqpy
from utils import *
import psutil

dname = 'chengfei_test'
datadir = 'D:\kwCodes\REcodes\RC_EE_keras\datasets\chengfei'
trains = ljqpy.LoadJsons(os.path.join(datadir, 'train.json'))  # 用path.join报错，是系统不同问题？
valids = ljqpy.LoadJsons(os.path.join(datadir, 'dev.json'))
tests = ljqpy.LoadJsons(os.path.join(datadir, 'test.json'))
# trains = ljqpy.LoadJsons(datadir + '/train.json')
# valids = ljqpy.LoadJsons(datadir + '/dev.json')
# tests = ljqpy.LoadJsons(datadir + '/dev.json')

if not os.path.isdir(dname): os.makedirs(dname)


def wdir(x): return os.path.join(dname, x)


cpu_kernel_nums = psutil.cpu_count()
rels = ljqpy.TokenList(wdir('rels.txt'), 1, trains, lambda z: [x['label'] for x in z['spo_list']])
print('rels_num:', rels.get_num())

maxlen = 256
keras_bert_path = "./chinese_roberta_wwm_ext.h5"
bert_lock_layers = 5


def NormalizeText(text):
    return re.sub('[\u3000\uac00-\ud7ff]+', ' ', text)


class RCModel:
    def __init__(self):
        print("⭐⭐⭐RC_Model_inited...")
        print('⭐Current file path : ', os.getcwd())
        print("⭐Current used pretrained model : ", keras_bert_path)
        print("⭐BERT locked layers :", bert_lock_layers)
        # 模型结构
        self.bert = load_model(keras_bert_path)
        self.bbert = bt.get_dummy_seg_model(self.bert)
        xx = Lambda(lambda x: x[:, 0])(self.bbert.output)
        pos = Dense(rels.get_num(), activation='sigmoid')(xx)
        self.model = tf.keras.models.Model(inputs=self.bbert.input, outputs=pos)
        bt.lock_transformer_layers(self.bert, bert_lock_layers)
        self.model_ready = False

    def gen_golden_y(self, datas):
        for dd in datas:
            dd['rc_obj'] = list(set(x['label'] for x in dd.get('spo_list', [])))

    def make_model_data(self, datas, aug=False):
        self.gen_golden_y(datas)
        for dd in tqdm(datas, desc='tokenize'):
            text = dd['text'] = NormalizeText(dd['text'])
            if aug: text = ''.join([x for x in text if random.random() > 0.1])
            tokens = bt.tokenizer.tokenize(text, maxlen=maxlen)
            dd['tokens'] = tokens
        N = len(datas)
        # X = [np.zeros((N, maxlen), dtype='int32'), np.zeros((N, maxlen), dtype='int32')]
        X = [np.zeros((N, maxlen), dtype='int32')]
        Y = np.zeros((N, rels.get_num()))
        for i, dd in enumerate(tqdm(datas, desc='gen XY', total=N)):
            tokens = dd['tokens']
            X[0][i][:len(tokens)] = bt.tokenizer.tokens_to_ids(tokens)
            for x in dd['rc_obj']: Y[i][rels.get_id(x)] = 1
        return X, Y

    def load_model(self):
        self.model.load_weights(wdir('rc.h5'))
        self.model.compile('adam', neg_log_mean_loss, metrics=['accuracy'])
        self.model_ready = True

    def train(self, datas, batch_size=8, epochs=10, loadold=False):
        if loadold: self.model.load_weights(wdir('rc.h5'))
        self.X, self.Y = self.make_model_data(datas, aug=True)
        self.optimizer = bt.get_suggested_optimizer(5e-5, len(datas) * epochs // batch_size)
        self.model.compile(self.optimizer, neg_log_mean_loss, metrics=['accuracy'])
        self.cb_mcb = ModelCheckpoint(wdir('rc.h5'), save_weights_only=True)
        self.model.fit(self.X, self.Y, batch_size, epochs=epochs, shuffle=True,
                       validation_split=0.01, callbacks=[self.cb_mcb], use_multiprocessing=True,
                       workers=cpu_kernel_nums)
        self.model_ready = True

    def get_output(self, datas, pred, threshold=0.5):
        for dd, pp in zip(datas, pred):
            # for i, sc in enumerate(pp):
            #	if sc > 0.1: print(rels.get_token(i), sc)
            dd['rc_pred'] = list(rels.get_token(i) for i, sc in enumerate(pp) if sc > threshold)
            dd['rc_pred_score'] = {rels.get_token(i): float(sc) for i, sc in enumerate(pp)}

    def pretty_output(self, datas, ofile):
        with open(wdir(ofile), 'w', encoding='utf-8') as fout:
            for dd in datas:
                golden = set(dd['rc_obj'])
                predict = set(dd['rc_pred'])
                fout.write('\n' + dd['text'] + '\n')
                fout.write(str(dd['tokens']) + '\n')
                for x in predict & golden:
                    fout.write(f'o {x}:{dd["rc_pred_score"][x]:.4f}\n')
                for x in predict - golden:
                    fout.write(f'- {x}:{dd["rc_pred_score"][x]:.4f}\n')
                for x in golden - predict:
                    fout.write(f'+ {x}:{dd["rc_pred_score"][x]:.4f}\n')

    def evaluate(self, datas):
        ccnt, gcnt, ecnt = 0, 0, 0
        for dd in datas:
            plabels = set(dd['rc_pred'])
            ecnt += len(plabels)
            gcnt += len(set(dd['rc_obj']))
            ccnt += len(plabels & set(dd['rc_obj']))
        return ljqpy.CalcF1(ccnt, ecnt, gcnt)

    def predict(self, datas, threshold=0.5, ofile=None, evaluate=True):
        if not self.model_ready: self.load_model()
        self.vX, self.vY = self.make_model_data(datas)
        pred = self.model.predict(self.vX, batch_size=8, verbose=1)
        self.get_output(datas, pred, threshold)
        if ofile is not None:
            ljqpy.SaveList(map(lambda x: json.dumps(x, ensure_ascii=False), datas), wdir(ofile))
        if evaluate:
            f1str = self.evaluate(datas)
            print(f1str)
            return f1str


if __name__ == '__main__':
    when = 0
    # rc = None
    # if 'trainrc' in sys.argv:
    rc = RCModel()
    rc.train(trains, batch_size=2, epochs=1, loadold=True)
    # if 'testrc' in sys.argv:
    if rc == None: rc = RCModel()
    rc.predict(valids, threshold=0.1, ofile=f'valid_rc_{when}.json')
    valids = ljqpy.LoadJsons(wdir(f'valid_rc_{when}.json'))
    rc.pretty_output(valids, 'valid_rc_pretty.txt')
    rc.predict(tests, threshold=0.1, ofile=f'test_rc_{when}.json')
    tests = ljqpy.LoadJsons(wdir(f'test_rc_{when}.json'))
    rc.pretty_output(tests, 'test_rc_pretty.txt')
    print('done')
