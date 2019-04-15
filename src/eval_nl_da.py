#!/usr/bin/env python3

from itertools import permutations, chain
from model import Model, batch_run
from tqdm import tqdm
from trial import config as C, paths as P, train as T
from util import partial, comp, select
from util_io import pform, load_txt, save_txt
from util_np import np, partition, batch_sample
from util_sp import load_spm, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)

C.trial = "m4_"

langs = 'en', 'nl', 'de', 'da', 'sv'
vocab = tuple(load_spm(pform(P.data, "vocab_{}.model".format(lang))) for lang in langs)
sents = tuple(encode(voc, load_txt(pform(P.data, "eval_{}.txt".format(lang)))) for lang, voc in zip(langs, vocab))

index = (1, 3), (3, 1)
model = Model.new(**select(C, *Model._new))
model = tuple(model.data(i, j).infer() for i, j in index)

sess = tf.InteractiveSession()
saver = tf.train.Saver()

def trans(sents, model, vocab):
    for preds in batch_run(sess, model, model.pred, sents, batch= C.batch_infer):
        yield from decode(vocab, preds)

for ckpt in 5, 6, 7, 8, 9:
    saver.restore(sess, pform(P.ckpt, C.trial, ckpt))
    for (i, j), m in zip(index, model):
        print(langs[j], "<-", langs[i])
        save_txt(pform(P.pred, C.trial, ckpt, "_", langs[j], "_", langs[i]), trans(sents[i], m, vocab[j]))
