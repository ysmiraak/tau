#!/usr/bin/env python3

from itertools import permutations
from model import Model, batch_run
from trial import config as C, paths as P, train as T
from util import partial, comp, select
from util_io import pform, load_txt, save_txt
from util_np import np, partition, batch_sample
from util_sp import load_spm, encode, decode
from util_tf import tf, init_or_restore
tf.set_random_seed(C.seed)

C.trial = "t3_"
C.ckpt = 3

langs = 'en', 'el', 'it', 'sv', 'fi'
vocab = tuple(load_spm(pform(P.data, "vocab_{}.model".format(lang))) for lang in langs)
sents = tuple(encode(voc, load_txt(pform(P.data, "eval_{}.txt".format(lang)))) for lang, voc in zip(langs, vocab))

index = tuple(permutations(range(len(langs)), 2))
model = Model.new(langs, **select(C, *Model._new))
model = tuple(model.data(langs[i], langs[j]).infer() for i, j in index)

sess = tf.InteractiveSession()

def trans(sents, model, vocab):
    for preds in batch_run(sess, model, model.pred, sents, batch= C.batch_infer):
        yield from decode(vocab, preds)

init_or_restore(sess, pform(P.ckpt, C.trial, C.ckpt))
for (i, j), m in zip(index, model):
    print(langs[j], "<-", langs[i])
    save_txt(pform(P.pred, C.trial, langs[j], "_", langs[i]), trans(sents[i], m, vocab[j]))
