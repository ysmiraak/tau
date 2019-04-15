#!/usr/bin/env python3

from itertools import permutations, chain, islice
from model import Model, batch_run
from tqdm import tqdm
from trial import config as C, paths as P, train as T
from util import partial, comp, select, Record
from util_io import pform, load_txt, save_txt
from util_np import np, partition, sample, batch_sample, vpack
from util_sp import load_spm, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)
np.random.seed(C.seed)

C.trial = 't5_'
C.cap = 128
C.batch_train = 128

#############
# load data #
#############

langs = 'en', 'el', 'it', 'sv', 'fi'

data_train = Record(np.load(pform(P.data, "train.npz")))
data_valid = Record(np.load(pform(P.data, "valid.npz")))

vocab = load_spm("../trial/data/alien/vocab_fi.model")
data_train = tuple(decode(vocab, data_train["fi_fi"]))

def encode_capped_sample(vocab, sent, cap, alpha_min= 0.1, alpha_max= 1.0, alpha_step= 0.1):
    alpha = alpha_min
    while alpha <= alpha_max:
        x = vocab.sample_encode_as_ids(sent, -1, alpha)
        if len(x) <= cap:
            return x
        alpha += alpha_step
    raise ValueError("cannot encode sentence under {} pieces".format(cap))

def batch(size= C.batch_train
        , seed= C.seed
        , vocab= vocab
        , eos= vocab.eos_id()
        , tgt= data_train
        , cap= C.cap):
    bas, bat = [], []
    for i in sample(len(tgt), seed):
        if size == len(bas):
            yield vpack(bas, (size, cap), eos, np.int32) \
                , vpack(bat, (size, cap), eos, np.int32)
            bas, bat = [], []
        sent = tgt[i]
        try:
            bas.append(encode_capped_sample(vocab, sent, cap))
            bat.append(encode_capped_sample(vocab, sent, cap))
        except ValueError:
            pass

data_train = pipe(batch, (tf.int32, tf.int32))
data_valid = (data_valid['fi'], data_valid['en']), (data_valid['en'], data_valid['fi'])
data_index = ('fi', 'en'), ('en', 'fi')

###############
# build model #
###############

model = Model.new(langs, **select(C, *Model._new))
valid = tuple(model.data(sid, tid).valid() for sid, tid in data_index)
train = model.data('fi', 'fi', *data_train).train(**T)

model.lr   = train.lr
model.step = train.step
model.errt = train.errt
model.loss = train.loss
model.down = tf.train.AdamOptimizer(model.lr, T.beta1, T.beta2, T.epsilon) \
                     .minimize(model.loss, model.step, (model.embeds['fi'].logit,))

############
# training #
############

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
tf.train.Saver(
    [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'embed_fi' not in v.name]
).restore(sess, pform(P.ckpt, 't1_', 16))

saver = tf.train.Saver()

model.step.assign(0).eval()

def summ(step, wtr = tf.summary.FileWriter(pform(P.log, C.trial))
         , summary = tf.summary.merge(
             ( tf.summary.scalar('step_errt', model.errt)
             , tf.summary.scalar('step_loss', model.loss)))):
    errt, loss = map(comp(np.mean, np.concatenate), zip(*chain(*(
        batch_run(sess, m, (m.errt_samp, m.loss_samp), s, t, batch= C.batch_valid)
        for m, (s, t) in zip(valid, data_valid)))))
    wtr.add_summary(sess.run(summary, {model.errt: errt, model.loss: loss}), step)
    wtr.flush()

for _ in range(9): # ~7.13 epochs per round
    for _ in range(250):
        for _ in tqdm(range(400), ncols= 70):
            sess.run(model.down)
        step = sess.run(model.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
