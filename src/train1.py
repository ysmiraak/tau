#!/usr/bin/env python3

from itertools import permutations, chain, islice
from model import Model, batch_run
from tqdm import tqdm
from trial import config as C, paths as P, train as T
from util import partial, comp, select, Record
from util_io import pform, load_txt, save_txt
from util_np import np, partition, sample, batch_sample
from util_sp import load_spm, encode, decode
from util_tf import tf, pipe
tf.set_random_seed(C.seed)
np.random.seed(C.seed)

C.trial = 't1_'

#############
# load data #
#############

langs = 'en', 'el', 'it', 'sv'

data_train = Record(np.load(pform(P.data, "train.npz")))
data_valid = Record(np.load(pform(P.data, "valid.npz")))

def batch(size= C.batch_train // 2
        , srcs= tuple(data_train["{}_{}".format(lang, lang)] for lang in langs[1:])
        , tgts= tuple(data_train["{}_{}".format(lang, 'en')] for lang in langs[1:])
        , seed= C.seed):
    corps = np.arange(len(tgts))
    sizes = np.array(list(map(len, tgts)))
    props = sizes / sizes.sum()
    samps = tuple(sample(size, seed) for size in sizes)
    while True:
        c, freqs = np.unique(np.random.choice(corps, size, p= props), return_counts= True)
        if np.array_equal(corps, c):
            yield tuple(
                (src[bat], tgt[bat])
                for src, tgt, bat in zip(srcs, tgts, (
                        np.fromiter(islice(samp, freq), np.int, freq)
                        for samp, freq in zip(samps, map(int, freqs)))))

double = lambda pairs: tuple(chain(*(((x, y), (y, x)) for x, y in pairs)))
data_index = double((lang, 'en') for lang in langs[1:])
data_train = double(pipe(batch, ((tf.int32,tf.int32),)*len(langs[1:])))
data_valid = tuple((data_valid[sid], data_valid[tid]) for sid, tid in data_index)

###############
# build model #
###############

model = Model.new(langs, **select(C, *Model._new))
valid = tuple(model.data(sid, tid).valid() for sid, tid in data_index)
train = tuple(model.data(sid, tid, src, tgt).train(**T) for (sid, tid), (src, tgt) in zip(data_index, data_train))

model.lr   = train[0].lr
model.step = train[0].step
model.errt = train[0].errt
model.loss = tf.add_n([t.loss for t in train])
model.down = tf.train.AdamOptimizer(model.lr, T.beta1, T.beta2, T.epsilon).minimize(model.loss, model.step)

############
# training #
############

sess = tf.InteractiveSession()
saver = tf.train.Saver()
if C.ckpt:
    saver.restore(sess, pform(P.ckpt, C.trial, C.ckpt))
else:
    tf.global_variables_initializer().run()

def summ(step, wtr = tf.summary.FileWriter(pform(P.log, C.trial))
         , summary = tf.summary.merge(
             ( tf.summary.scalar('step_errt', model.errt)
             , tf.summary.scalar('step_loss', model.loss)))):
    errt, loss = map(comp(np.mean, np.concatenate), zip(*chain(*(
        batch_run(sess, m, (m.errt_samp, m.loss_samp), s, t, batch= C.batch_valid)
        for m, (s, t) in zip(valid, data_valid)))))
    wtr.add_summary(sess.run(summary, {model.errt: errt, model.loss: loss}), step)
    wtr.flush()

for _ in range(16): # 3.29 epochs per round
    for _ in range(250):
        for _ in tqdm(range(400), ncols= 70):
            sess.run(model.down)
        step = sess.run(model.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
