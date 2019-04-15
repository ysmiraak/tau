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

C.trial = 't2_'

#############
# load data #
#############

langs = 'en', 'el', 'it', 'sv', 'fi'

data_train = Record(np.load(pform(P.data, "train.npz")))
data_valid = Record(np.load(pform(P.data, "valid.npz")))

def batch(size= C.batch_train // 2
        , src= data_train["fi_fi"]
        , tgt= data_train["fi_en"]
        , seed= C.seed):
    for bat in batch_sample(len(tgt), size, seed):
        yield src[bat], tgt[bat]

double = lambda pair: (pair, pair[::-1])
data_index = double(('fi', 'en'))
data_train = double(pipe(batch, (tf.int32,tf.int32)))
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

for _ in range(6): # 8.36 epochs per round
    for _ in range(250):
        for _ in tqdm(range(400), ncols= 70):
            sess.run(model.down)
        step = sess.run(model.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
