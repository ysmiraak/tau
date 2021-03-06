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

C.trial = 'm1_'

#############
# load data #
#############

valid_en, train_en = np.load(pform(P.data, "valid_en.npy")), np.load(pform(P.data, "train_en.npy"))
# valid_nl, train_nl = np.load(pform(P.data, "valid_nl.npy")), np.load(pform(P.data, "train_nl.npy"))
valid_de, train_de = np.load(pform(P.data, "valid_de.npy")), np.load(pform(P.data, "train_de.npy"))
# valid_da, train_da = np.load(pform(P.data, "valid_da.npy")), np.load(pform(P.data, "train_da.npy"))
valid_sv, train_sv = np.load(pform(P.data, "valid_sv.npy")), np.load(pform(P.data, "train_sv.npy"))

data_index =        0,        2,        4
data_valid = valid_en, valid_de, valid_sv
data_train = train_en, train_de, train_sv

def batch(arrs, size= C.batch_train, seed= C.seed):
    size //= len(arrs) * (len(arrs) - 1)
    for i in batch_sample(len(arrs[0]), size, seed):
        yield tuple(arr[i] for arr in arrs)

perm = comp(tuple, partial(permutations, r= 2))
data_index = perm(data_index)
data_valid = perm(data_valid)
data_train = perm(pipe(partial(batch, data_train), (tf.int32,)*len(data_train), prefetch= 16))

###############
# build model #
###############

model = Model.new(**select(C, *Model._new))
valid = tuple(model.data(i, j).valid() for i, j in data_index)
train = tuple(model.data(i, j, s, t).train(**T) for (i, j), (s, t) in zip(data_index, data_train))

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

for _ in range(9): # ~4 epoch per round
    for _ in range(250):
        for _ in tqdm(range(400), ncols= 70):
            sess.run(model.down)
        step = sess.run(model.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 100000), write_meta_graph= False)
