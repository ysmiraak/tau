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

C.trial = 'm4_'

#############
# load data #
#############

# valid_en, train_en = np.load(pform(P.data, "valid_en.npy")), np.load(pform(P.data, "train_en.npy"))
valid_nl, train_nl = np.load(pform(P.data, "valid_nl.npy")), np.load(pform(P.data, "train_nl.npy"))
# valid_de, train_de = np.load(pform(P.data, "valid_de.npy")), np.load(pform(P.data, "train_de.npy"))
valid_da, train_da = np.load(pform(P.data, "valid_da.npy")), np.load(pform(P.data, "train_da.npy"))
# valid_sv, train_sv = np.load(pform(P.data, "valid_sv.npy")), np.load(pform(P.data, "train_sv.npy"))

train_nl = train_nl[:2**17].copy()
train_da = train_da[:2**17].copy()

data_index =        1,        3
data_valid = valid_nl, valid_da
data_train = train_nl, train_da

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
model.down = tf.train.AdamOptimizer(model.lr, T.beta1, T.beta2, T.epsilon) \
                     .minimize(model.loss, model.step, (model.embeds[1].logit, model.embeds[3].logit))

############
# training #
############

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
tf.train.Saver(
    [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if not ('embed' in v.name and 'Adam' in v.name)]
).restore(sess, pform(P.ckpt, 'm1_', 9))

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

for _ in range(9): # ~22.888 epoch per round
    for _ in range(100):
        for _ in tqdm(range(400), ncols= 70):
            sess.run(model.down)
        step = sess.run(model.step)
        summ(step)
    saver.save(sess, pform(P.ckpt, C.trial, step // 40000), write_meta_graph= False)
