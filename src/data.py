#!/usr/bin/env python3

from collections import defaultdict
from tqdm import tqdm
from trial import config as C, paths as P, train as T
from util import partial
from util_io import pform, load_txt, save_txt
from util_np import np, vpack
from util_sp import spm, load_spm, encode

langs = 'en', 'el', 'it', 'sv', 'fi'

#######################
# align all 5 corpora #
#######################

def select(src, tgt):
    for s, t in zip(src, tgt):
        s = s.strip()
        t = t.strip()
        if 3 <= len(s) and 3 <= len(t):
            # sentences shorter than 3 characters are weird and useless
            yield s, t

# load all corpora
corp2pairs = {corp: tuple(select(
      load_txt(pform(P.raw, "europarl-v7.{}-en.{}".format(corp, corp)))
    , load_txt(pform(P.raw, "europarl-v7.{}-en.en".format(corp)))))
              for corp in langs[1:]}

# partition into equivalence classes
sent2class = defaultdict(set)
for corp, pairs in corp2pairs.items():
    for s, t in tqdm(pairs, ncols= 70):
        s = s, corp
        t = t, 'en'
        c = set.union(sent2class[s], sent2class[t])
        c.add(s)
        c.add(t)
        for s in c:
            sent2class[s] = c
del corp2pairs

# extract classes which cover all languages uniquely
classes = {id(cls): cls for cls in sent2class.values()}
del sent2class
aligned = []
for sent_lang in tqdm(classes.values(), ncols= 70):
    lang2sents = defaultdict(list)
    for sent, lang in sent_lang:
        lang2sents[lang].append(sent)
    if len(langs) == len(lang2sents) and all(1 == len(sents) for sents in lang2sents.values()):
        aligned.append(tuple(lang2sents[lang][0] for lang in langs))
aligned.sort()
del classes

# save aligned corpora
for lang, sents in zip(langs, zip(*aligned)):
    save_txt(pform(P.raw, lang), sents)
del aligned

##################
# prep and split #
##################

# train one sentencepiece model for each language
vocab = tuple(spm(pform(P.data, "vocab_{}".format(lang)), pform(P.raw, lang), C.dim_voc, C.bos, C.eos, C.unk) for lang in langs)

# remove long sentences from aligned
short = []
for sents in zip(*(load_txt(pform(P.raw, lang)) for lang in langs)):
    if all(len(v.encode_as_ids(s)) <= C.cap for v, s in zip(vocab, sents)):
        short.append(sents)

np.random.seed(C.seed)
np.random.shuffle(short)

# split evaluation and validation instances
evals = tuple(short[:4096])
valid = tuple(short[4096:5120])
del short

# save evaluation data
for lang, sents in zip(langs, zip(*evals)):
    save_txt(pform(P.data, "eval_{}.txt".format(lang)), sents)

# convert and save validation data
np.savez_compressed(pform(P.data, "valid.npz"), **{
    lang: encode(voc, sents, dtype= np.int16) for lang, voc, sents in zip(langs, vocab, zip(*valid))})

# filter and convert training data
lang2evals = dict(zip(langs, map(set, zip(*(evals + valid)))))
lang2vocab = dict(zip(langs, vocab))
corp2train = {}
for corp in langs[1:]:
    print(corp)
    eset = lang2evals[corp]
    voc0 = lang2vocab[corp]
    voc1 = lang2vocab['en']
    src, tgt = [], []
    for s, t in select(
              load_txt(pform(P.raw, "europarl-v7.{}-en.{}".format(corp, corp)))
            , load_txt(pform(P.raw, "europarl-v7.{}-en.en".format(corp)))):
        if s not in eset:
            s = voc0.encode_as_ids(s)
            t = voc1.encode_as_ids(t)
            if max(len(s), len(t)) <= C.cap:
                src.append(s)
                tgt.append(t)
    corp2train["{}_{}".format(corp, corp)] = vpack(src, (len(src), max(map(len, src))), C.eos, np.uint16)
    corp2train["{}_{}".format(corp, 'en')] = vpack(tgt, (len(tgt), max(map(len, tgt))), C.eos, np.uint16)
    del tgt, src, voc1, voc0, eset
del lang2vocab, lang2evals

# save training data
np.savez_compressed(pform(P.data, "train.npz"), **corp2train)
del corp2train
