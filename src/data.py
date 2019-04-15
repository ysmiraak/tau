#!/usr/bin/env python3

from collections import defaultdict
from tqdm import tqdm
from trial import config as C, paths as P, train as T
from util import partial
from util_io import pform, load_txt, save_txt
from util_np import np, vpack
from util_sp import spm, load_spm, decode

langs = 'en', 'nl', 'de', 'da', 'sv'

#######################
# align all 5 corpora #
#######################

# load all corpora
corp2pairs = {corp: tuple((s, t) for s, t in zip(
      map(str.strip, load_txt(pform(P.raw, "europarl-v7.{}-en.{}".format(corp, corp))))
    , map(str.strip, load_txt(pform(P.raw, "europarl-v7.{}-en.en".format(corp)))))
                          if 0 < len(s) and 0 < len(t))
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

# remove long sentences
short = []
for sents in zip(*(load_txt(pform(P.raw, lang)) for lang in langs)):
    sents = [v.encode_as_ids(s) for v, s in zip(vocab, sents)]
    if all(len(sent) <= C.cap for sent in sents):
        short.append(sents)

np.random.seed(C.seed)
np.random.shuffle(short)

# pack instances into arrays
corpora = tuple(vpack(corp, (len(corp), C.cap), C.eos, np.uint16) for corp in zip(*short))
del short

# split and save
for lang, voc, corp in zip(langs, vocab, corpora):
    save_txt(pform(P.data, "eval_{}.txt".format(lang)), decode(voc, corp[:4096]))
    np.save(pform(P.data, "valid_{}.npy".format(lang)), corp[4096:5120])
    np.save(pform(P.data, "train_{}.npy".format(lang)), corp[5120:])
