from util import Record, identity
from util_np import np, partition
from util_tf import tf, scope, placeholder, trim, Normalize, Smooth, Dropout, Embed, Conv, SepConv, Attention


def causal_mask(t, name= 'causal_mask'):
    """returns the causal mask for `t` steps"""
    with scope(name):
        return tf.linalg.LinearOperatorLowerTriangular(tf.ones((t, t))).to_dense()


def sinusoid(dim, time, freq= 1e-4, array= False):
    """returns a rank-2 tensor of shape `dim, time`, where each column
    corresponds to a time step and each row a sinusoid, with
    frequencies in a geometric progression from 1 to `freq`.

    """
    assert not dim % 2
    if array:
        a = (freq ** ((2 / dim) * np.arange(dim // 2))).reshape(-1, 1) @ (1 + np.arange(time).reshape(1, -1))
        return np.concatenate((np.sin(a), np.cos(a)), -1).reshape(dim, time)
    else:
        assert False # figure out a better way to do this
        a = tf.reshape(
            freq ** ((2 / dim) * tf.range(dim // 2, dtype= tf.float32))
            , (-1, 1)) @ tf.reshape(
                1 + tf.range(tf.to_float(time), dtype= tf.float32)
                , (1, -1))
        return tf.reshape(tf.concat((tf.sin(a), tf.cos(a)), axis= -1), (dim, time))


class Sinusoid(Record):

    def __init__(self, dim, cap= None, name= 'sinusoid'):
        self.dim = dim
        self.name = name
        with scope(name):
            self.pos = tf.constant(sinusoid(dim, cap, array= True), tf.float32) if cap else None

    def __call__(self, time, name= None):
        with scope(name or self.name):
            return sinusoid(self.dim, time) if self.pos is None else self.pos[:,:time]


class MlpBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.lin  = Conv(4*dim, dim, name= 'lin')
            self.lex  = Conv(dim, 4*dim, name= 'lex')
            self.norm = Normalize(dim)

    def __call__(self, x, dropout, name= None):
        with scope(name or self.name):
            return self.norm(x + dropout(self.lex(tf.nn.relu(self.lin(x)))))


class AttBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.att  = Attention(dim)
            self.norm = Normalize(dim)

    def __call__(self, x, v, m, dropout, name= None):
        with scope(name or self.name):
            return self.norm(x + dropout(self.att(x, v, m)))


class BiattBlock(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.latt = Attention(dim, name= 'latt')
            self.ratt = Attention(dim, name= 'ratt')
            self.norm = Normalize(dim)

    def __call__(self, x, v, m, w, n, dropout, name= None):
        with scope(name or self.name):
            return self.norm(tf.add_n(((dropout(self.latt(x, v, m)), x, dropout(self.ratt(x, w, n))))))


class GluBlock(Record):

    def __init__(self, dim, name, mid= 128, depth= 2):
        self.name = name
        with scope(name):
            self.ante =       Conv(mid, dim, size= 1, name= 'ante')
            self.gate = tuple(Conv(mid, mid, size= 2, name= "gate{}".format(1+i)) for i in range(depth))
            self.conv = tuple(Conv(mid, mid, size= 2, name= "conv{}".format(1+i)) for i in range(depth))
            self.post =       Conv(dim, mid, size= 1, name= 'post')
            self.norm = Normalize(dim, name= 'norm')

    def __call__(self, x, dropout, name= None):
        with scope(name or self.name):
            y = self.ante(x)
            for gate, conv in zip(self.gate, self.conv):
                y = tf.pad(y, ((0,0),(0,0),(conv.shape()[0]-1,0)))
                y = tf.sigmoid(gate(y)) * conv(y)
            return self.norm(x + dropout(self.post(y)))


class SepBlock(Record):

    def __init__(self, dim, name, size= 5, depth= 2):
        self.name = name
        with scope(name):
            self.ante =          Conv(dim, dim, size=    1, name= 'ante')
            self.gate = tuple(SepConv(dim, dim, size= size, name= "gate{}".format(1+i)) for i in range(depth))
            self.conv = tuple(SepConv(dim, dim, size= size, name= "conv{}".format(1+i)) for i in range(depth))
            self.post =          Conv(dim, dim, size=    1, name= 'post')
            self.norm = Normalize(dim, name= 'norm')

    def __call__(self, x, dropout, name= None):
        with scope(name or self.name):
            y = self.ante(x)
            for gate, conv in zip(self.gate, self.conv):
                y = tf.pad(y, ((0,0),(0,0),(conv.shape()[0]-1,0)))
                y = tf.sigmoid(gate(y)) * conv(y)
            return self.norm(x + dropout(self.post(y)))


class Encode(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.blocks = AttBlock(dim, 's1') \
                ,         MlpBlock(dim, 'm1') \
                ,         AttBlock(dim, 's2') \
                ,         MlpBlock(dim, 'm2') \
                ,         AttBlock(dim, 's3') \
                ,         MlpBlock(dim, 'm3') \
                # ,         AttBlock(dim, 's4') \
                # ,         MlpBlock(dim, 'm4') \
                # ,         AttBlock(dim, 's5') \
                # ,         MlpBlock(dim, 'm5') \
                # ,         AttBlock(dim, 's6') \
                # ,         MlpBlock(dim, 'm6')

    def __call__(self, x, m, dropout, name= None):
        with scope(name or self.name):
            for block in self.blocks:
                btype = block.name[0]
                if   'c' == btype: x = block(x, dropout)
                elif 's' == btype: x = block(x, x, m, dropout)
                elif 'm' == btype: x = block(x, dropout)
                else: raise TypeError('unknown encode block')
            return x


class Decode(Record):

    def __init__(self, dim, name):
        self.name = name
        with scope(name):
            self.blocks = AttBlock(dim, 's1') \
                ,         AttBlock(dim, 'a1') \
                ,         MlpBlock(dim, 'm1') \
                ,         AttBlock(dim, 's2') \
                ,         AttBlock(dim, 'a2') \
                ,         MlpBlock(dim, 'm2') \
                ,         AttBlock(dim, 's3') \
                ,         AttBlock(dim, 'a3') \
                ,         MlpBlock(dim, 'm3') \
                # ,         AttBlock(dim, 's4') \
                # ,         AttBlock(dim, 'a4') \
                # ,         MlpBlock(dim, 'm4') \
                # ,         AttBlock(dim, 's5') \
                # ,         AttBlock(dim, 'a5') \
                # ,         MlpBlock(dim, 'm5') \
                # ,         AttBlock(dim, 's6') \
                # ,         AttBlock(dim, 'a6') \
                # ,         MlpBlock(dim, 'm6')

    def __call__(self, x, m, w, n, dropout, name= None):
        with scope(name or self.name):
            for block in self.blocks:
                btype = block.name[0]
                if   'c' == btype: x = block(x, dropout)
                elif 'b' == btype: x = block(x, x, m, w, n, dropout)
                elif 's' == btype: x = block(x, x, m, dropout)
                elif 'a' == btype: x = block(x, w, n, dropout)
                elif 'm' == btype: x = block(x, dropout)
                else: raise TypeError('unknown decode block')
            return x


class Model(Record):
    """-> Record

    model = Model.new( ... )
    train = model.data( ... ).train( ... )
    valid = model.data( ... ).valid( ... )
    infer = model.data( ... ).infer( ... )

    """
    _new = 'dim_emb', 'dim_mid', 'dim_voc', 'cap', 'eos', 'bos'

    @staticmethod
    def new(langs, dim_emb, dim_mid, dim_voc, cap, eos, bos):
        """-> Model with fields

          decode : Decode
          encode : Encode
         emb_tgt : Embed
         emb_src : Embed

        """
        assert not dim_emb % 2
        return Model(
              embeds = {lid: Embed(dim_emb, dim_voc, name= "embed_{}".format(lid)) for lid in langs}
            , decode= Decode(dim_emb, name= 'decode')
            , encode= Encode(dim_emb, name= 'encode')
            , dim_emb= dim_emb
            , dim_voc= dim_voc
            , bos= bos
            , eos= eos
            , cap= cap + 1)

    def data(self, sid, tid, src= None, tgt= None):
        """-> Model with new fields

        position : Sinusoid
            src_ : i32 (b, ?)     source feed, in range `[0, dim_src)`
            tgt_ : i32 (b, ?)     target feed, in range `[0, dim_tgt)`
             src : i32 (b, s)     source with `eos` trimmed among the batch
             tgt : i32 (b, t)     target with `eos` trimmed among the batch, padded with `bos`
            mask : b8  (b, t)     target sequence mask
            true : i32 (?,)       target references
         max_tgt : i32 ()         maximum target length
         max_src : i32 ()         maximum source length
        mask_tgt : f32 (1, t, t)  target attention mask
        mask_src : f32 (b, 1, s)  source attention mask

        """
        src_ = placeholder(tf.int32, (None, None), src, 'src_')
        tgt_ = placeholder(tf.int32, (None, None), tgt, 'tgt_')
        with scope('src'):
            src, msk, max_src = trim(src_, self.eos)
            mask_src = tf.log(tf.expand_dims(tf.to_float(msk), axis= 1))
        with scope('tgt'):
            tgt, msk, max_tgt = trim(tgt_, self.eos)
            mask = tf.pad(msk, ((0,0),(1,0)), constant_values= True)
            btru = tf.pad(tgt, ((0,0),(1,0)), constant_values= self.bos)
            true = tf.pad(tgt, ((0,0),(0,1)), constant_values= self.eos)
            true, tgt = tf.boolean_mask(true, mask), btru
            max_tgt += 1
            mask_tgt = tf.log(tf.expand_dims(causal_mask(max_tgt), axis= 0))
        return Model(
            position= Sinusoid(self.dim_emb, self.cap)
            , src_= src_, mask_src= mask_src, max_src= max_src, src= src
            , tgt_= tgt_, mask_tgt= mask_tgt, max_tgt= max_tgt, tgt= tgt
            , true= true, mask= mask
            , emb_src = self.embeds[sid]
            , emb_tgt = self.embeds[tid]
            , **self)

    def infer(self):
        """-> Model with new fields, autoregressive

        len_tgt : i32 ()      steps to unfold aka t
           pred : i32 (b, t)  prediction, hard

        """
        dropout = identity
        with scope('infer'):
            with scope('encode'):
                w = self.position(self.max_src) + self.emb_src(self.src)
                w = self.encode(w, self.mask_src, dropout) # bds
            with scope('decode'):
                cap = placeholder(tf.int32, (), self.cap)
                msk = tf.log(tf.expand_dims(causal_mask(cap), axis= 0)) # 1tt
                pos = self.position(cap) # dt
                i,q = tf.constant(0), tf.zeros_like(self.src[:,:1]) + self.bos
                def body(i, q):
                    j = i + 1
                    x = pos[:,:j] + self.emb_tgt(q) # bdj <- bj
                    x = self.decode(x, msk[:,:j,:j], w, self.mask_src, dropout) # bdj
                    p = tf.expand_dims( # b1
                        tf.argmax( # b
                            self.emb_tgt( # bn
                                tf.squeeze( # bd
                                    x[:,:,-1:] # bd1 <- bdj
                                    , axis= -1))
                            , axis= -1, output_type= tf.int32)
                        , axis= -1)
                    return j, tf.concat((q, p), axis= -1) # bk <- bj, b1
                cond = lambda i, q: ((i < cap) & ~ tf.reduce_all(tf.equal(q[:,-1], self.eos)))
                _, p = tf.while_loop(cond, body, (i, q), back_prop= False, swap_memory= True)
                pred = p[:,1:]
        return Model(self, len_tgt= cap, pred= pred)

    def valid(self, dropout= identity, smooth= None):
        """-> Model with new fields, teacher forcing

           output : f32 (?, dim_tgt)  prediction on logit scale
             prob : f32 (?, dim_tgt)  prediction, soft
             pred : i32 (?,)          prediction, hard
        errt_samp : f32 (?,)          errors
        loss_samp : f32 (?,)          losses
             errt : f32 ()            error rate
             loss : f32 ()            mean loss

        """
        with scope('emb_src_'): w = self.position(self.max_src) + dropout(self.emb_src(self.src))
        with scope('emb_tgt_'): x = self.position(self.max_tgt) + dropout(self.emb_tgt(self.tgt))
        w = self.encode(w, self.mask_src,                   dropout, name= 'encode_') # bds
        x = self.decode(x, self.mask_tgt, w, self.mask_src, dropout, name= 'decode_') # bdt
        with scope('logit_'):
            y = self.emb_tgt( # ?n
                tf.boolean_mask( # ?d
                    tf.transpose(x, (0,2,1)) # btd <- bdt
                    , self.mask))
        with scope('prob_'): prob = tf.nn.softmax(y, axis= -1)
        with scope('pred_'): pred = tf.argmax(y, axis= -1, output_type= tf.int32)
        with scope('errt_'):
            errt_samp = tf.to_float(tf.not_equal(self.true, pred))
            errt = tf.reduce_mean(errt_samp)
        with scope('loss_'):
            loss_samp = tf.nn.softmax_cross_entropy_with_logits_v2(labels= smooth(self.true), logits= y) \
                if smooth else tf.nn.sparse_softmax_cross_entropy_with_logits(labels= self.true, logits= y)
            loss = tf.reduce_mean(loss_samp)
        return Model(self, output= y, prob= prob, pred= pred
                     , errt_samp= errt_samp, errt= errt
                     , loss_samp= loss_samp, loss= loss)

    def train(self, dropout= 0.1, smooth= 0.1, warmup= 4e3, beta1= 0.9, beta2= 0.98, epsilon= 1e-9):
        """-> Model with new fields, teacher forcing

        step : i64 () global update step
          lr : f32 () learning rate for the current step
          up :        update operation

        along with all the fields from `valid`

        """
        dropout, smooth = Dropout(dropout, (None, self.dim_emb, None)), Smooth(smooth, self.dim_voc)
        self = self.valid(dropout= dropout, smooth= smooth)
        with scope('lr'):
            s = tf.train.get_or_create_global_step()
            t = tf.to_float(s + 1)
            lr = (self.dim_emb ** -0.5) * tf.minimum(t ** -0.5, t * (warmup ** -1.5))
        # up = tf.train.AdamOptimizer(lr, beta1, beta2, epsilon).minimize(self.loss, s)
        return Model(self, dropout= dropout, smooth= smooth, step= s, lr= lr)


def batch_run(sess, model, fetch, src, tgt= None, batch= None):
    if batch is None: batch = len(src)
    for i, j in partition(len(src), batch, discard= False):
        feed = {model.src_: src[i:j]}
        if tgt is not None:
            feed[model.tgt_] = tgt[i:j]
        yield sess.run(fetch, feed)
