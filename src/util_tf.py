from util import Record
import tensorflow as tf


scope = tf.variable_scope


def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def init_or_restore(sess, ckpt, verbose= True):
    """restores saved variables from `ckpt` and initializes the rest."""
    names = {name for name, _ in tf.train.list_variables(ckpt)}
    saved, extra = [], []
    for var in tf.global_variables():
        if var.name[:-2] in names:
            saved.append(var)
        else:
            extra.append(var)
    tf.train.Saver(saved).restore(sess, ckpt)
    if verbose: print("restored {} variables from {}".format(len(saved), ckpt))
    sess.run(tf.variables_initializer(extra))
    if verbose: print("initialized {} variables".format(len(extra)))


def pipe(gen_func, gen_types, map_func= None, map_types= None, para_map= 4, prefetch= 4, name= 'pipe'):
    """returns iterator tensors of `gen_types` from generator `gen_func`.
    see `tf.data.Dataset.from_generator`.

    when specified, `map_func` is called on the generator outputs (as
    numpy arrays) and tensors of `map_types` are returned instead.
    `para_map` number of calls are processed in parallel.  `map_func`
    must be stateless.  otherwise simply transform the data in
    `gen_func`.  it should be used only for parallelizing heavy
    transformations.  see `tf.data.Dataset.map` and `tf.py_func`.

    """
    with scope(name):
        ds = tf.data.Dataset.from_generator(gen_func, gen_types)
        if map_func is not None:
            ds = ds.map(
                lambda *args: tf.py_func(map_func, args, map_types, stateful= False)
                , num_parallel_calls= para_map)
        return ds.prefetch(prefetch).make_one_shot_iterator().get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`

    if tensor `x` is given, converts and uses it as default

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def trim(x, eos, name= 'trim'):
    """trims a tensor of sequences

    x   : tensor i32 (b, ?)
    eos : tensor i32 ()
       -> tensor i32 (b, t)  the trimmed sequence tensor
        , tensor b8  (b, t)  the sequence mask
        , tensor i32 ()      the maximum non-eos sequence length t

    each row aka sequence in `x` is assumed to be any number of
    non-eos followed by any number of eos

    """
    with scope(name):
        with scope('not_eos'): not_eos = tf.not_equal(x, eos)
        with scope('len_seq'): len_seq = tf.reduce_sum(tf.to_int32(not_eos), axis= 1)
        with scope('max_len'): max_len = tf.reduce_max(len_seq)
        return x[:,:max_len], not_eos[:,:max_len], max_len


def get_shape(x, name= 'shape'):
    """returns the shape of `x` as a tuple of integers (static) or int32
    scalar tensors (dynamic)

    """
    with scope(name):
        shape = tf.shape(x)
        shape = tuple(d if d is not None else shape[i] for i, d in enumerate(x.shape.as_list()))
        return shape


def variable(name, shape, init= 'rand', initializers=
             {  'zero': tf.initializers.zeros()
              , 'unit': tf.initializers.ones()
              , 'rand': tf.glorot_uniform_initializer()
             }):
    """wraps `tf.get_variable` to provide initializer based on usage"""
    return tf.get_variable(name, shape, initializer= initializers.get(init, init))


class Normalize(Record):
    """layer normalization"""

    def __init__(self, dim, name= 'normalize'):
        self.name = name
        with scope(name):
            self.gain = variable('gain', (1, dim, 1), init= 'unit')
            self.bias = variable('bias', (1, dim, 1), init= 'zero')

    def __call__(self, x, name= None):
        with scope(name or self.name):
            mean, var = tf.nn.moments(x, 1, keep_dims= True)
            return (x - mean) * tf.rsqrt(var + 1e-12) * self.gain + self.bias


class Smooth(Record):
    """binary smoothing if dim is None or channel-last one-hot smoothing"""

    def __init__(self, rate, dim= None, name= 'smooth'):
        self.dim = dim
        self.name = name
        with scope(name):
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.shared = self.rate / (dim or 2)
            self.smooth = 1.0 - self.rate

    def __call__(self, x, name= None):
        with scope(name or self.name):
            if self.dim:
                return tf.one_hot(x, self.dim, self.smooth + self.shared, self.shared)
            else:
                return x * self.smooth + self.shared


class Dropout(Record):
    """dropout shape may contain None (to be dynamically filled) or 1 (to
    be broadcasted) or some fixed dimension, such as `(None, 256, 1)`

    """

    def __init__(self, rate, shape= None, name= 'dropout'):
        self.shape = shape
        self.name = name
        with scope(name):
            self.rate = placeholder(tf.float32, (), rate, 'rate')
            self.keep = 1.0 - self.rate

    def __call__(self, x, name= None):
        with scope(name or self.name):
            shape = get_shape(x)
            if self.shape is not None:
                shape = [d or shape[i] for i, d in enumerate(self.shape)]
            return tf.nn.dropout(x, self.keep, shape)


class Embed(Record):
    """input and output embedding

    tensor i32 (b, t) -> tensor f32 (b, n, t)
    tensor f32 (?, n) -> tensor f32 (?, m)

    """

    def __init__(self, n, m, name= 'embed'):
        self.name = name
        with scope(name):
            self.logit = variable('kern', (n, m))
            self.embed = tf.transpose(self.logit) * (n ** 0.5)

    def __call__(self, x, name= None):
        assert 2 == len(x.shape)
        with scope(name or self.name):
            if x.dtype.is_integer:
                return tf.transpose(tf.gather(self.embed, x), (0, 2, 1))
            else:
                return x @ self.logit


class Conv(Record):
    """convolution from `m` to `n` channels

    the default parameters make a position-wise linear layer

    """

    def __init__(self, n, m= None, size= 1, name= 'conv'):
        if m is None: m = n
        self.name = name
        with scope(name):
            self.kern = variable('kern', (size, m, n))

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return tf.nn.convolution(x, self.kern, padding= 'VALID', data_format= 'NCW')

    def shape(self):
        return get_shape(self.kern)


class SepConv(Record):
    """separable convolution from `m` to `n` channels"""

    def __init__(self, n, m= None, size= 2, name= 'conv'):
        if m is None: m = n
        self.name = name
        with scope(name):
            self.kern_depthwise = variable('kern_depthwise', (1, size, m, 1))
            self.kern_pointwise = variable('kern_pointwise', (1,    1, m, n))

    def __call__(self, x, name= None):
        with scope(name or self.name):
            return tf.squeeze( # bdt
                tf.nn.separable_conv2d(
                    tf.expand_dims(x, axis= 2) # bd1t
                    , depthwise_filter= self.kern_depthwise  # 1sm1
                    , pointwise_filter= self.kern_pointwise  # 11mn
                    , strides= (1, 1, 1, 1)
                    , padding= 'VALID'
                    , data_format= 'NCHW')
                , axis= 2)

    def shape(self):
        _, s, _, _ = get_shape(self.kern_depthwise)
        _, _, m, n = get_shape(self.kern_pointwise)
        return (s, m, n)


class Attention(Record):
    """computes multi-head scaled dot-product attention

    query : tensor f32 (b, d_q, t)
    value : tensor f32 (b, d_v, s)
     mask : tensor f32 (b,   t, s)
         -> tensor f32 (b, d_q, t)

    `dim` must be divisible by `head`

    `mask` has on-values 0 and off-values -inf

    """

    def __init__(self, dim, d_q= None, d_v= None, head= 8, name= 'attention'):
        assert not dim % head
        if d_q is None: d_q = dim
        if d_v is None: d_v = dim
        self.dim = dim
        self.head = head
        self.name = name
        with scope(name):
            self.v = Conv(dim, d_v, name= 'v')
            self.k = Conv(dim, d_v, name= 'k')
            self.q = Conv(dim, d_q, name= 'q')
            self.p = Conv(d_q, dim, name= 'p')

    def __call__(self, query, value, mask= None, name= None):
        with scope(name or self.name):
            d,h,c = self.dim, self.head, self.dim // self.head
            b,_,t = get_shape(query)
            b,_,s = get_shape(value)
            # pretransformations
            v = tf.reshape(self.v(value), (b,h,c,s)) # bhcs <- bds <- bvs
            k = tf.reshape(self.k(value), (b,h,c,s)) # bhcs <- bds <- bvs
            q = tf.reshape(self.q(query), (b,h,c,t)) # bhct <- bdt <- bqt
            # weight
            a = tf.matmul(q, k, transpose_a= True) # bhts <- (bhtc <- bhct) @ bhcs
            a *= c ** -0.5
            if mask is not None: a += tf.expand_dims(mask, axis= 1)
            a = tf.nn.softmax(a, axis= -1)
            # attend
            y = tf.matmul(v, a, transpose_b= True) # bhct <- bhcs @ (bhst <- bhts)
            # posttransformation
            return self.p(tf.reshape(y, (b,d,t))) # bqt <- bdt <- bhct
