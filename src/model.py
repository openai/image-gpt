import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def gelu2(x):
    return x * tf.sigmoid(1.702 * x)

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[axis].value
        #x = x - tf.reduce_mean(x,axis=axis,keepdims=True)#normalize to zero mean
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        s = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
        x = x * tf.rsqrt(s + epsilon)
        x = x*g
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])), start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams,debug=None):
    if debug:
        debug['ln_1'].append(x)
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        if not hparams.bert:
            w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        *start, nx = shape_list(x)

        wk = tf.get_variable("k_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        wq = tf.get_variable("q_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        wv = tf.get_variable("v_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state)))
        k = tf.einsum("bsf,hef->bhse", x, wk)
        q = tf.einsum("bsf,hef->bhse", x, wq)
        v = tf.einsum("bsf,hef->bhse", x, wv)

        if debug:
            debug["query"].append(q)
            debug["key"].append(k)
            debug["value"].append(v)

        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        wc = tf.get_variable("c_proj", [hparams.n_head, nx // hparams.n_head, n_state], initializer=tf.random_normal_initializer(stddev=1.0/np.sqrt(n_state*hparams.n_layer)))
        a = tf.einsum("bhse,hef->bsf", a, wc)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu2(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams,debug=None):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams,debug=debug)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, Y=None, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        if hparams.bert:
            M = tf.greater(tf.random.uniform([batch, sequence]), hparams.bert_mask_prob)
            M = tf.cast(M, tf.float32)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        wtet = tf.get_variable('wtet', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.0))
        past_length = 0 if past is None else tf.shape(past)[-2]

        results['debug']={}

        results['debug']['wte'] = wte
        results['debug']['wpe'] = wpe
        results['debug']['wtet'] = wtet

        h = tf.gather(wte, X)

        h_before_sos = tf.identity(h)
        results['debug']['h_before_sos'] = h_before_sos
        if hparams.bert:
            h = h * tf.expand_dims(M, 2)
        else:
            sos = tf.get_variable('sos', [hparams.n_embd],
                                 initializer=tf.random_normal_initializer(stddev=0.02))
            sos_tok = tf.ones([batch, 1, hparams.n_embd], dtype=tf.float32) * sos
            h = tf.concat([sos_tok, h[:,:-1,:]], axis=1)

        h_after_sos = tf.identity(h)
        results['debug']['h_after_sos'] = h_after_sos

        results['debug']['positions'] = positions_for(X, past_length)
        results['debug']['wpe_to_add'] = tf.gather(wpe, positions_for(X, past_length))

        h += tf.gather(wpe, positions_for(X, past_length))
        h_after_wpe = tf.identity(h)
      
        results['debug']['h_after_wpe'] = h_after_wpe 
        results['debug']['ln_1']=[]
        results['debug']['query']=[]
        results['debug']['key']=[]
        results['debug']['value']=[]
        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams,debug=None)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Generative loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        gen_logits = tf.matmul(h_flat, wtet, transpose_b=True)
        gen_logits = tf.reshape(gen_logits, [batch, sequence, hparams.n_vocab])

        results['gen_logits'] = gen_logits
        results['debug']['gen_logits'] = gen_logits

        gen_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=gen_logits, labels=X)
        if hparams.bert:
            IM = 1.0 - M
            gen_losses = tf.reduce_sum(gen_losses * IM, axis=1) / tf.reduce_sum(IM, axis=1)
            results['gen_loss'] = tf.reduce_mean(gen_losses)
        else:
            results['gen_loss'] = tf.reduce_mean(gen_losses)

        # Classification loss.
        with tf.variable_scope('clf', reuse=reuse):
            classes = shape_list(Y)[1]
            if hparams.clf:
                wclf = tf.get_variable('wclf', [classes, hparams.n_embd],
                                      initializer=tf.random_normal_initializer(stddev=0.0))
            else:
                wclf = tf.zeros([classes, hparams.n_embd], dtype=tf.float32)

        h = tf.reduce_mean(h, axis=1)  # average pool over sequence
        clf_logits = tf.matmul(h, wclf, transpose_b=True)
        clf_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=clf_logits, labels=Y)
        results['clf_loss'] = tf.reduce_mean(clf_losses)

        correct = tf.equal(tf.argmax(clf_logits, -1), tf.argmax(Y, -1))
        results['accuracy'] = tf.reduce_mean(tf.cast(correct, tf.float32)) * 100.0

        return results
