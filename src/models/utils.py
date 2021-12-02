
# import tensorflow as tf
import tensorflow.compat.v2 as tf
# from models.network import ResnetPeriodEstimator

def get_sims(embs, temperature):
  """Calculates self-similarity between batch of sequence of embeddings."""
  batch_size = tf.shape(embs)[0]
  seq_len = tf.shape(embs)[1]
  embs = tf.reshape(embs, [batch_size, seq_len, -1])

  def _get_sims(embs):
    """Calculates self-similarity between sequence of embeddings."""
    dist = pairwise_l2_distance(embs, embs)
    sims = -1.0 * dist
    return sims

  sims = tf.map_fn(_get_sims, embs)
  sims /= temperature
  sims = tf.nn.softmax(sims, axis=-1)
  sims = tf.expand_dims(sims, -1)
  return sims


def pairwise_l2_distance(a, b):
  """Computes pairwise distances between all rows of a and all rows of b."""
  norm_a = tf.reduce_sum(tf.square(a), 1)
  norm_a = tf.reshape(norm_a, [-1, 1])
  norm_b = tf.reduce_sum(tf.square(b), 1)
  norm_b = tf.reshape(norm_b, [1, -1])
  dist = tf.maximum(norm_a - 2.0 * tf.matmul(a, b, False, True) + norm_b, 0.0)
  return dist


def flatten_sequential_feats(x, batch_size, seq_len):
  """Flattens sequential features with known batch size and seq_len."""
  x = tf.reshape(x, [batch_size, seq_len, -1])
  return x

# Transformer from https://www.tensorflow.org/tutorials/text/transformer .
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.

  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    outputs: shape == (..., seq_len_q, depth_v)
    attention_weights: shape == (..., seq_len_q, seq_len_k)
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk.
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  # (..., seq_len_q, seq_len_k)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  outputs = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return outputs, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])



