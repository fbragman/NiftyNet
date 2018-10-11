import tensorflow as tf


def _add_to_vec(c_rounded, value_interest, C):
    '''
    Given a naive rounded vector, add or substract 1 based on value of interest to make sure
    sum of res == C
    :param c_rounded:
    :param value_interest:
    :param C:
    :return:
    '''

    mask = tf.not_equal(c_rounded, value_interest)
    zero = tf.ones(tf.shape(c_rounded))
    res = tf.multiply(zero, tf.cast(mask, zero.dtype))
    res = tf.abs(tf.ones(tf.shape(c_rounded)) - res)

    res = tf.cond(tf.reduce_sum(c_rounded) < C, lambda: tf.add(c_rounded, res), lambda: tf.add(c_rounded, -res))

    return res


def _get_value_to_change(raw, new):

    rounding_delta = tf.abs(new - raw)
    max_rounding = tf.argmax(rounding_delta, axis=1)
    num_examples = tf.cast(tf.shape(new)[0], dtype=max_rounding.dtype)
    idx = tf.stack([tf.range(num_examples), max_rounding], axis=-1)
    max_delta = tf.gather_nd(new, idx)

    return max_delta


def smartround(input_tensor, C):
    '''
    Given an input vector that determines how to split up a matrix of depth C,
    calculate the depth of each group by smart rounding the new groups such the sum
    of rounded grouping coefficients stays the same after rounding

    e.g. input = [0.4978, 0.3242, 0.1779], C = 100
         grouping = [30.50216, 11.675754, 57.822083]
         rounded = [31, 12, 58], sum --> 101
         smarted_rounded = [30, 12, 58], sum --> 100

    :param input:
    :param C:
    :return:
    '''

    # Get raw grouping
    c_per_group = tf.multiply(input_tensor, C)

    # Get naive rounding
    c_rounded = tf.round(c_per_group)

    # Calculate value to change if naive round is bad
    value_of_interest = _get_value_to_change(c_per_group,
                                             c_rounded)

    # Determine if smart rounding is necessary
    flag = tf.cond(tf.not_equal(tf.reduce_sum(c_rounded), C),
                   lambda: True,
                   lambda: False)

    tmp = _add_to_vec(c_rounded, value_of_interest, C)
    res = tf.cond(flag,
                  lambda: tmp,
                  lambda: c_rounded)

    return res
