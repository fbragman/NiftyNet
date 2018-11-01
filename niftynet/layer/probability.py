import tensorflow as tf
import numpy as np

class Dirichlet(object):

    def __init__(self, mean, precision, batch_size=1, num_samples=1):
        """
        batch_size automatically set to 1 --> Dirichlet grouping equal for all batches
        Otherwise, set batch_size to batch_size
        :param mean:
        :param precision:
        :param num_samples:
        """
        self.batch_size = batch_size
        self.alpha = self.calculate_alpha(mean, precision)
        self.num_samples = num_samples

    def __call__(self):
        """
        Sample from a Dirichlet distribution
        :return:
        """
        # create distribution object
        dist = tf.distributions.Dirichlet(self.alpha)
        # sample from the distribution
        samples = dist.sample(self.num_samples)

        if self.num_samples > 1:
            # Calculate expectation
            samples_mean = dist.mean(name='dirichlet_expectation')
        else:
            samples_mean = None

        return tf.squeeze(samples), tf.squeeze(samples_mean)

    def calculate_alpha(self, mean, precision):
        """
        Calculate Dirichlet concentration parameter from mean and precision parameterisation
        :param mean:
        :param precision:
        :return:
        """
        alpha = tf.multiply(mean, precision[:, tf.newaxis])
        if self.batch_size == 1:
            alpha = alpha[tf.newaxis, :]
        return alpha


class HardCategorical(object):

    def __init__(self, proportions, N):

        self.p = proportions
        self.N = N

    def __call__(self):
        """
        Create one-hot mask for N rows based on proportions defined by probabilities
        :return:
        """
        num = self.p * self.N
        num = [int(x) for x in num]
        num_sum = np.sum(num)
        if num_sum != self.N:
            # difference in expected number
            delta = self.N - num_sum
            # biggest proportions
            idx = np.argmax(self.p)
            num[idx] = num[idx] + delta

        onehot = np.zeros((self.N, 3), dtype=np.float32)
        onehot[:num[0], 0] = 1
        onehot[num[0]:num[1]+num[2], 1] = 1
        onehot[num[1]+num[2]:-1, 2] = 1

        cat_mask = tf.constant(onehot, dtype=tf.float32)
        return cat_mask


class GumbelSoftmax(object):

    def __init__(self, probabilities, temperature):

        self.probs = probabilities
        self.temp = temperature

    def sample_gumbel(self, shape, eps=1e-20):
        """
        Sample from a Gumbel(0, 1)
        :param shape:
        :param eps:
        :return:
        """
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def gumbel_softmax_sample(self):
        """
        Draw a sample from the Gumbel-Softmax distribution
        :return:
        """
        y = self.probs + self.sample_gumbel(tf.shape(self.probs))
        return tf.nn.softmax(y / self.temp)

    def __call__(self, hard):
        """
        Sample from the Gumbel-Softmax
        If hard=True
            a) Forward pass:    argmax is taken
            b) Backward pass:   Gumbel-Softmax approximation is used
                                since it is differentiable
        else
            a) Gumbel-softmax used
        :param hard:
        :return:
        """
        y_sample = self.gumbel_softmax_sample()
        if hard:
            # argmax of y sample
            y_hard = tf.cast(tf.equal(y_sample,
                                      tf.reduce_max(y_sample, 1, keepdims=True)),
                             y_sample.dtype)
            # let y_sample = y_hard but only allowing y_sample to be used for derivative
            y_sample = tf.stop_gradient(y_hard - y_sample) + y_sample
        return y_sample


def entropy_loss(probabilities: tf.Tensor):
    """Sum of entropies.

    Args:
        probabilities:  A `Tensor` of size [d_1, d_2, ..., d_{n-1}, num_classes]
            which sums up to 1.0 along the last dimension.
    Returns:
        loss: Loss output
    """
    probs_clipped = tf.clip_by_value(probabilities, 1e-10, 0.9999999)
    return tf.reduce_sum(-probs_clipped * tf.log(probs_clipped))


def entropy_loss_by_layer(probability_list: list):
    """
    Sum of per layer average entropies
    :param probability_list:
    :return: sum of average per layer categoricals
    """
    summed_entropy = 0
    for layer_p in probability_list:
        probs_clipped = tf.clip_by_value(layer_p, 1e-10, 0.9999999)
        avg_layer_entropy = tf.reduce_mean(entropy(probs_clipped))
        summed_entropy += avg_layer_entropy

    return summed_entropy


def entropy(probs: tf.Tensor):
    """
    H(p) = -p * log(p)
    :param probs:
    :return: an [N,1] tensor where for every row, the entropy of the probs is
             calculated
    """
    return tf.reduce_sum(-probs * tf.log(probs), axis=1)
