import tensorflow as tf


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

    def __call__(self, hard=False):
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