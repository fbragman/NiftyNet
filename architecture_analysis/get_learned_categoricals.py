import tensorflow as tf
import tensorflow.contrib.image # weird hack when import_meta_graph has KeyError for ImageProjectionTransform

def get_learned_categoricals(path_to_model_checkpoint):

    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)

    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(config=config) as sess:

            # restore model
            saver = tf.train.import_meta_graph(path_to_model_checkpoint + '.meta')
            saver.restore(sess, path_to_model_checkpoint)

            # get trainable variables
            trainables = g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            # get categorical probabilities
            cat_ps = [x for x in trainables if 'categorical_p' in x.name]

            # get names
            cat_names = [x.name for x in cat_ps]

            # take softmax
            cat_ps_softmax = []
            for cat in cat_ps:
                cat_ps_softmax.append(tf.nn.softmax(cat, axis=1))

            cats = sess.run(cat_ps_softmax)

    return cats, cat_names