import tensorflow as tf
import tensorflow_hub as hub
from estimator_extension import ExtendedModeKeys

def elmo_threshold_model_fn(features, labels, mode, params):
    '''
    Hyperparameter: Learning rate
    '''
    name_scope = params['name_scope'] if params.get('name_scope', None) else 'elmo_threshold_model'
    with tf.variable_scope(name_scope):
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

        # default: a fixed mean - pooling of all contextualized word representations with shape[batch_size, 1024].
        speech_embedding = elmo(features['speech'], signature="default", as_dict=True)["default"]
        argument_embedding = elmo(features['argument'], signature="default", as_dict=True)["default"]

        normed_speech_embedding = tf.nn.l2_normalize(speech_embedding, axis=1)
        normed_argument_embedding = tf.nn.l2_normalize(argument_embedding, axis=1)
        cos_similarities = tf.diag_part(tf.matmul(normed_speech_embedding, normed_argument_embedding, transpose_b=True))
        #threshold = tf.Variable(0.5, name="threshold")
        #predicted_classes = tf.math.greater(cos_similarities, threshold)
        W = tf.get_variable("W_softmax",
                            shape=[1, 2],
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.Variable(tf.constant(0.1, shape=[2]))
        # tf.matmul does not allow for shapes with rank<2
        scores = tf.matmul(tf.expand_dims(cos_similarities,1), W) + b

        predicted_classes = tf.argmax(scores, axis=1)

        if mode == ExtendedModeKeys.PREDICT:
            predictions = {
                'class_ids': predicted_classes,
                'similarities': cos_similarities,
                'scores': scores
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=labels)
        loss = tf.reduce_mean(losses)

        #labels_direct = tf.argmax(labels)
        # TODO: Here we need to compute the real metric
        #streaming_accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')

        eval_metrics = {}
        if mode == ExtendedModeKeys.EVAL:
            #eval_metrics['accuracy'] = streaming_accuracy
            #tf.summary.scalar('accuracy', streaming_accuracy)
            return tf.estimator.EstimatorSpec(mode, loss=loss)#, eval_metric_ops=eval_metrics)

        if mode == ExtendedModeKeys.TRAIN:
          optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
          train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
          return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)