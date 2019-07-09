import logging
import subprocess

import tensorflow as tf

import wn_config as c
from util.process_util import run_ssh_or_shell_command
from util.timer import Timer


class EvalRoutineCheckpointSaverListener(tf.train.CheckpointSaverListener, tf.train.SessionRunHook):
  """
  This class is a solution for this problem: https://github.com/tensorflow/tensorflow/issues/14283
  """

  def __init__(self, model_dir, path_eval_script, server=None,
               params=None, **kwargs):
    """
    If data_path = model_fn = input_fn = None then path_eval_script needs to point to eval_routine_multitask.py
    :param model_dir: directory from which to load the model parameters
    :param path_eval_script: path to eval script, can point to shell script that executes tf cpu only
    :param data_path: path to tf records file from which to load the data
    """
    self._logger = logging.getLogger()
    self._server = server
    self._params = " ".join(["%s=%s" % (str(k), str(v)) for k, v in params.items()])
    self._kwargs = kwargs

    if 'model_fn' in kwargs:
      self._kwargs['model_fn'] = kwargs['model_fn'].__name__

    if 'input_fn' in kwargs:
      self._kwargs['input_fn'] = kwargs['input_fn'].__name__

    # SessionRunHook Interface
    self._current_best_loss = None
    self.MAX_WAIT = c.TENACITY
    self._get_current_best_dev_loss = None
    self._new_best_dev_loss = None
    self._set_current_best_loss = None
    self._patience_counter = None
    self._increment_counter = None
    self._reset_counter = None

    # CheckpointSaverListener Interface
    self.eval_process = None
    self._model_dir = model_dir
    self.path_eval_script = path_eval_script
    self.timer = Timer()

    # graph variables for maintaining best dev-loss
    self._get_current_best_dev_loss = None
    self._get_current_dev_loss = None
    self._new_dev_loss = None
    self._set_dev_current_loss = None
    self._new_best_dev_loss = None
    self._set_current_best_loss = None

    # graph variables for maintaining patience
    self._patience_counter = None
    self._increment_patience = None
    self._reset_patience = None

    # graph variables for flagging when to check for new results
    self._is_dirty = None
    self._new_dirty_value = None
    self._set_dirty = None

  """
  CheckpointSaverListener Interface: launch eval after checkpoint has been issued
  """

  def after_save(self, session, global_step_value):
    if self.eval_process is None:
      self._launch_eval(global_step_value)
    else:
      result = self._fetch_result()
      self._logger.info(result)
      result_dict = eval(result[0])
      pid = result_dict.pop('pid', None)
      if pid:
        EvalRoutineCheckpointSaverListener._clean_up_dummy_process(pid, self._server)
      loss = result_dict["loss"]

      self._logger.info("set dirty true with dev-loss: %s" % str(loss))
      session.run([self._set_dev_current_loss, self._set_dirty],
                  {self._new_dev_loss: loss, self._new_dirty_value: True})
      self._launch_eval(global_step_value)

  def _fetch_result(self):
    self._logger.info("Waiting for eval worker's results.")
    try:
      eval_results = self.eval_process.communicate(timeout=20 * 30)
    except subprocess.TimeoutExpired:
      self._logger.warning("Result not fetchable after 10 seconds, waiting another (max) 20m.")
      try:
        eval_results = self.eval_process.communicate(timeout=20 * 60)
        duration = self.timer.stop()
        self._logger.error("Evaluation took longer than train %s, increase train_steps." % duration)
      except subprocess.TimeoutExpired as e:
        self._logger.error("Result not fetchable after 10min, spawn new evaluator in the next run.")
        self.eval_process = None
        raise e
    self._logger.info("Results retrieved.")
    return eval_results

  def _launch_eval(self, global_step_value):
    self.global_step_value = global_step_value
    self.timer.start()
    args = [c.PYTHON_RUNTIME, self.path_eval_script]
    for k, v in self._kwargs.items():
      args.extend(['--' + str(k), str(v)])
    args.extend(['--model_dir', self._model_dir])
    args.extend(['--params', self._params])

    command = " ".join(args)
    print("Launched Eval process with:\n%s" % command)
    self.eval_process = run_ssh_or_shell_command(command, self._server)

  @staticmethod
  def _clean_up_dummy_process(pid, server=None):
    """
    Eval routines are allowed to launch a dummy process to occupy the gpu if they finish earlier than
    the train process. Those resources are released here.
    :param pid:
    :return:
    """
    kill_command = 'kill -9 %s' % pid
    run_ssh_or_shell_command(kill_command, server)

  """
  SessionRunHook Interface: If new eval result exists check if its new highscore and if so, save it
  """

  def begin(self):
    """
    Add necessary operations for patience and for eval loss
    :return:
    """
    # if self.graph_should_be_constructed: #TODO: check why in tensorboard namescope eval_loss_1 is created
    with tf.variable_scope("eval_loss", reuse=tf.AUTO_REUSE):
      init = tf.constant(10000000.0)
      self._get_current_best_dev_loss = tf.get_variable(name="current_best_loss", dtype=tf.float32, initializer=init)
      self._new_best_dev_loss = tf.placeholder(dtype=tf.float32)
      self._set_current_best_loss = tf.assign(self._get_current_best_dev_loss, self._new_best_dev_loss)

      self._get_current_dev_loss = tf.get_variable(name="current_loss", dtype=tf.float32, initializer=init)
      self._new_dev_loss = tf.placeholder(dtype=tf.float32)
      self._set_dev_current_loss = tf.assign(self._get_current_dev_loss, self._new_dev_loss)

    with tf.variable_scope("patience_counter", reuse=tf.AUTO_REUSE):
      self._patience_counter = tf.get_variable(dtype=tf.int32, name="counter", initializer=tf.zeros_initializer(),
                                               shape=())
      self._increment_patience = tf.assign_add(self._patience_counter, 1)
      self._reset_patience = tf.assign(self._patience_counter, 0)

      self._is_dirty = tf.get_variable(dtype=tf.bool, initializer=tf.constant(False), name="is_dirty")
      self._new_dirty_value = tf.placeholder(dtype=tf.bool)
      self._set_dirty = tf.assign(self._is_dirty, self._new_dirty_value)

  def before_run(self, run_context):
    requests = {"is_dirty": self._is_dirty,
                "current_dev_loss": self._get_current_dev_loss,
                "current_best_dev_loss_loss": self._get_current_best_dev_loss}
    return tf.train.SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    is_dirty = run_values[0]["is_dirty"]

    # Do nothing for as long as there is no new eval result
    if is_dirty:
      self._logger.info(run_values)
      current_smallest = run_values[0]["current_best_dev_loss_loss"]
      loss = run_values[0]["current_dev_loss"]
      new_best_dev_loss = loss < current_smallest

      # hits on regular checks
      self._logger.info("current eval/dev loss: %s " % str(loss))
      if new_best_dev_loss:
        fetches = [self._reset_patience, self._set_current_best_loss]
        feed_dict = {self._new_best_dev_loss: loss}
        patience_count, _ = run_context.session.run(fetches, feed_dict)
      else:
        feed_dict = {}
        fetches = [self._increment_patience]

      feed_dict[self._new_dirty_value] = False
      fetches.extend([self._set_dirty])
      results = run_context.session.run(fetches, feed_dict)
      patience_count = results[0]

      if loss < current_smallest:
        self._logger.info("New best loss: %s" % loss)
      else:
        self._logger.info("Loss not improved, incremented patience to %s" % patience_count)

      if all([self.MAX_WAIT, patience_count >= self.MAX_WAIT]):  # TODO: test this expression
        run_context.request_stop()
        self._logger.info("No loss improvement after %s steps, requesting stop" % self.MAX_WAIT)

  def end(self, session, global_step_value=None):
    # if global step is none then the method is called when object takes role of CheckpointSaverListener
    # otherwise its called as SessionHook
    if global_step_value:
      result = self._fetch_result()
      self._logger.info(result)
      result_dict = eval(result[0])
      pid = result_dict.pop('pid', None)
      if pid:
        EvalRoutineCheckpointSaverListener._clean_up_dummy_process(pid, self._server)
      loss = result_dict["loss"]
      self._logger.info("last dev-loss: %s" % str(loss))


class ExtendedModeKeys(tf.estimator.ModeKeys):
  MULTI_TASK = "multitask"
  ENCODE = "encode"


  # TODO: move all args to kwargs and make api calls generic that way
