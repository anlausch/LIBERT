import tensorflow as tf

class DummyCheckpointSaver(tf.train.CheckpointSaverHook):
    """
    Something is conflicting between pycharm and tensorflow checkpoint saving and it gets stuck. This class is used
    for debugging purposes in order to test async eval with tensorflow estimators.
    """

    def __init__(self):
        self.saving_listeners=[]
        self._listeners=[]

    def begin(self):
        pass

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        pass

    def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):  # pylint: disable=unused-argument
        pass

    def end(self, session):
        pass

