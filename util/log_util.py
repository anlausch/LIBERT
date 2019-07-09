import logging
import config as c

def log_specs(logger):
  assert isinstance(logger, logging.Logger)
  logger.info("Epochs: %s" % str(c.EPOCHS))
  logger.info("Batch size: %s" % str(c.BATCH_SIZE))
  logger.info("Hidden size: %s" % str(c.HIDDEN_SIZE))
  logger.info("Do Eval: %s" % str(c.DO_EVAL))
  logger.info("Learning rate: %s" % str(c.LEARNING_RATE))
  logger.info("Orthogonality: %s (lambda=%s)" % (str(c.IMPOSE_ORTHOGONALITY), str(c.LAMBDA)))
  logger.info("Shared private: %s" % str(c.SHARED_PRIVATE))
  logger.info("Max Seq Len: %s" % str(c.MAX_SEQ_LEN))
  logger.info("Vocabulary size: %s" % str(c.VOCAB_SIZE))