import codecs
import numpy as np
import os
import argparse

def parse_predictions(input_path, output_path, task="STSB"):
  """
  :param input_path:
  :param output_path:
  :param task:
  :return:
  >>> parse_predictions("/test_results.tsv", "test.tsv", task="MNLI")
  """
  if task != "STSB":
    import run_classifier_wordnet
  else:
    import run_regression_wordnet
  predicted_labels = []
  if task == "MRPC":
    #ids = MrpcProcessor().get_test_examples(os.environ['GLUE_DIR'] + "/MRPC")
    labels = run_classifier_wordnet.MrpcProcessor().get_labels()
  if task == "RTE":
    labels = run_classifier_wordnet.RTEProcessor().get_labels()
  if task == "QNLI":
    labels = run_classifier_wordnet.QNLIProcessor().get_labels()
  if task == "QNLIV2":
    labels = run_classifier_wordnet.QNLIProcessor().get_labels()
  if task == "MNLI":
    labels = run_classifier_wordnet.MnliProcessor().get_labels()
  if task == "SST2":
    labels = run_classifier_wordnet.SST2Processor().get_labels()
  if task == "CoLA":
    labels = run_classifier_wordnet.ColaProcessor().get_labels()
  if task == "QQP":
    labels = run_classifier_wordnet.QQPProcessor().get_labels()
  if task == "diagnostic":
    labels = run_classifier_wordnet.DiagnosticProcessor().get_labels()
  with codecs.open(input_path, "r", "utf8") as f_in:
    for line in f_in.readlines():
      predictions = np.array(line.split("\t"), dtype=np.float32)
      if task != "STSB":
        predicted_index = np.argmax(predictions)
        predicted_labels.append(labels[predicted_index])
      else:
        predicted_labels.append(predictions[0])
    f_in.close()
  with codecs.open(output_path, "w", "utf8") as f_out:
    f_out.write("index\tprediction\n")
    for i, prediction in enumerate(predicted_labels):
      f_out.write(str(i) + "\t" + str(prediction) + "\n")
    f_out.close()


def write_fake_predictions(output_path, task="WNLI"):
  """
  :param input_path:
  :param output_path:
  :param task:
  :return:
  >>> write_fake_predictions("/WNLI.tsv", task="WNLI")
  """
  if task != "STSB":
    import run_classifier_wordnet
  else:
    import run_regression_wordnet
  if task == "MNLI":
    test_examples = run_classifier_wordnet.MnliProcessor().get_test_examples(os.environ['GLUE_DIR'] + "/" + task, False)
    labels = run_classifier_wordnet.MnliProcessor().get_labels()
  elif task == "QQP":
    test_examples = run_classifier_wordnet.QQPProcessor().get_test_examples(os.environ['GLUE_DIR'] + "/" + task)
    labels = run_classifier_wordnet.QQPProcessor().get_labels()
  elif task == "WNLI":
    test_examples = run_classifier_wordnet.WNLIProcessor().get_test_examples(os.environ['GLUE_DIR'] + "/" + task)
    labels = run_classifier_wordnet.WNLIProcessor().get_labels()
  elif task == "CoLA":
    test_examples = run_classifier_wordnet.ColaProcessor().get_test_examples(os.environ['GLUE_DIR'] + "/" + task)
    labels = run_classifier_wordnet.ColaProcessor().get_labels()
  elif task == "STSB":
    test_examples = run_regression_wordnet.STSBProcessor().get_test_examples(os.environ['GLUE_DIR'] + "/" + task)
  elif task == "diagnostic":
    test_examples = run_classifier_wordnet.DiagnosticProcessor().get_test_examples(os.environ['GLUE_DIR'] + "/" + task)
    labels = run_classifier_wordnet.DiagnosticProcessor().get_labels()
  with codecs.open(output_path, "w", "utf8") as f_out:
    f_out.write("index\tprediction\n")
    if task != "STSB":
      for i, data in enumerate(test_examples):
        f_out.write(str(i) + "\t" + str(labels[0]) + "\n")
    else:
      for i, data in enumerate(test_examples):
        f_out.write(str(i) + "\t" + str(2.5) + "\n")
    f_out.close()



def main():
  parser = argparse.ArgumentParser(description="Running prediction parser")
  parser.add_argument("--task", type=str, default=None,
                      help="Input path in case train and dev are in a single file", required=True)
  parser.add_argument("--input_path", type=str, default="/work/anlausch/replant/bert/predictions/wn_binary_32_5e-05_3.0/test_results.tsv",
                      help="Input path in case train and dev are in a single file", required=False)
  parser.add_argument("--output_path_root", type=str, default="/work/anlausch/replant/bert/predictions/wn_binary_32_5e-05_3.0/",
                      help="Input path in case train and dev are in a single file", required=False)

  args = parser.parse_args()
  task = args.task
  input_path = args.input_path
  root = args.output_path_root
  output_path = root + str(task) + ".tsv"
  parse_predictions(input_path, output_path, task)

if __name__ == "__main__":
  main()