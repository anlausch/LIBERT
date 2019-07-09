"""
When debugging multiple runs/versions you can configure your IDE to execute this script, it'll
clear everything in MODEL_HOME such that the folders don't need to be cleaned up manually
"""
import shutil
import config as c
import os

if os.path.exists(c.MODEL_HOME):
  shutil.rmtree(c.MODEL_HOME)
  os.makedirs(c.MODEL_HOME)
