import os
import subprocess
from socket import gethostname
from socket import getfqdn

def run_ssh_or_shell_command(command, server=None):
  env = os.environ.copy()
  if env.get("LD_LIBRARY_PATH", None):
    command = "export LD_LIBRARY_PATH=%s; " % env["LD_LIBRARY_PATH"] + command

  if env.get("PYTHONPATH", None):
    command = "export PYTHONPATH=%s; " % env["PYTHONPATH"] + command

  if server is not None and server != gethostname() and server != getfqdn():
    command = "ssh %s '%s'" % (server, command)
  return subprocess.Popen(['/bin/bash', '-c', command], env=env, stdout=subprocess.PIPE)