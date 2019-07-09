#! /usr/bin/env python
import subprocess
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", default=False)
args = parser.parse_args()

print("")
global_results = {}
for server in ["dws04", "dws06", "dws07"]:
    gpu_user_memory = {}

    query_string = "%s nvidia-smi" % server
    process = subprocess.Popen(query_string.split(), stdout=subprocess.PIPE)
    result = process.communicate()
    result = result[0].decode("utf-8").split("\n")

    for line in result:
        components = re.sub( '\s+', ' ', line).strip().split()
        if len(components) > 1 and components[3] == "C" and "MiB" in components[5]:
            pid = components[2]
            gpu = components[1]
            memory = components[5]
            memory = int(memory[:-3])
            try:
                user_query_string = "%s ps -p %s -u --no-headers" % (server,pid)
                get_user_process = subprocess.Popen(user_query_string.split(),stdout=subprocess.PIPE)
                tmp_result = get_user_process.communicate()
                user = tmp_result[0].decode("utf-8").split()[0]
            except BaseException:
                user = pid

            if gpu not in gpu_user_memory:
                gpu_user_memory[gpu] = { user: (memory, [pid]) }
            else:
                user_memory = gpu_user_memory[gpu]
                if user not in user_memory:
                    user_memory[user] = (memory, [pid])
                else:
                    old_memory, pids = user_memory[user]
                    pids.append(pid)
                    user_memory[user] = (old_memory + memory, pids)

            pass
    global_results[server] = gpu_user_memory


for server, data in global_results.items():
    for gpu, usages in data.items():
        mb = sum([mb for mb, pids in usages.values()])
        print("{:<8} {:<5} {:>10} MiB".format(server, gpu, mb))
        server=""
    print("")

if args.verbose:
    print("-------------------\n")
    for server, data in global_results.items():
        for gpu, usages in data.items():
            for user, mb_pids in usages.items():
                mb, pids = mb_pids
                print("{:<8} {:<5} {:<10} {:>10} MiB {:<10}".format(server, gpu, user, mb, " "*5 + str(pids)))
                server=""
        print("")
