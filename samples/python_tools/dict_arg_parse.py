import argparse
import ast
import json

parser = argparse.ArgumentParser(prog="run_tiles")
parser.add_argument("-m", "--mapping", type=json.loads)
parser.add_argument("-e", "--eval", type=ast.literal_eval)

sample_str_dict = '{"ab":"dc"}'
sample_str_dict2 = "{'ab':'dc'}"

if False:
    res = parser.parse_args(["-m", sample_str_dict])
elif True:
    res = parser.parse_args(["-e", sample_str_dict]), parser.parse_args(
        ["-e", sample_str_dict2]
    )
else:
    res = parser.parse_args([])

print(res)
