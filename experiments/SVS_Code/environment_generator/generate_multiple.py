# 
# python-edition of "generate_multiple.bash"
# 
# usage:
# > python generate_multiple.py N DENSITY_MULTIPLIER
#

import sys
import os
import argparse
import shutil
import random

parser = argparse.ArgumentParser(description='Generate multiple environments')
parser.add_argument("N", help="Number of environments, e.g. 10",
                    default=10, type=int)
parser.add_argument("DENSITY_MULTIPLIER", help="DENSITY_MULTIPLIER, e.g. 0.5(easy), 1.0(medium), or 1.5(hard)",
                    default=1.0, type=float)
args = parser.parse_args()

homepath = os.path.dirname(__file__)
generated_envs_path = os.path.join(homepath, "generated_envs")

if os.path.exists(generated_envs_path):
    shutil.rmtree(generated_envs_path, ignore_errors=True)

for i in range(args.N):
    thispath = os.path.join(generated_envs_path, f"environment_{i}")
    os.makedirs(thispath)
    os.system(
        f"python {homepath}/obstacle_generator.py {random.randint(0,32767)} {args.DENSITY_MULTIPLIER}")
    shutil.move(os.path.join(homepath, "static_obstacles.csv"), thispath)
