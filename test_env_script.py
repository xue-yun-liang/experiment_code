import os
from subprocess import Popen
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

proms = [
	"test_CRLDSE.py",
	"test_ERDSE.py",
	"test_MOMPRDSE.py",
]

for prom in proms:
	cmd = "python3 {}".format(prom)
	proc = Popen(cmd, shell=True)
	proc.wait()