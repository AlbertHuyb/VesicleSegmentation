import os
from config import *
# from utils.test import test
model_list = [MODEL_NAME]

synapse_list = os.listdir(os.path.join(OUTPUT_DIR,'full'))

for i in model_list:
    for s in synapse_list:
        if i[-4:] == '.pth':
            print(i)
            # test(i,s)
            os.system("python test.py %s %s"%(i,s))
            print("python test.py %s %s"%(i,s))
        else:
            print(i[-4:])

