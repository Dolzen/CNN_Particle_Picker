import os
import pickle
import CNN_Model_Experimental
import sys
#import model_runner
#import



model_template_path = (os.path.dirname(os.path.realpath(sys.argv[0]))) + '/CN_Template_21.py'



def Train_Models(list_of_runs):
print('Loading Params')
run_params = pickle.load(open('list_of_runs.p', "rb"))
for items in run_params:
    CNN_Model_Experimental.run_model(items)

