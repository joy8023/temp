## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_mnist import MNIST
from utils import prepare_data
from worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator
import utils
import pickle
import numpy as np

class newdata:
    def __init__(self, od, ol, ad, al):
        self.origin_data = np.array(od)
        self.origin_label= np.array(ol)
        self.adv_data = np.array(ad)
        self.adv_label= np.array(al)

def load_data(filepath):
	f = open(filepath,'rb')
	newdata = pickle.load(f)
	f.close
#	print(np.array(newdata.origin_data).shape)

	return newdata

detector_I = AEDetector("./defensive_models/MNIST_I", p=2)
detector_II = AEDetector("./defensive_models/MNIST_II", p=1)
reformer = SimpleReformer("./defensive_models/MNIST_I")

id_reformer = IdReformer()
classifier = Classifier("./models/mnist")

detector_dict = dict()
detector_dict["I"] = detector_I
detector_dict["II"] = detector_II

data = load_data('./mnist/100start0.pkl')

operator = Operator(MNIST(), classifier, detector_dict, reformer)
thrs = operator.get_thrs(drop_rate={"I": 0.8, "II": 0.8})
pas = operator.filter(data.origin_data, thrs)
print('True positive:',pas)
pas2 = operator.filter(data.adv_data, thrs)
print('True negative:', 1 - pas2)
operator.operate(data)



'''
idx = utils.load_obj("example_idx")
_, _, Y = prepare_data(MNIST(), idx)
f = "example_carlini_0.0"
testAttack = AttackData(f, Y, "Carlini L2 0.0")

evaluator = Evaluator(operator, testAttack)
evaluator.plot_various_confidences("defense_performance",
                                   drop_rate={"I": 0.001, "II": 0.001})
'''
