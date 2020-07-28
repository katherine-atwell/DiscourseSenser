from dsenser import DiscourseSenser
import _pickle as cPickle

senser = DiscourseSenser(None)
senser.train(train_set, dsenser.WANG,
             dsenser/data/models/pdtb.sense.model.WangSenser, dev_set)
