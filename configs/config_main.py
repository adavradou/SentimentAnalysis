# -*- coding: utf-8 -*-
"""Model config in json format"""


CFG = {
   "data": {
       "url": "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
	   "name": "Sentiment140"
   },
   "train": {
       "batch_size": 64,
       "epoches": 20,
       "optimizer": {
           "type": "adam"
       },
       "metrics": ["accuracy"]
   },
   "model": {
       "input_shape": (1,),
	   "start_ch": 256,
	   "input_dim": 75000+1,
	   "output_dim": 128,
	   "return_sequences": True,
	   "dense_nodes": 64,
	   "dropout": 0.3
	   
   }
}