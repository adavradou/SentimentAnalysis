# -*- coding: utf-8 -*-
"""Model config in json format"""


CFG = {
  "data": {
      "url": "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
      "name": "Sentiment140",
      "base_path": "/content/mydrive/SentimentAnalysis/datasets"
   },
   "train": {
      "batch_size": 1024,
      "epochs": 20,
      "optimizer": {
        "type": "adam"
      },
      "metrics": ["accuracy"],
      "loss": "binary_crossentropy"
   },
    "model": {
      "output_dim": 64,
      "return_sequences": True,
      "dense_nodes": 64,
      "dropout": 0.3,
      "model_path": "/content/mydrive/SentimentAnalysis/models",
      "version": 'lstm_model_2',
      "data_vectorization": {
        "max_features": 75000,
        "max_len": 40
	   }
	   
   }
}