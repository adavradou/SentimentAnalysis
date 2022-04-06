"""
The LSTM model.
"""

import os
import sys
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Embedding, Dense
from BaseModel import BaseModel
from sklearn.model_selection import train_test_split


class LSTM(BaseModel):
           
    def __init__(self, config):
      super().__init__(config)

      self.input_shape = self.config.model.input_shape
      self.start_ch = self.config.model.start_ch
      self.input_dim = self.config.model.input_dim
      self.output_dim = self.config.model.output_dim
      self.return_sequences = self.config.model.return_sequences
      self.dense_nodes = self.config.model.dense_nodes
      self.dropout = self.config.model.dropout
      self.model = None
      self.train_df = None
      self.val_df = None
      self.train_dataset = []
      self.val_dataset = []
      self.test_dataset = []
      self.vec_layer = None

 
    """
    Data Pipeline Function using TF Dataset API
    """ 
    def create_data_pipeline(self, texts, labels, batch_size=32, is_training=True):
      # Convert the inputs to a Dataset.
      dataset = tf.data.Dataset.from_tensor_slices((texts,labels))
      # Shuffle, repeat, and batch the examples.
      dataset = dataset.cache()
      if is_training:
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
      dataset = dataset.batch(batch_size, drop_remainder=True)
      # Return the dataset.
      return dataset

    def split_training_data(self, train_data):
        #Split training dataset into train and validation sets
        self.train_df, self.val_df = train_test_split(train_data, test_size=0.2)
 
        # Data pipelines for 2 different datasets
        self.train_dataset = self.create_data_pipeline(self.train_df.tweet, self.train_df.sentiment, batch_size=1024)
        self.val_dataset = self.create_data_pipeline(self.val_df.tweet, self.val_df.sentiment, batch_size=128, is_training=False)


    def data_vectorization(self, max_features=75000, max_len=50):
      self.vec_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_features, output_sequence_length=max_len)
      self.vec_layer.adapt(self.train_df.tweet.values)


    def build_model(self):    
      print(self.input_shape)  
      print(self.vec_layer)
      words = Input(shape=self.input_shape, dtype=tf.string)
      vectors = self.vec_layer(words)
      embeddings = Embedding(input_dim=self.input_dim, output_dim=self.output_dim)(vectors)
      output = tf.keras.layers.LSTM(self.start_ch, return_sequences=True, name='LSTM_1')(embeddings)
      output = tf.keras.layers.LSTM(self.start_ch, name='LSTM_2')(output)
      output = Dropout(self.dropout)(output)
      output = Dense(self.dense_nodes, activation='relu')(output)
      output = Dense(1,activation='sigmoid')(output)

      self.model = Model(words,output)

      print(self.model.summary())

      return self.model
      
		
    def train(self):
        #pass
        batch_size = 1024
        epochs = 3
        steps_per_epoch = self.train_df.tweet.shape[0] // batch_size

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fitting the model
        model_history = self.model.fit(self.train_dataset, epochs=epochs, batch_size=batch_size, 
                  steps_per_epoch=steps_per_epoch, validation_data=self.val_dataset)        
		

        return model_history.history['loss'], model_history.history['val_loss']
		
		
    def evaluate(self):
        pass
		#predictions = []
        #for image, mask in self.dataset.take(1):
            #predictions.append(self.model.predict(image))

        #return predictions