
"""
The LSTM model.
"""

import os
import sys
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Embedding, Dense, Bidirectional
from BaseModel import BaseModel
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
sys.path.insert(0, '../utils')
from visualization import plot_loss_graph, plot_metric_graph
from keras.models import Sequential


class LSTM(BaseModel):
           
    def __init__(self, config):
      super().__init__(config)

      self.output_dim = self.config.model.output_dim
      self.return_sequences = self.config.model.return_sequences
      self.dense_nodes = self.config.model.dense_nodes
      self.dropout = self.config.model.dropout
      self.model = None
      self.train_df = None
      self.val_df = None
      self.test_df = None
      self.train_dataset = []
      self.val_dataset = []
      self.test_dataset = []
      self.vec_layer = None
 

    """
    Creates a data pipeline using the TF Dataset API.
    Input: data(tf.data), labels(tf.data), batch_size(int), is_trainins(bool)
    Output: dataset(tf.dataset)
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



    """
    Creates the training pipeline.
    Splits the training data to train and validation using the train_test_split() 
    function from sklearn.
    Input: train_data(tf.data)
    """ 
    def create_train_pipeline(self, train_data):
        #Split training dataset into train and validation sets
        self.train_df, self.val_df = train_test_split(train_data, test_size=0.2)
 
        # Data pipelines for 3 different datasets
        self.train_dataset = self.create_data_pipeline(self.train_df.tweet, 
                  self.train_df.sentiment, batch_size=1024)
        self.val_dataset = self.create_data_pipeline(self.val_df.tweet, 
                  self.val_df.sentiment, batch_size=128, is_training=False)



    """
    Vectorizes the data using the Text Vectorization API from tf.keras 
    (https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization).
	
	A TextVectorization layer should always be either adapted over a dataset or supplied with a vocabulary.
	Calling adapt() on a TextVectorization layer is an alternative to passing in a precomputed vocabulary on construction via the vocabulary argument.
	During adapt(), the layer will build a vocabulary of all string tokens seen in the dataset, sorted by occurance count, with ties broken by sort order of the tokens (high to low).
	At the end of adapt(), if max_tokens is set, the voculary wil be truncated to max_tokens size. 
	For example, adapting a layer with max_tokens=1000 will compute the 1000 most frequent tokens occurring in the input dataset.
	If output_mode='tf-idf', adapt() will also learn the document frequencies of each token in the input dataset.
	
    Input: num of maximum features(int), num of max length(int)
    """ 
    def data_vectorization(self, max_features=75000, max_len=50):
      self.vec_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_features, output_sequence_length=max_len)
      self.vec_layer.adapt(self.train_df.tweet.values)



    """
    Create the model architecture.
    Output: the created model(Model)
    """ 
    def build_model(self):    

      self.model = tf.keras.Sequential([
      self.vec_layer,
      tf.keras.layers.Embedding(
          input_dim=len(self.vec_layer.get_vocabulary()),
          output_dim=self.output_dim,
          # Use masking to handle the variable sequence lengths
          mask_zero=True),
      tf.keras.layers.Dropout(self.dropout),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.dense_nodes,
      dropout=0.2, recurrent_dropout=0.2)),
      tf.keras.layers.Dense(self.dense_nodes, activation='relu'),
      tf.keras.layers.Dropout(self.dropout),
      tf.keras.layers.Dense(1)
      ])

      print(self.model.summary())

      return self.model



    """
    Return the name of the saved model.
    """
    def get_model_name(self):
      return os.path.join(self.config.model.model_path, 'best_model.ckpt')


    """
    Get training callbacks (ModelCheckpoint, ReduceLROnPlateau and EarlyStopping)
    """
    def get_callbacks(self):

      model_checkpoint = ModelCheckpoint(self.get_model_name(), monitor='val_loss', 
                save_best_only=True)
      rls = ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0)
      early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5)

      return [model_checkpoint, rls, early_stopping]

      
    """
    Performs the model training.
    Output: the training history of the model
    """ 		
    def train(self):
        batch_size = self.config.train.batch_size
        epochs = self.config.train.epochs
        steps_per_epoch = self.train_df.tweet.shape[0] // batch_size
        
        self.model.compile(loss=self.config.train.loss, 
        optimizer=self.config.train.optimizer.type,
                  metrics=self.config.train.metrics)

        callbacks = self.get_callbacks()

        # Fitting the model
        model_history = self.model.fit(self.train_dataset, epochs=epochs, 
                  batch_size=batch_size, 
                  steps_per_epoch=steps_per_epoch, validation_data=self.val_dataset,
                  callbacks=callbacks)     

        print("Plotting the loss and accuracy graphs...")
        plot_loss_graph(model_history)
        plot_metric_graph(model_history)                  		

        # Saving Model
        self.model_path = self.config.model.model_path
        version = self.config.model.version
        export_path = os.path.join(self.config.model.model_path, 
                  self.config.model.version)
        print('export_path = {}\n'.format(export_path))

        tf.keras.models.save_model(
            self.model,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format='tf',
            signatures=None,
            options=None
        )



    """
    Creates the data pipeline for the test dataset.
    Input: train_data(tf.data)
    """ 
    def create_test_pipeline(self, test_data):
        self.test_df = test_data
 
        # Data pipelines for test datasets
        self.test_dataset = self.create_data_pipeline(self.test_df.tweet, 
                  self.test_df.sentiment, batch_size=1024)



    """
    Performs the model evaluation.
    Output: the resulted predictions of the model
    """ 				
    def test(self):
        pass

        #Create data pipeline for test
        test_dataset = self.create_data_pipeline(self.test_df.tweet, self.test_df.sentiment, 
                                                      batch_size=128, is_training=False)

        model_path = os.path.join(self.config.model.model_path, 
                  self.config.model.version)
        model = tf.keras.models.load_model(model_path)
        model.evaluate(test_dataset)



