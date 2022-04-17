# SentimentAnalysis
Sentiment analysis of Twitter data.

An LSTM model is trained from scratch to perform the sentiment analysis. 
The TextVectorization preprocessing layer (https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) is used before feeding the data to the model's Embedding layer. 

The model achieves a good accuracy of 82.81% when testing on the two classes (positive and negative).

The training loss and accuracy history are depicted on the following plots: 

![image](https://user-images.githubusercontent.com/30274421/163729477-95d4c0b9-417d-4534-aee6-010386259173.png)

![image](https://user-images.githubusercontent.com/30274421/163729480-08e6eb90-717b-42b1-bbd9-9e2c169f6b9c.png)



In order to access the source files from colab, the project should be located under your Google drive, as shown below:
![image](https://user-images.githubusercontent.com/30274421/163729367-7d3d43c5-cf22-467d-afbf-9e2a6dac1eb2.png)

The folders "datasets" and "models" should also be created as shown, where will be saved the downloaded dataset and trained models, respectively. 
