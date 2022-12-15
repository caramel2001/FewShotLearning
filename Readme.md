# 1. Introduction

Computer Vision is an ever growing field with a plethora of interesting problems to tackle ranging from object detection to image segmentation. For decades traditional machine learning was used to tackle these problems (and have done so quite well) but the emergence of deep learning based approaches have allowed us to reach groundbreaking results. Intrigued by this, our team selected the classic problem of image recognition and explored the current deep learning models for this task.

### 1.1 Problem Statement

Image recognition is the process of identifying an object or a feature in an image or video. Existing methods work by learning features and patterns in the training samples and use these to perform a multi-class classification. Convolutional Neural Networks were the forerunners in this field with modern approaches leveraging the attention mechanism of the transformer architecture to generate encodings for the input images.
However it is well known these models are quite data hungry, requiring a large number of training samples per class. This poses a challenge in domains where labeling data is expensive or sufficient data does not exist. With the aim of tackling this problem our team decided to delve deep into the problem of small datasets and developing clever architectures to overcome them.

### 1.2 Literature Review

The problem of learning with insufficient training examples has been explored extensively in the statistical and machine learning domains. This problem is especially relevant for deep learning methods, as these algorithms are characteristically data hungry. One of the solutions to this problem is the use of deep transfer learning, where the latent knowledge of one problem domain learned by the deep networks can be utilized to solve the problems in another domain (Tan et al., 2018).
Another approach to tackle the insufficient training data problem is the use of Siamese networks. The first Siamese network was created for signature verification by comparing images (Bromley et al., 1993). Later, a contrastive energy function was proposed to increase the difference between dissimilar pairs and decrease the difference between similar pairs (LeCun et al., 2005). A seminal paper by Facebook explored the viability of Siamese networks in Face Verification, and generated embeddings by using the second to the last layer in a deep neural network trained with a SoftMax loss (Taigman et al., 2014). The triplet loss function utilizing the Anchor-Positive-Negative framework with the margin parameter was described in a paper by Google (Schroff et al., 2015). Additionally, this paper also discussed the methods for sampling these triplets and suggested training on hard examples rather than randomly sampled triplets from a batch during training.
The domain of flower classification with the Oxford 102 flowers dataset has been explored before by various researchers (Xia et al., 2017; Wu et al., 2018) to varying degrees of success. Our model performs better than a majority of the published methods, and matches the benchmarks of contemporary approaches with 98% accuracy on the test set.

### 1.3 Dataset Description

We felt the Oxford Flowers 102 dataset was very relevant for our problem. The dataset is a collection of 102 flower categories commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images. The images have large scale, pose and light variations. Additionally, there are categories that have large variations within the category and several very similar categories.

The dataset is divided into a training set, a validation set and a test set. The training set and validation set each consist of 10 images per class (a total of 1020 images each). The test set consists of the remaining 6149 images (minimum 20 per class).

In summary we wanted to train a model to perform a 102 class classification problem, using a dataset that only had 10 samples per class. Sounds ambitious right! But this is a good example of how there might not be an abundance of training data to work with.

### 1.4 Our Approach

To tackle this classification problem we have to reformulate it in terms of a Few Shot learning task. In case of standard classification, the input image is fed into a series of layers, and finally at the output layer we generate a probability distribution over all the classes (typically using a Softmax layer). However, in few shot classification the network learns a similarity function that takes in the encodings of a pair of images as input and will produce a similarity score denoting the chances that the two input images belong to the same class. In other words, our network is not learning to classify an image directly to any of the output classes. Rather, it is learning a similarity function to numerically express how similar they are.
This has 2 major benefits,

● From our experiments we will show that this network does not require many instances of a class to attain a reasonable accuracy. This is the domain of Few Shot Learning.

● We can easily expand the dataset with more classes without touching the architecture of the network. This makes it more flexible than a classification network that would require changes to both the last layer and retraining to accommodate for the new class.

A valid question is how to build a classifier from output. A simple solution would be to keep representative images for every training class on hand, allowing us to quickly form pairs with a given training image before feeding it into our network. If our network is trained well, the pair with the highest similarity score would the pair form the same class!
We implemented the idea outlined above and this report will go through the process step by step highlighting the successes and key learnings in detail.

## Conclusion and Discussion

Although few-shot learning is a promising direction for future research, more work is needed to improve its effectiveness. In particular, more research is needed on how to effectively select and use training data, how to design better models, and how to better exploit prior knowledge.
The main aim of our project was to develop a model to perform image classification with a very small amount of training data. We achieved this by leveraging pre-trained models, transfer learning and creating clever architectures. We started by benchmarking the performance of transfer learning by pretraining on a large dataset and then fine tuning on our smaller flower dataset. We found the BiT models performed the best.
We then explored the use of Siamese Networks for Few Shot Learning. This architecture allowed us to generate encodings for images. We built a model and trained it to produce similar encodings for images from the same class, and different encodings for images from different classes. We validated the model using the N-Way Few Shot Learning task, and found it performed reasonably well.
We then explored the use of the Triplet Loss function to further improve the encodings generated. We used semi-hard triplet sampling to train our model, and found the encodings to be greatly improved. We built a classifier using the embeddings (in essence performing a 102-way few shot learning task) and found that the model performed exceptionally well, reaching a classification accuracy of 98.8%.
10
We were quite pleased with the performance of the model. However, there are several areas of improvement that we would like to work on in the future.

1. More Training Data for creating Embeddings
   One of the most significant improvements we could make to the model would be to increase the number of training samples for generating the embeddings. Here, we used the 1020 images in the train dataset to train the model using triplet loss. Ideally, a much larger dataset should be used to train this deep network.
   Once the model is trained on a variety of flower images, we could use it to learn new flower encodings from another dataset and predict on the test set. This would require relatively few training images of these new flowers.

2. Model Architecture
   We could also try alternative model architecture for few-shot learning. For instance, the models we implemented were from papers published in 2016. Since then a variety of novel architectures have been developed to tackle this few-shot learning problem. A particularly interesting paper utilized GANs to generate images for the triplet loss, leading to higher quality embeddings of the anchor.
