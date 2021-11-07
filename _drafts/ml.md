---
filename: ml
layout: post
title: machine learning
---

# what is machine learning?
Audience: Freshman-year student
Theme: The intuition behind machine learning.

The first time that I heard the word machine learning I thought that robots learned task to imitate us toward to performs task. I thought this was magic and difficult. Everything far from reality.

# Introduction

Machine learning learns a model or decision function based on previous data.

I believe everyone had been listen a about machine learning and how is changing the industries and science. 
Machine learning had been accelerating science and industries. 

Protein folding, antibiotic discovery, robust animal behavior monitoring are some examples of how machine learning have been accelerating advances in science. Machine learning had been applied to many industries health, finance and even the software engineering industry. For some tasks that are hard to program, machine learning had changed the paradigms from explicitly programing to train a machine learning model. Andrej Karpathy explained more about this is his blog post titled [Software 2.0](https://link.medium.com/YcPpazSFZkb). 

Seems that machine learning works like magic, but isn't. The purpose of this blog post is to uncover the magic behind of machine learning and answer; how machine learning learn from the data to make decision even in new examples?

The goal of machine learning is to learn a decision function based on training data meanwhile generalize to new examples.
From this goal there three aspect are crucial for machine learning: 1) the decision function; 2) how efficiently represent the input data?; 3) how measure the generalization of the model? In the next section I will introduce the intuition of this aspects but first a motivation example. 

Example beach day. 

Imagine a simple case where you are designing an application to know if an specific day is a good day to go to the beach? Normally, when I go to the beach there two facts that I check: 1) the precipitation probability and 2) the swell size. Figure X shows an example of previos days where the days where labeled as good day (blue dot) and bad day (red dot) to go to the beach. The x-axis shows the precipitation probability and the y-axis shows the swell size. In the graph the the days labeled as good day are clustered on the bottom left which means low precipitation probability and low swell size. 

# Decision Function
If we see the Figure X, that collect some data points about good and bad days to go to the beach we can see a clear pattern. The pattern looks like this:

If today is in the bottom left area of the graph will be a good day to go to the beach, otherwise will be a bad day to go to the beach.

But how to formalize this pattern? First we need to define what is a decision function. A decision function is a learned function that  receive the features and decide which class will be the today. For a given training data could exists many decision function that perform very well on the training dataset. This depends on the complexity of the model.

A decision function is a function that evaluate the features and decide which class to assign to that data point based on previous data or training data.
 

Figure XXX shows three different decision functions that are valid for our example dataset. The Fig XXXa shows a simple line that divides good day examples from bad day examples. 

 The decision function is a function that is learned by the training data. This 

# Feature Representation
Two things are important in machine learning the model and the data. The data should have the enough and the right information about the problem to solve. Figure XX shows a graph where the y-axis was changed to wind speed which is an irrelevant feature to solve this task. In the best case the model ignore the features of wind speed and relies on only precipitation probability. Relying on just precipitation probability do have the complete information make a good choice. Machine learning learn about reasonable patterns but do not make magic from data that not make sense.  

The representation of the problem should have a complete information.
Generally, data scientist spend a considerable amount of time considering a large list of features to solve the task. This process is called feature engineering which consist on collect and transform certain features in some way to simplify the feature representation of the problem.

# Generalization

Now we have a good representation and a good model that learn perfectly the training data. Are we ready to deploy our application to try out on new data? Well not yet, first we need to verify if our model can generalize to new data points never seen before. How can we measure the generalization of all possible future data points? We should collect the whole possible data points in our training set? The answer is no. Generally we divide the dataset in two: the training set and the testing set. The ideas is that the testing set contains novel examples that do not appears in the training set. Therefore, we can use the testing set to approximate the the generalization error.

Now that we have a way to approximate the generalization of a model, you can encounter the following scenarios:

1. Poor training and testing performance (Underfitting)
2. Good Training and poor testing performance (Overfitting)
3. Good training and testing performance

Poor training performance is a indicative of underfitting meaning that your feature representation is not the adequate or that the model is not complex enough for the training dataset. In my opinion, if various models do not provide a good performance maybe you should simplify the feature representation by doing feature engineering.

Assume that you have model that learns how to perform some task, and when you tried to evaluate the model on the testing dataset you found that have a poor performance. This means that the model learned was capable to learn the training but do not generalize well. This is a sign of overfitting, which maybe is due to high complexity of your model.

# Conclusion

After this post I hope you have a good intuition about how machine learning works and understand that is not magic. Decision functions uses training data to make a decision. Good Machine Learning models relies on have a good feature representation relevant to the task to complete. Generalization measure the robustness of the model to novel examples never seen before. For this reason, Testing datasets are required to measure the generalization error need.