---
layout: post
title: Machine Learning
tags: [ml]
description: This blog introduces machine learning.
---

# what is machine learning?
Audience: Freshman-year student
Theme: The intuition behind machine learning.

# Introduction
I believe everyone has listened about machine learning and how it is changing industries and science. Machine learning has been accelerating science and industries. Protein folding, antibiotic discovery, robust animal behavior monitoring are examples of how machine learning has been accelerating advances in science. Many industries such as finance, health and even the software engineering industry have been applying machine learning to facilitate, automatize or guide essential processes. For some tasks that are hard to program, machine learning had changed the paradigms from explicitly programing to train a machine learning model. Andrej Karpathy explained more about this is his blog post titled [Software 2.0](https://link.medium.com/YcPpazSFZkb). 

In many applications machine learning seems to works like magic, but isn't magic. The purpose of this blog post is to uncover the magic behind of machine learning and answer; how machine learning learn from the data to make decisions?

Keep in mind that the goal of machine learning is to learn a decision function based on training data meanwhile generalizing to new examples. From this goal there three aspects are crucial for machine learning: 1) the decision function; 2) how efficiently represent the input data?; 3) how to measure the model generalization? In the next section I will introduce the intuition of this aspects but first a motivation example. 

# Decision Function 

Imagine a simple case where you are designing an application to know if today is a good day to go to the beach? Normally, when I go to the beach I check two measurements: 1) precipitation probability and 2) rip currents. If we collect these measurements from previous days and plot them, we get the following graph in Figure X. This graph has labeled examples of good days (blue dots) and bad days (red dots). Note that the x-axis shows the precipitation probability and the y-axis shows rip currents. You can immediately notice a pattern from this graph, good days to go to the beach are clustered on the bottom left of the graph meaning that good days have low precipitation probability and low rip current. 

But how to formalize this pattern? First we need to define what is a decision function. A decision function is a function that recieve features or measurements as inputs and decide which class to assign based on the training data points. For a given training data could exist many decision functions. This depends on the complexity of the model and the data. Figure XX shows three different decision functions that are valid for our example training dataset. The Fig XXa shows a simple line (Logistic Regression) that divides good day examples from bad day examples.

# Feature Representation
In machine learning there two thing we can have control the model, and the data. In this section I will talk about the more important of both; the data. The data can have multiples representations and many features some relevent and other irrelvenat to respect the target task. The model should recieve enough information to solve the target problem. You cannot expect that machine learning model can figure out the solution with irrelevant features or incomplete information. Figure XXX shows a graph where the y-axis was changed from precipitation probability to wind speed which is an irrelevant feature to solve this task. Notice that we not only introduce an irrelevant feature but remove part of information relevant to solve this task. In the best case the model ignore the wind speed features and relies on only rip currets. Relying on just precipitation probability, do have the complete information to make a good choice. The lesson here is that machine learning learn about reasonable patterns but do not make magic from data that not make sense.

Generally, data scientists spend a considerable amount of time considering which features are relevant to solve the task. It is critical to remove irrelevant features because they can introduce noise to the model. Some features might are not good by themselves but combining them with others and transforming them into new combined features could result in relevant features. This process is called feature engineering which consist on collect and transform certain features in some way to simplify the feature representation of the problem.

# Generalization

Now that we have a good representation and a good model that learn perfectly the training data. Are we ready to deploy our application to predict data from users? Well not yet, first we need to verify if our model can generalize to new data points never seen before. But how can we measure the generalization to all possible future data points? Should we collect the whole possible data points in our training set? The answer is no. Generally we divide the dataset in two: the training set and the testing set. The idea is evaluate the model on the testing set that contains novel examples that do not appears in the training set to approximate the the generalization error.

Now that we have a way to approximate the generalization of a model, you can encounter the following scenarios:

1. Poor training and testing performance (Underfitting)
2. Good Training and poor testing performance (Overfitting)
3. Good training and testing performance

Poor training performance is a indicative of underfitting meaning that your feature representation is not the adequate or that the model is not complex enough for the training dataset. In my opinion, if various models do not provide a good performance maybe you should simplify the feature representation by doing feature engineering.

Assume that you have model that learns how to perform some task, and when you tried to evaluate the model on the testing dataset you found that have a poor performance. This situation means that the model learned was capable of memorizing the training but did not generalize well. This is a sign of overfitting, which maybe is due to high complexity of your model. You could try techniques to avoid overfiiting. One of them is to reduce the complexity of your model.

If you encounter a good training and testing performance you are good to go. Note that the testing performance most of the time is lower that the training performance. Maybe you should try to optimize some models parameter to improve your model performance.

# Conclusion

In this introductory post about machine learning, we discused how decision function uses training data to make a decisions. We made an emphasis on how the input data representation could affect your model performance. Also, we introduce the testing set as a way to approximate the generalization error.  I hope that after this post, now you have good intuition and understand how machine learning works.