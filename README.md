# CMPE-258-Assignment-6

## Assignment Description

Note this is multiple colabs assignment - please put proper readme with all colabs - do not copy paste from hint colabs. write it in your own way.

Give a detailed video going over the code of each colab and upload it.

Make sure you organize your GitHub directory properly and read.me

You should execute and show while explaining what you wrote in colab s.

### Part 01 Description

Write simple colabs to illustrate various data augmentation and generalization techniques (with A/B tests).
Write in tensorflow.

a. l1 l2

b. dropout

c. earlystop

d. montecarlo dropout

e. various initializations and when to use what

f. batch norm

g. custsom dropout, custom regularization

h. using callbacks and tensorboard

i. Using keras tuner

j. use keras cv data augmentation

k. Data Augmentation and Classification for Image

L. FastAI Data Augmentation Capabilities

#### Colab links for the above parts:
a) to g)

https://colab.research.google.com/drive/1xI0fHj8Cx8OCuX7PVVQtu9R3fKJY-oN4?usp=sharing

h)callbacks and tensorboard

https://colab.research.google.com/drive/1AXzZ_e00dTh6m-itqV5ZuwqcKsRVSsZw?usp=sharing

i) using keras tuner
https://colab.research.google.com/drive/1xI0fHj8Cx8OCuX7PVVQtu9R3fKJY-oN4?usp=sharing
j)keras_cv_data_aug
https://colab.research.google.com/drive/1K4AOhki_pjBxnGi3LZbBDe64Ob8sC3Xg?usp=sharing

k)
Audio-data-aug: https://colab.research.google.com/drive/1Mk7ii4ATOeRH1WAdRSPJ-s2tYFkxM9HC?usp=sharing

Text:  https://colab.research.google.com/drive/1Ki5fH70D6DhJkpbT2oPWQdbOKJqWWj_d?usp=sharing

Image: https://colab.research.google.com/drive/11JbEo0HzgV8t-OBgCyaYHY0p5fUy08Ow?usp=sharing

Timeseries: https://colab.research.google.com/drive/1d0R5lxv1LStR_hXOawmxDvFwpPT8QFAB?usp=sharing

video: https://colab.research.google.com/drive/1K7X40cZmw9x_pIUtFzs8Rr84Jo_vmpBL?usp=sharing

l)
fast_ai_aug
https://colab.research.google.com/drive/1wJfzgHzKy9aRRDK4U31dr56jpVrqnmiu?usp=sharing

### Part 02

Write a colab/colabs where you use advanced Keras deep learning constructs and concepts.

Please ensure you use the links provided for hints as examples. Use your own creativity. Properly annotate your colab/colabs appropriately and write proper explanation and description. Properly demonstrate each of these aspects with either individual colabs or one colab having all these.

A. User custom learning rate scheduler (one cycler scheduler example)

B. Use custom dropout (MCAlphaDropout section)

C. Use custom normalization (MaxNormDense section)

D. Use tensorboard (see links)

E. Use custom loss function (HuberLoss section)

F. Use custom activation function, initializer regularizer and kernel weight constraint (sections leaky_relu, my_glorot_initializer, MyL1Regularizer, my_positive_weights)

G. Use custom metric (HuberMetric section)

H. Use custom layers (Sections: exponential_layer, MyDense, AddGaussianNoise, LayerNormalization)

I. Use custom model (ResidualRegressor and ResidualBlock sections)

J. Custom optimizer (MyMomentumOptimizer sections)

K. Custom Training Loop  

Colab link for part-2:
https://colab.research.google.com/drive/10tBlabBr99qtqmo9rCh4Gw9718dF2Qxz?usp=sharing

## Assignment Deliverables

Please see below for the list of deliverables that have been submitted for this assignment.


 [Demo Video](https://drive.google.com/file/d/11kFkc2svSrb8EBoHJ0mTqJ-EFM2MCQCw/view?usp=drive_link): Video that contains the following:

## References Used

1. [10_neural_nets_with_keras.ipynb](https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb)
2. [11_training_deep_neural_networks.ipynb](https://github.com/ageron/handson-ml3/blob/main/11_training_deep_neural_networks.ipynb)
3. [awesome-data-augmentation](https://brunokrinski.github.io/awesome-data-augmentation/)
4. [keras_cv](https://keras.io/keras_cv/)
5. [tensorflow](https://www.tensorflow.org/)
6. [AugLy](https://github.com/facebookresearch/AugLy)
7. [AugLy: A new data augmentation library to help build more robust AI models](https://ai.meta.com/blog/augly-a-new-data-augmentation-library-to-help-build-more-robust-ai-models/)
8. [TensorFlow Core: Data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
9. [07_sizing_and_tta.ipynb](https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb)
10. [11_training_deep_neural_networks.ipynb](https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb)
11. [12_custom_models_and_training_with_tensorflow.ipynb](https://github.com/ageron/handson-ml2/blob/master/12_custom_models_and_training_with_tensorflow.ipynb)





# CMPE-Assignment-7: Computer Vision

##  Description

### Part-1 - Supervised Contrastive Learning

Write a colab to demonstrate supervised contrastive learning loss based supervised classification versus regular softmax based one. Please provide necessary visualizations.


### Part-1 - Transfer Learning on Various Modalities

Write simple colabs to transfer learn on images, videos, audios - with both as a feature extractor as well as a fine tuning usecases.

Image: Showcase basic transfer learning for a classification task (either cats/dogs or breeds of dogs) in a colab - with both as a feature extractor as well as a fine tuning usecase


### Part-3 - Zero-Shot Transfer Learning

Write a colab showcasing the following:

- zero-shot transfer learning using the CLIP model
- transfer learning using state of art models from tfhub (Eg: use bigtransfer for example)


## Assignment Deliverables

Please see below for the list of deliverables that have been submitted for this assignment.

1. `.ipynb`:  Colab for Part 1 portion of assignment.
2. `.ipynb`:  Colab for Part 2 portion of assignment.
3. `.ipynb`:  Colab for Part 3 portion of assignment.

###  References

1. [Deep Vision with CNN](https://docs.google.com/presentation/d/1UxtHDwjViC7VpSb0zB-kajGQ-TwznQmc-7LsbHRfO3s/edit#slide=id.p)
2. [Contrastive loss for supervised classification](https://towardsdatascience.com/contrastive-loss-for-supervised-classification-224ae35692e7)
3. [Supervised Contrastive Learning](https://keras.io/examples/vision/supervised-contrastive-learning/)


1. [Transfer Learning for Audio Data with YAMNet](https://blog.tensorflow.org/2021/03/transfer-learning-for-audio-data-with-yamnet.html)
2. [Action Recognition with an Inflated 3D CNN](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)
3. [Text Classification with Movie Reviews](https://www.tensorflow.org/hub/tutorials/tf2_text_classification)
4. [Transfer Learning in NLP with Tensorflow Hub and Keras](https://amitness.com/posts/tensorflow-hub-for-transfer-learning)
5. [Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)
6. [Applying transfer-learning in CNNs for dog breed classification](https://towardsdatascience.com/dog-breed-classification-using-cnns-and-transfer-learning-e36259b29925)
7. [Transfer Learning with TensorFlow Part 1: Feature Extraction](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/04_transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb)
8. [Transfer Learning with TensorFlow Part 2: Fine-tuning](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/05_transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb)
9. [Transfer Learning with TensorFlow Part 3: Scaling up (Food Vision mini)](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/06_transfer_learning_in_tensorflow_part_3_scaling_up.ipynb)


1. [How to Try CLIP: OpenAI’s Zero-Shot Image Classifier](https://towardsdatascience.com/how-to-try-clip-openais-zero-shot-image-classifier-439d75a34d6b)
2. [Image Classification using BigTransfer (BiT)](https://keras.io/examples/vision/bit/)
3. [Transfer learning with TensorFlow Hub](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub)



