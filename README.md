# Problem Statement
This assignment is about working with __Seq2Seq__ class of problems using __Attention__

We are supposed to use 2 public datasets to train on the given example code.
One of the challenges is that the datasets are not meant only for seq2seq problems and have some extra fields which needs to be removed. 
Basically we need to create a dataset with tab seperated lines

first1 second1

first2  second2

...

..


Once we get this kind of structure we can apply the code and train the architecture. 

![image](img/datacreation.png)


## Dataset1: Question-Answer Pair

Given a question find the associated answer.
The problem can be easily formulated as a seq2seq problem where encoder is working with the question part and the decoder is generating the answer

We have used the dataset available at  https://www.kaggle.com/c/quora-question-pairs/data this is same as that on http://www.cs.cmu.edu/~ark/QA-data/ (some issues in accessing this site so took the kaggle data)

The data is a tab seperated file with the fields like

ArticleTitle | Question | Answer | DifficultyFromQuestioner | DifficultyFromAnswerer | ArticleFile

### Data Samples:

Abraham_Lincoln Who suggested Lincoln grow a beard? Grace Bedell. hard medium data/set3/a4

Abraham_Lincoln When did the Gettysburg address argue that 
America was born? 1776 hard hard data/set3/a4

Abraham_Lincoln When did the Gettysburg address argue that America was born? 1776. hard hard data/set3/a4


We are only interested in the fields corresponding to the "Question" and "Answer" and hence that has been extracted and saved in a separate file


[QAPairs](https://github.com/TSAI-END3-Group/Session_6_Attention/blob/master/Attention_pytorch_QA_Pairs.ipynb)  contains the code 



## Dataset2: Quora question similarity

* Dataset consists of over 400,000 lines of potential question duplicate pairs.

* Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair.

### Data Fields:

* id - the id of a training set question pair
* qid1, qid2 - unique ids of each question 
* question1, question2 - the full text of each question
* is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

### Data sample

![image](https://qph.fs.quoracdn.net/main-qimg-ea50c7a005eb7750af0b53b07c8caa60)

The Problem  can be formulated as as Seq2Seq problem in the following way  
Given a question find out another way of writing the same question. These sort of problems are A=>B  (__text entailement__)

We have created a separate file containing only the duplicate pairs. This can be identified by the flag is_duplicate
The network will be trained to understand how the same sentence can be written in another way

question1, question2


[Quora]https://github.com/TSAI-END3-Group/Session_6_Attention/blob/master/Attention_pytorch_QA_Pairs.ipynb)  contains the code 


## Results

### Question Answers Pair

#### Training result
Model is converging 
![image](img/training_QA.png)

#### Evaluation
Testing random samples with the trained model
![image](img/training_result_QA.png)

#### Attention visualization 
We can see an example of how the attention is being used
![image](img/attention_res_QA.png)








### Attention

