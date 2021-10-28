<p align="center">
  <img align="center" width="130" height="80" src="https://upload.wikimedia.org/wikipedia/commons/3/31/Boston_University_wordmark.svg">
</p>

<h5 align="center">MET CS 777 - Big Data Analytics</h5>
<h6 align="center">Fall 2021</h6>

<h2 align="center"> Natural Language Processing with PySpark </h2>
<h2 align="center"> Disaster Tweets classification </h2>

### Author 
* Maryam Asghari
* Email : masghari@bu.edu

## Project description 
predicting whether a given tweet is about a real disaster or not using pySpark by using ml libraries and by my own implementations 

<p align="center">
<img alt="arg" src="https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/images/wordcloud.png" width="800"/>
</p>

### Dataset

Source : https://www.kaggle.com/c/nlp-getting-started/data

#### Files     
* train.csv - the training set
* test.csv - the test set (Does not include labels)

#### Columns
- id - a unique identifier for each tweet
- text - the text of the tweet
- location - the location the tweet was sent from (may be blank)
- keyword - a particular keyword from the tweet (may be blank)
- target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

### Python scripts

#### Utils 

Python files for functions that I used in The Notebooks 

* Plots.py
* prep_ml.py
* prep_rdd.py
* nn_func.py

#### Scripts to run each classifier in spark

* LogisticRegression.py
* NaiveBayes.py
* SVM.py
* Trees.py

<p align="center">
<img alt="arg" src="https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/images/mlresults.png" width="800"/>
</p>

* RDD_logisticRegression.py

<p align="center">
<img alt="arg" src="https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/images/LR_cost.png" width="400"/>
</p>

* LR_Optimizers.py

<p align="center">
<img alt="arg" src="https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/images/lr.png" width="800"/>
</p>

* RDD_SVM.py

<p align="center">
<img alt="arg" src="https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/images/Svm_cost.png" width="400"/>
</p>

* SVM_Optimizer.py

<p align="center">
<img alt="arg" src="https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/images/svm.png" width="800"/>
</p>

* RDD_NN.py

<p align="center">
<img alt="arg" src="https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/images/plotSGD_ADAm.png" width="400"/>
</p>

#### Notebooks

* [Part1](https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/docs/Notebooks/Part1.ipynb) 

    * Data visualization
    * LogisticRegression
    * NaiveBayes
    * SVM
    * Trees
    * RDD_logisticRegression
    * LR_Optimizers
    * RDD_SVM
    * SVM_Optimizer

* [Part2](https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/docs/Notebooks/Part2.ipynb)

    * RDD_NN
    
#### Presentation

* [Presentation](https://github.com/metcs/met-cs-777-assignment-project-maryamxasghari/blob/master/docs/presentation_CS777.pptx)

#### How to run the scripts

```python
spark-submit LogisticRegression.py './nlp-getting-started/train.csv'
```
```python
spark-submit NaiveBayes.py './nlp-getting-started/train.csv'
```
```python
spark-submit SVM.py './nlp-getting-started/train.csv'
```
```python
spark-submit Trees.py './nlp-getting-started/train.csv'
```

**NOTE:** Following scripts need NLTK library

```python
spark-submit RDD_logisticregression.py './nlp-getting-started/train.csv' './output_LR'
```
```python
spark-submit RDD_svm.py './nlp-getting-started/train.csv' './output_svm'
```
```python
spark-submit LR_Optimizers.py './nlp-getting-started/train.csv' './out/optimizer:'
```
```python
spark-submit SVM_Optimizers.py './nlp-getting-started/train.csv' './out/optimizer2'
```
```python
spark-submit RDD_NN.py './nlp-getting-started/train.csv' './out/NN_rdd'
```




