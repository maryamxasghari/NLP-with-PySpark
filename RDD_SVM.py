import sys
import time
import re
from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType ,ArrayType
from pyspark.sql import SparkSession
import nltk
import unidecode
from nltk.stem import WordNetLemmatizer,LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 
import pandas as pd
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: RDD_SVM.py <file> <output>", file=sys.stderr)
        exit(-1)

  
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('tweet').getOrCreate()        
    
    tweet = pd.read_csv(sys.argv[1])

    tweet_df= tweet[['id','text','target']]
    
    mySchema = StructType([ StructField("id", StringType(), True)\
                       ,StructField("text", StringType(), True)\
                       ,StructField("class", StringType(), True)])
            
    sparkDF=spark.createDataFrame(tweet_df, mySchema) 
    RDD = sparkDF.rdd
    RDD = RDD.map(lambda x: (x[0],x[2],x[1]))
    
    
    newStopWords = ['may','u','would','also','th','one','na','gt','w','two','us','mh','via','to','rt','pm', 'due','many','much','however','since','although','often','m','s','ll','ve','tweet','tweeter','blog']
    stop_words = set(stopwords.words('english')+ newStopWords)
    lemmer = WordNetLemmatizer()
    stemer = SnowballStemmer(language='english')

    def word_tokenizer(x):
        #lower case
        x = x.lower()
        #remove websites and mentiones
        x = re.sub(r'(?:\@|https?\://)\S+', '', x)
        # remove all non letter characters
        x = re.sub(r'[^a-zA-Z]', ' ', x).lstrip()
        # Remove non-unicode
        x = unidecode.unidecode(x)
        x = nltk.word_tokenize(x)
        # Remove stopwords and lemmatize/stemming
        x = [lemmer.lemmatize(w) for w in x if w not in stop_words]
    #     x = [stemer.stem(w) for w in x]
        return x    

    def get_dict(keyAndListOfWords):
        """make a 10000 words dictionary"""
        allWords = keyAndListOfWords.flatMap(lambda x: (x[2])).map(lambda x: (x, 1))
        allCounts = allWords.reduceByKey(lambda x,y: x + y) # I will use lambda instead of add maybe it would be faster in spark 
        topWords = allCounts.top(10000, key= lambda x: x[1])
        topWordsK = sc.parallelize(range(10000))
        dictionary = topWordsK.map(lambda x : (topWords[x][0], x))
        return dictionary        
    
    
    
    def freqArray (listOfIndices, numberofwords):
        """ function to get TF array provided in the homework"""
        returnVal = np.zeros(10000)

        for index in listOfIndices:
            returnVal[index] = returnVal[index] + 1
        returnVal = np.divide(returnVal, numberofwords)
        return returnVal

    def get_tfArray(keyAndListOfWords):
        """generates TF array same as homework 2"""
        # get a set of ('word1', ('ID', label , count)), ex : [('deed', ('1', '1', 5))]
        allWordsWithDocID = keyAndListOfWords.map(lambda x: (x[0], x[2], len(x[2]) , x[1])).flatMap(lambda x: ((j, ((x[0],x[3], x[2]))) for j in x[1]))  
        # join and link them, to get a set of ("word1", (dictionaryPos, ('docID', label,count) ))  [('family', (19, ('2585', '0', 15)))]
        allDictionaryWords = dictionary.join(allWordsWithDocID.distinct())
        # get (('docID',label, count) , dictionaryPos) 
        justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]) )
        # get a set of (('docID', count), [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
        allDictionaryWordsInEachDoc = justDocAndPos.groupByKey().mapValues(list)
        # get (docID , Array) . Array is normalized 
        allDocsWithLabel = allDictionaryWordsInEachDoc.map(lambda x: (x[0][0],x[0][1], freqArray(x[1], x[0][2])))
        return allDocsWithLabel    
    
    
    def get_tf_idfArray(tf_array, numberOfDocs):
        """Get tf_idfArray with true labels"""
        zeroOrOne = tf_array.map(lambda x: (x[0],x[1], np.where(x[2] != 0, 1, 0) ) )
        dfArray = zeroOrOne.reduce(lambda x, y: ("","", np.add(x[2], y[2])))[2]
        idfArray = np.log(np.divide(np.full(10000, numberOfDocs), dfArray))
        allDocsAsNumpyArraysTFidf = tf_array.map(lambda x: (x[0],x[1], np.multiply(x[2], idfArray)))
        return allDocsAsNumpyArraysTFidf   
    
    def top5coeff(coef):
        """Return the top 5 words with highest coef """
        # argpartition is faster since it will not sort the numbers
        Top5Pos = np.argpartition(coef, -5)[-5:]    
        top5coeff = []
        for i in Top5Pos:
            top5coeff.append([i , coef[i]])
        # we will sort now when we only have 5 values
        top5coeff_sorted = sorted(top5coeff, key=lambda x: abs(x[1]))[-5:]
        top5 = []
        # we will save the index which is the same as dictionary to get the words from dic later 
        for pair in top5coeff_sorted:
            top5.append(pair[0])
        return top5

    def loss_svm(row, w):
        """A function to calculate loss and gradient"""
        x = row[2]
        y = 1 if int(row[1]) ==1 else -1        
        ypy = y * np.dot(w , x)
        cost = max(0, 1 - ypy)
        gradient = 0 if ypy > 1 else -y * x
        return cost, gradient
    
    
    def svm(data,numberOfDocs):
        num_iteration = 200
        learning_rate = 0.01
        w = np.random.normal(0, 0.1, 10000)
        precision = 0.0001
        cost_array = []
        old_cost = 0
        c = 0.5
        n = numberOfDocs
        reg_lambda = 1 / (c * n)

        for i in range(num_iteration):
            rdd = data.map(lambda x: (loss_svm(x, w)))
            result = rdd.reduce(lambda x,y: [x[0] + y[0], x[1] + y[1] ])

            gradient = result[1] + reg_lambda * w
            cost = result[0] +  (reg_lambda /2) * np.linalg.norm(w)

            w = w - learning_rate * gradient

            #bold driver
            if cost < old_cost:
                learning_rate = learning_rate * 1.05
            else:
                learning_rate = learning_rate * 0.5        

            cost_array.append(cost)
 
            if( ((abs(old_cost-cost)) <= precision) ):
                print(f"Stoped at iteration {i} the Diff_cost = {old_cost-cost}" )
                break        

            old_cost = cost
            if i%20 == 0 :
                print('Iteration', i , ', Cost =', cost )
#             print('Iteration', i , ', Cost =', cost )
            

        Top5Pos = top5coeff(w)

        print('\nThe top 5 words with largest coefficients:\n')
        print("-"*100)
        top5_words = dictionary.filter(lambda x: x[1] in Top5Pos).map(lambda x: x[0]).collect()
        print(top5_words)
        return w , cost_array
    
    
    def prediction( x , w):
        """calculate the theta and label with 1 if theta > 0  """
        theta = np.dot(x[2], w)
        label = 1 if theta > 0 else 0
        return x[0] , int(x[1]), label
    
    def t_f_Pos_Neg(x):
        """compare the y and label , save 4 values for each row to keep track of tp,tn,fp,fn"""
        tp = 1 if (x[1] == x[2] and x[1] == 1) else 0
        tn = 1 if (x[1] == x[2] and x[1] == 0) else 0
        fp = 1 if (x[1] != x[2] and x[1] == 0) else 0
        fn = 1 if (x[1] != x[2] and x[1] == 1) else 0
        return x[0],tp,tn,fp,fn

    
    keyAndListOfWords =  RDD.map(lambda x : (x[0],x[1],word_tokenizer(x[2]) ))
    numberOftweets = RDD.count()
    dictionary = get_dict(keyAndListOfWords)


    allDocsWithLabel = get_tfArray(keyAndListOfWords)
    tf_idfArray = get_tf_idfArray(allDocsWithLabel, numberOftweets)
    
    training, testing = tf_idfArray.randomSplit([0.8, 0.2], seed=123)

    
    train_start = time.time()
    w, cost_array = svm(training,numberOftweets)
    train_end = time.time()
    train_time = train_end - train_start
    
    test_start = time.time()
    test_pred = testing.map(lambda x: prediction(x, w))
    Pos_Neg = test_pred.map(lambda x :  t_f_Pos_Neg(x))
    
    tp,tn,fp,fn = Pos_Neg.map(lambda x : (x[1],x[2],x[3],x[4])).reduce( lambda x,y :(x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))

    f1 = tp / (tp + 0.5*(fp+fn))
    acc = (tp+tn) / (tp+tn+fp+fn)
    test_end = time.time()
    test_time = test_end - test_start
    
    print("\n============== Results: ==============\n")
    print("Accuracy of model was: %{0:2.2f}\n".format(acc*100))
    print("f1 score of model was: %{0:2.2f}\n".format(f1*100))

    print(f'- Train time = {round(train_time,4)}\n')
    print(f'- Test time = {round(test_time,4)}\n')

   
    dataToASingleFile = sc.parallelize(cost_array).coalesce(1)
    dataToASingleFile.saveAsTextFile(sys.argv[2])
    

    
    sc.stop()

