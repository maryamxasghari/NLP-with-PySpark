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
        print("Usage: SVM_Optimizers.py <file> <output>", file=sys.stderr)
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

    def svm(data , optimizer):
        """Learning using diff optimizer"""
        num_iteration = 200
        learning_rate = 0.01
#         coef = np.zeros(10000)
        coef =np.random.normal(0, 0.1, 10000)
        precision = 10e-8
        cost_array = []
        old_cost = 0
        old_coef = coef
        beta = 0.9
        beta2 = 0.999
        c = 0.5
        n = data.count()
        reg_lambda = 1 / (c * n)
        momentum = np.zeros(10000)
        prev_mom = np.zeros(10000)
        second_mom = np.array(10000)
        gti = np.zeros(10000)
        epsilon = 10e-8
        l2_old = np.linalg.norm(coef)
        
        
        for i in range(num_iteration):
            rdd = data.map(lambda x: (loss_svm(x, coef)))
            result = rdd.reduce(lambda x,y: [x[0] + y[0], x[1] + y[1] ])

            gradient = result[1] + reg_lambda * coef
            cost = result[0] +  (reg_lambda /2) * np.linalg.norm(coef)

            if optimizer == 'SGD':
                coef = coef - learning_rate * gradient

            if optimizer =='Momentum':
                momentum = beta * momentum + learning_rate * gradient
                coef = coef - momentum

            if optimizer == 'Nesterov':
                parameter_temp = coef - beta * prev_mom
                coef = parameter_temp - learning_rate *gradient
                prev_mom = momentum
                momentum = beta * momentum + learning_rate * gradient

            if optimizer == 'Adam':
                momentum = beta * momentum + (1 - beta) *gradient
                second_mom = beta2 * second_mom + (1 - beta2) * (gradient**2)
                momentum_ = momentum / (1 - beta**(i + 1))
                second_mom_ = second_mom / (1 - beta2**(i + 1))
                coef = coef - learning_rate * momentum_ / (np.sqrt(second_mom_) + epsilon)

            if optimizer == 'Adagrad':
                gti += gradient**2
                adj_grad = gradient / (np.sqrt(gti)  + epsilon)
                coef = coef - learning_rate  * adj_grad

            if optimizer == 'RMSprop':
                sq_grad = gradient**2
                exp_grad = beta * gti / (i + 1) + (1 - beta) * sq_grad
                coef = coef - learning_rate / np.sqrt(exp_grad + epsilon) * gradient
                gti += sq_grad            

            
            #bold driver
            if cost < old_cost:
                learning_rate = learning_rate * 1.05
            else:
                learning_rate = learning_rate * 0.5


            cost_array.append(cost)
    #         Stop if the cost is not descreasing
#             diff_l2 = abs(np.linalg.norm(coef) - np.linalg.norm(old_coef))
#             if (abs(old_cost-cost) <= precision) :
#                 print("Stoped at iteration", i )
#                 break

            old_coef = coef
            old_cost = cost
            if i%20 == 0 :
                print('Iteration', i , ', Cost =', cost ) #, "diff in the l2 norm of coef :" ,diff_l2 )

        return coef, cost_array

    def prediction(x, coef):
        """calculate the theta and label with 1 if theta > 0  """
        theta = np.dot(x[2], coef)
        label = 1 if theta > 0 else 0
        return x[0] ,int(x[1]), label
    
    def t_f_Pos_Neg(x):
        """compare the y and label , save 4 values for each row to keep track of tp,tn,fp,fn"""
        tp = 1 if (x[1] == x[2] and x[1] == 1) else 0
        tn = 1 if (x[1] == x[2] and x[1] == 0) else 0
        fp = 1 if (x[1] != x[2] and x[1] == 0) else 0
        fn = 1 if (x[1] != x[2] and x[1] == 1) else 0
        return x[0],tp,tn,fp,fn
    
    def result_print(testing, coef):
        """calculate and Print the results"""
        test_pred = testing.map(lambda x: prediction(x, coef))
        Pos_Neg = test_pred.map(lambda x :  t_f_Pos_Neg(x))
        tp,tn,fp,fn = Pos_Neg.map(lambda x : (x[1],x[2],x[3],x[4])).reduce( lambda x,y :(x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))
        f1 = tp / (tp + 0.5*(fp+fn))
        acc = (tp+tn) / (tp+tn+fp+fn)
        print("\n============== Results: ==============\n")
        print("Accuracy of model was: %{0:2.2f}\n".format(acc*100))
        print("f1 score of model was: %{0:2.2f}\n".format(f1*100))

    def run_save(training,testing,optimizer):
        """Run the learning function and save the cost array"""
        coef, cost_array = svm(training,optimizer)
        result_print(testing, coef)
        dataToASingleFile = sc.parallelize(cost_array).coalesce(1)
        dataToASingleFile.saveAsTextFile(sys.argv[2]+optimizer)
        
        
    keyAndListOfWords =  RDD.map(lambda x : (x[0],x[1],word_tokenizer(x[2]) ))
    numberOftweets = RDD.count()
    dictionary = get_dict(keyAndListOfWords)
    allDocsWithLabel = get_tfArray(keyAndListOfWords)
    tf_idfArray = get_tf_idfArray(allDocsWithLabel, numberOftweets)
    training, testing = tf_idfArray.randomSplit([0.8, 0.2], seed=123)   
    
    for optimizer in ['SGD','Momentum','Nesterov','Adam', 'Adagrad' ,'RMSprop' ]:
        print("\n" +"-"*15 + " "+ optimizer +" "+ "-"*15)
        run_save(training,testing,optimizer)

    
    sc.stop()

