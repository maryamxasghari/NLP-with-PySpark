import nltk
import unidecode
from nltk.stem import WordNetLemmatizer,LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 
import numpy as np
import re


newStopWords =['may','would','also','th','one','two','us','to', 'due','new','via','many','much','however','since','although','often','m','s','ll','ve','tweet','tweeter','blog','amp']
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

# def get_dict(keyAndListOfWords):
#     """make a 1000 words dictionary"""
#     allWords = keyAndListOfWords.flatMap(lambda x: (x[2])).map(lambda x: (x, 1))
#     allCounts = allWords.reduceByKey(lambda x,y: x + y) # I will use lambda instead of add maybe it would be faster in spark 
#     topWords = allCounts.top(10000, key= lambda x: x[1])
#     topWordsK = sc.parallelize(range(10000))
#     dictionary = topWordsK.map(lambda x : (topWords[x][0], x))
#     return dictionary


def freqArray (listOfIndices, numberofwords):
    """ function to get TF array provided in the homework"""
    returnVal = np.zeros(10000)

    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

def get_tfArray(keyAndListOfWords,dictionary):
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

def llH(x, coeficients):
    """" LLH - loss function and gradiant """
    theta = np.dot(x[2], coeficients)
    # y = x[1] 1 for AU - 0 for general 
    cost = - int(x[1]) * theta + np.log(1 + np.exp(theta))
    gradient = - x[2] * int(x[1] )+ x[2] * (np.exp(theta) / (1 + np.exp(theta)))
    return cost, gradient

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

def logistic_regression(data,dictionary):
    """logistic regression with balenced-llh"""
    num_iteration = 300
    learning_rate = 0.01
    coef = np.zeros(10000)
    precision = 0.0001
    cost_array = []
    old_cost = 0
    old_coef = coef
    reg_lambda = 1.2 
    l2_old = np.linalg.norm(coef)

    for i in range(num_iteration):
        rdd = data.map(lambda x: (llH(x, coef)))
        result = rdd.reduce(lambda x,y: [x[0] + y[0], x[1] + y[1] ])

        #better results and prof said this approch with small reg_lambda will give a good results
        gradient = result[1] + 2 * reg_lambda * coef
        cost = result[0] +  reg_lambda * np.sum(coef**2)

        coef = coef - learning_rate * gradient

        #bold driver
        if cost < old_cost:
            learning_rate = learning_rate * 1.05
        else:
            learning_rate = learning_rate * 0.5


        cost_array.append(cost)
#         Stop if the cost is not descreasing
#         diff_l2 = abs(np.linalg.norm(coef) - np.linalg.norm(old_coef))
        if( (abs(old_cost-cost) <= precision) ):
            print("Stoped at iteration", i )
            break

        old_coef = coef
        old_cost = cost
        print('Iteration', i , ', Cost =', cost ) #, "diff in the l2 norm of coef :" ,diff_l2 )

    #argpartition will return the index of the 5 large values
    Top5Pos = top5coeff(coef)
    print('\n*******************************************************************************************\n')
    print('The top 5 words with largest coefficients:')
    top5_words = dictionary.filter(lambda x: x[1] in Top5Pos).map(lambda x: x[0]).collect()
    print(top5_words)
    print('\n*******************************************************************************************\n')
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