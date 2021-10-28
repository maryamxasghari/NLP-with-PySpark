
import nltk
import unidecode
from nltk.stem import WordNetLemmatizer,LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 
import numpy as np
import re

def freqArray_1000 (listOfIndices, numberofwords):
    """ function to get TF array provided in the homework"""
    returnVal = np.zeros(1000)
    for index in listOfIndices:
        returnVal[index] = returnVal[index] + 1
    returnVal = np.divide(returnVal, numberofwords)
    return returnVal

def get_tfArray_1000(keyAndListOfWords,dictionary):
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
    allDocsWithLabel = allDictionaryWordsInEachDoc.map(lambda x: (x[0][0],x[0][1], freqArray_1000(x[1], x[0][2])))
    return allDocsWithLabel

def get_tf_idfArray_1000(tf_array, numberOfDocs):
    """Get tf_idfArray with true labels"""
    zeroOrOne = tf_array.map(lambda x: (x[0],x[1], np.where(x[2] != 0, 1, 0) ) )
    dfArray = zeroOrOne.reduce(lambda x, y: ("","", np.add(x[2], y[2])))[2]
    idfArray = np.log(np.divide(np.full(1000, numberOfDocs), dfArray))
    allDocsAsNumpyArraysTFidf = tf_array.map(lambda x: (x[0],x[1], np.multiply(x[2], idfArray)))
    return allDocsAsNumpyArraysTFidf



def onehotlabel(x):
    if int(x) == 1 :
        return np.array([0,1])
    else:
        return np.array([1,0])


def forward(x,w,b):
    return np.dot(x, w) + b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Lrelu(x):
    return np.where(x >= 0, x, x*0.01)

def cost(y_hat, y):
        return 0.5 * np.sum(np.power(y_hat - y, 2)) 

def der_sigmoid(x):
    sigm = 1 / (1 + np.exp(-x))
    return sigm *(1- sigm)

def der_Lrelu(x):
    return np.where(x >= 0, 1, 0.01)

def dEB2(y_pred, y_true, y_h):
    return (y_pred - y_true) * der_sigmoid(y_h)

def dEW2(h, dB2):
    return np.dot(h.T, dB2)

def dEB1(h_h, dB2, W2):
    return np.dot(dB2, W2.T) * der_Lrelu(h_h)

def dEW1(x, dB1):
    return np.dot(x.T, dB1)


def optimizer_adam(w ,b ,DW ,DB ,\
                   m_w,m_b ,\
                   v_w, v_b,\
                   i , learning_rate):
    epsilon = 10e-8
    beta1 = 0.9
    beta2 = 0.999  
    
    # momentum _ beta1
    # -- Weights --
    m_w = beta1 * m_w + (1 - beta1) *DW
    # -- biases --
    m_b = beta1 * m_b + (1 - beta1) *DB
    
    # rms _ beta2
    # -- Weights --   
    v_w = beta2 * v_w + (1 - beta2) * (DW**2)
    # -- biases --
    v_b = beta2 * v_b + (1 - beta2) * (DB)
    
    # correction
    # -- Weights -- 
    m_w_ = m_w / (1 - beta1**(i + 1))
    v_w_ = v_w / (1 - beta2**(i + 1))
    # -- biases --
    m_b_ = m_b / (1 - beta1**(i + 1))
    v_b_ = v_b / (1 - beta2**(i + 1))
    
    #Update weights
    w = w - learning_rate * m_w_ / (np.sqrt(abs(v_w_)) + epsilon)
    b = b - learning_rate * m_b_ / (np.sqrt(abs(v_b_)) + epsilon)
    return w , b ,m_w,m_b, v_w, v_b


def learning(num_iteration ,\
             training ,\
             learning_rate,\
             optimizer  , reg):
    num_iteration = 300
    cost_array = []
    old_cost = 0
    n = training.count()
    reg_lambda = 0.001
    epsilon = 10e-8
    beta1 = 0.9
    beta2 = 0.999    
    m_w1 , m_b1 = np.zeros((1000,128)),np.zeros((1,128))
    m_w2 , m_b2 = np.zeros((128,2)),np.zeros((1,2))
    v_w1 , v_b1 = np.zeros((1000,128)),np.zeros((1,128))
    v_w2 , v_b2 = np.zeros((128,2)),np.zeros((1,2))
    
    
    w1 =np.random.uniform(low = -0.5 , high = 0.5 , size = (1000,128))  
    w2 =np.random.uniform(low = -0.5 , high = 0.5 , size = (128,2)) 
    b1 =np.random.rand( 1,128)-0.5 
    b2 =np.random.rand(1,2)-0.5


    for i in range(num_iteration):
        Cost_grad = training.map(lambda x: (x[1],x[2]))\
            .map(lambda x: (onehotlabel(x[0]), x[1] ))\
            .map(lambda x: (x[0],x[1],forward(x[1],w1,b1)))\
            .map(lambda x: (x[0],x[1], x[2], Lrelu(x[2])))\
            .map(lambda x: (x[0],x[1], x[2] ,x[3], forward(x[3],w2,b2)))\
            .map(lambda x: (x[0],x[1], x[2] ,x[3], x[4], sigmoid(x[4])))\
            .map(lambda x: (x[0],x[1], x[2] ,x[3], x[4], x[5] ,cost(x[5], x[0])))\
            .map(lambda x: (x[0],x[1], x[2] ,x[3], x[6], dEB2(x[5], x[0] , x[4])))\
            .map(lambda x: (x[1],x[2], x[4], x[5], dEW2(x[3], x[5])))\
            .map(lambda x: (x[0],x[2], x[3], x[4], dEB1(x[1], x[3], w2 )))\
            .map(lambda x: (x[1], x[2],x[3],x[4], dEW1(x[1], x[4] )))\
            .reduce(lambda x,y : (x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3],x[4]+y[4] ))

        Cost ,DB2,DW2,DB1,DW1 = Cost_grad
        Cost = Cost/n
        DB2,DW2,DB1,DW1 = DB2/n,DW2/n,DB1/n,DW1/n
        
#         if reg == True:
#             DB2 = DB2 + reg_lambda * b2
#             DW2 = DW2 + reg_lambda * w2
#             DB1 = DB1 + reg_lambda * b1
#             DW1 = DW1 + reg_lambda * w1        
        
        if optimizer =='SGD' :
            b2 = b2 - learning_rate * DB2
            w2 = w2 - learning_rate * DW2
            b1 = b1 - learning_rate * DB1
            w1 = w1 - learning_rate * DW1

        if optimizer =='Adam' :

            w1 ,b1 ,m_w1,m_b1, v_w1, v_b1=  optimizer_adam(w1, b1, DW1 ,\
                                                             DB1 ,\
                                                             m_w1 ,\
                                                             m_b1 ,\
                                                             v_w1,\
                                                             v_b1, i ,\
                                                             learning_rate)

            w2 ,b2 ,m_w2,m_b2, v_w2, v_b2=  optimizer_adam(w2, b2, DW2 ,\
                                                             DB2 ,\
                                                             m_w2 ,\
                                                             m_b2 ,\
                                                             v_w2,\
                                                             v_b2, i ,\
                                                             learning_rate)
        
        #bold driver
        if Cost < old_cost:
            learning_rate = learning_rate * 1.05
        else:
            learning_rate = learning_rate * 0.5


        cost_array.append(Cost)

        old_cost = Cost
        if i%20 == 0 :
            print("Iteration No.", i, " Cost=", Cost)
#         print('Iteration', i , ', Cost =', Cost )  

    return cost_array , w1, b1, w2, b2

def print_result(test_pred):

    Pos_Neg = test_pred.map(lambda x :  t_f_Pos_Neg(x))
    tp,tn,fp,fn = Pos_Neg.map(lambda x : (x[1],x[2],x[3],x[4])).reduce( lambda x,y :(x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))

    acc = (tp+tn) / (tp+tn+fp+fn)
    f1 = tp / (tp + 0.5*(fp+fn))

    print("\n============== Results: ==============\n")
    print("Accuracy of model was: %{0:2.2f}\n".format(acc*100))
    print("f1 score of model was: %{0:2.2f}\n".format(f1*100))