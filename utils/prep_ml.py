import re
from pyspark.ml.feature import Tokenizer
from pyspark.sql import functions as f
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import length
from pyspark.sql.types import StructType,StructField, StringType, IntegerType ,ArrayType

def word_tokenizer(x):
    """ Function to remove anything besides alphabets and to tokenize the words"""
    #lower case
    x = x.lower()
    #Remove websites and mentions
    x = re.sub(r'(?:\@|https?\://)\S+', '', x)
    # remove all non letter characters
    x = re.sub(r'[^a-zA-Z]', ' ', x).lstrip()
    return x

def stopwords(x):
    """ Function to remove additional stop words"""
    stop_words = ['m','lol','haha','s','ll','ve','tweet','tweeter','blog']
    x = list(filter(lambda w : w not in stop_words, x))
    return x

cleanup = f.udf(lambda x: word_tokenizer(x), StringType())
countTokens = f.udf(lambda x: len(x), IntegerType())
cleanstopwords = f.udf(lambda x: stopwords(x), ArrayType(StringType()))
