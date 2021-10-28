import sys
import time
import re
from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType ,ArrayType
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.sql import functions as f
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import length
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Trees.py <file> ", file=sys.stderr)
        exit(-1)

    def word_tokenizer(x): 
        """ Function to remove anything besides alphabets and to tokenize the words"""
        #lower case
        x = x.lower()
        #remove websites and mentiones
        x = re.sub(r'(?:\@|https?\://)\S+', '', x)
        # remove all non letter characters
        x = re.sub(r'[^a-zA-Z]', ' ', x).lstrip()
        return x 

    def stopwords(x): 
        """ Function to remove additional stop words"""
        stop_words = ['m','lol','haha','s','ll','ve','tweet','tweeter','blog']
        x = list(filter(lambda w : w not in stop_words, x))
        return x 
      
    # -- Start the session
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    spark = SparkSession.builder.appName('tweet').getOrCreate()        

#     sparkDF = spark.read.csv(sys.argv[1] ,inferSchema=True,header=True)
#     sparkDF= sparkDF.select(f.col("id"),f.col("text"),f.col("target").alias("class"))
   
    # -- reading the data
    tweet = pd.read_csv(sys.argv[1])
    tweet_df= tweet[['id','text','target']]
    
    # -- Spark data frame
    mySchema = StructType([ StructField("id", StringType(), True)\
                       ,StructField("text", StringType(), True)\
                       ,StructField("class", StringType(), True)])
    sparkDF=spark.createDataFrame(tweet_df, mySchema) 
    
    # -- UDF functions for clean up
    cleanup = f.udf(lambda x: word_tokenizer(x), StringType())
    countTokens = f.udf(lambda x: len(x), IntegerType())
    cleanstopwords = f.udf(lambda x: stopwords(x), ArrayType(StringType()))
    
    #-- Using the UDF functions
    sparkDF = sparkDF.withColumn("clean", cleanup(f.col("text")))
    
    #-- Using Ml libraries
    sparkDF = sparkDF.withColumn('length', length(sparkDF['clean']))
    
    #-- Using Ml libraries
    tokenizer = Tokenizer(inputCol="clean", outputCol="token_text")
    tokenized = tokenizer.transform(sparkDF)
    
    #-- Using the UDF functions
    tokenized_df= tokenized.withColumn("tokens", countTokens(f.col("token_text")))
    
    #-- Using Ml libraries
    remover = StopWordsRemover(inputCol="token_text", outputCol="filtered")
    tokenized_df = remover.transform(tokenized_df)
    
    #-- Using the UDF functions
    tokenized_df  = tokenized_df.withColumn("filteredmore", cleanstopwords(f.col("filtered")))
    tokenized_df= tokenized_df.withColumn("filtered_tokens", countTokens(f.col("filteredmore")))
    
    #-- Using Ml libraries - Pipeline
    count_vec = CountVectorizer(inputCol='filteredmore',outputCol='c_vec')
    idf = IDF(inputCol="c_vec", outputCol="tf_idf")
    tweet_class = StringIndexer(inputCol='class',outputCol='label')
    clean_up = VectorAssembler(inputCols=['tf_idf','filtered_tokens'],outputCol='features')

    data_prep_pipe = Pipeline(stages=[tweet_class,count_vec,idf,clean_up])
    prep = data_prep_pipe.fit(tokenized_df)
    dataset = prep.transform(tokenized_df)
    
    #-- Split into train and test
    dataset = dataset.select(['label','features'])
    (training,testing) = dataset.randomSplit([0.8,0.2])


    # DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
    dtc = DecisionTreeClassifier(labelCol='label',featuresCol='features')
    rfc = RandomForestClassifier(labelCol='label',featuresCol='features')
    gbt = GBTClassifier(labelCol='label',featuresCol='features')
    
    train_start = time.time()
    dtc_model = dtc.fit(training)    
    train_end = time.time()
    train_time = train_end - train_start
    
    test_start = time.time()
    dtc_predictions = dtc_model.transform(testing)
    test_end = time.time()
    test_time = test_end - test_start
    
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label')
    
    acc = evaluator.evaluate(dtc_predictions, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(dtc_predictions, {evaluator.metricName: "f1"})

    Start2end =train_time +test_time   
    
    print("\n============== DecisionTree ==============\n")
    print("Accuracy of model was: %{}\n".format(round(acc, 4)*100))
    print("f1 score of model was: %{}\n".format(round(f1, 3)*100))

    print(f'- Train time = {round(train_time,4)}\n')
    print(f'- Test time = {round(test_time,4)}\n')
    print(f'- Total time = {round(Start2end,4)}\n')

    train_start = time.time()
    rfc_model = rfc.fit(training)    
    train_end = time.time()
    train_time = train_end - train_start
    
    test_start = time.time()
    rfc_predictions = rfc_model.transform(testing)
    test_end = time.time()
    test_time = test_end - test_start
    
    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label')
    
    acc = evaluator.evaluate(rfc_predictions, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(rfc_predictions, {evaluator.metricName: "f1"})

    Start2end =train_time +test_time   
    
    print("\n============== Random Forest ==============\n")
    print("Accuracy of model was: %{}\n".format(round(acc, 4)*100))
    print("f1 score of model was: %{}\n".format(round(f1, 3)*100))

    print(f'- Train time = {round(train_time,4)}\n')
    print(f'- Test time = {round(test_time,4)}\n')
    print(f'- Total time = {round(Start2end,4)}\n')    
    
    train_start = time.time()
    gbt_model = gbt.fit(training)    
    train_end = time.time()
    train_time = train_end - train_start
    
    test_start = time.time()
    gbt_predictions = gbt_model.transform(testing)
    test_end = time.time()
    test_time = test_end - test_start

    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='label')
    
    acc = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "f1"})

    Start2end =train_time +test_time   
    
    print("\n============== Random Forest ==============\n")
    print("Accuracy of model was: %{}\n".format(round(acc, 4)*100))
    print("f1 score of model was: %{}\n".format(round(f1, 3)*100))

    print(f'- Train time = {round(train_time,4)}\n')
    print(f'- Test time = {round(test_time,4)}\n')
    print(f'- Total time = {round(Start2end,4)}\n')      
    
    sc.stop()
