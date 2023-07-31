################################### Sentiment Analysis MCh ####################################
###############################################################################################
############ This program was created in order to showcase the big data processing ############
############ capabilities of Spark with the use of Hadoop's Distributed            ############
############ File System and some powerfull Python libraries.                      ############
############ Author: https://github.com/mikch                                      ############
###############################################################################################

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


import pyspark
import pyspark.pandas as pd

import numpy as np

from textblob import TextBlob
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import nltk
from nltk import WordNetLemmatizer


from collections.abc import Iterable

import sklearn.utils as utils
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

from langdetect import DetectorFactory, detect
import translators as ts

import string

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import *

import warnings
warnings.filterwarnings('ignore')

import pickle
import io
from hdfs import InsecureClient

#---------------------------------------------------------------------------------------------------------



# A set with all the English stopwords
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'didnt', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', 'shes', 'should', 'shouldve', 'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', 'youd', 'youll', 'youre',
             'youve', 'your', 'yours', 'yourself', 'yourselves', 'the', 'no', 'not', 'but']




# Function for data cleaning
def cleanData(text):
    text = re.sub('@[A-Za-z0-9_]+', '', text) 
    text = re.sub('[0-9]+', '', text) 
    text = re.sub(r'"', '', text) 
    text = "".join(a for a in text if a not in "'")
    text = re.sub(r'(?<=[a-zA-Z])/(?=[a-zA-Z])', ' \g<0> ', text) 
    text = re.sub(r'(?<=[a-zA-Z])_(?=[a-zA-Z])', ' \g<0> ', text) 
    text = re.sub(r'(.)1+', r'1', text) 
    text = re.sub('#','',text) 
    text = re.sub('RT[\s]+','',text) 
    text = re.sub('https?:\/\/\S+', '', text) 
    text = re.sub('\n',' ',text) 
    text = text.strip() 
    return text


# Function for language identification
def langDetect(data):
    translation = ts.google(data)
    return translation
   
# Function for text formatting
def formatData(text):
    text =  text.translate(translator)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text


# Function for word lemmatizing
def lemmatizeData(text):
    newtext = [lm.lemmatize(word) for word in text.split()]
    return newtext


# Function for flattening a list
def flatten(list):
    for item in list:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item
 


# Function for subjectivity calculation
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Function for polarity calculation
def getPolarity(text):
    return TextBlob(text).sentiment.polarity



# Creates a function that checks the negative, neutral and positive results
def getAnalysis(score):
    if score<0:
        return 'Negative'
    elif score ==0:
        return 'Neutral'
    else:
        return 'Positive'



# Function for wordcloud creation
def create_wordcloud(text):    
    allWords = ' '.join([data for data in text])
    wordCloud = WordCloud(background_color='white', width=800, height=500, random_state=21, max_font_size=130).generate(allWords)
    plt.figure(figsize=(20,10))
    plt.imshow(wordCloud)
    plt.axis('off')
    global local_check
    if "local" not in local_check:
        plt.savefig(filename + '_wordcloud.png', bbox_inches='tight')
        client.upload(dir, filename + '_wordcloud.png', overwrite=True)
    else:
        plt.savefig(ldir + filename + '_wordcloud.png', bbox_inches='tight')



# Function for evaluation of Bernoulli Naive Bayes model
def model_Evaluate(model):
    y_pred = model.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    global local_check
    if "local" not in local_check:
        plt.savefig(filename + '_model_evaluation.png', bbox_inches='tight')
        client.upload(dir, filename + '_model_evaluation.png', overwrite=True)
    else:
        plt.savefig(ldir + filename + '_model_evaluation.png', bbox_inches='tight')




#---------------------------------------------------------------------------------------------------------



if __name__ == "__main__":


    # Cache memory settings
    cache_settings = pyspark.StorageLevel.MEMORY_AND_DISK_2 


   	# SparkSession creation
    spark = SparkSession.builder.appName('sentiment_analysis').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # Hadoop user name
    hadoopuser = "hadoopuser"
    # The address of the cluster's master node
    hdfs_root = "hdfs://master:9000"
    # The name of the csv file that will be loaded (without the file extension)
    filename = "dataset_name"
    # The name of the csv file's path
    dir = "/" + filename + "_files/" # The full hdfs path (/{$HADOOPUSER}/...)
    # The name of the pkl file in which the data from the evaluation model will be stored (without the file extension)
    modelname = "BNBmodel" 
    
    # A client connection with the hdfs master node
    client = InsecureClient('http://master:9870')

    # Checks if the program is running on cluster mode
    local_check = sc.master
    if "local" not in local_check:
        print("Spark application is running in cluster mode")        
        model_path = "/trained_models/" #"user/" + hadoopuser + "/trained_models/" # The hdfs path where the data from the training model will reside   
        model_full_path = hdfs_root + model_path + modelname + ".pkl" # The full hdfs path where the data from the training model will reside (including the master node URL)
        # Enables the hdfs API for Pyspark
        hdfs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())

        # Creates a directory for the files that are going to be extracted on the hdfs
        if not hdfs.exists(spark._jvm.org.apache.hadoop.fs.Path(model_path)):
            client.makedirs(model_path, permission="777")
        if not hdfs.exists(spark._jvm.org.apache.hadoop.fs.Path(dir)):
            client.makedirs(dir, permission="777")

        nltk.download('omw-1.4', download_dir='/usr/lib/nltk_data')

    else:
        print("Spark application is running in local mode")
        ldir = "./" + filename + "_results/" # The directory that will be used for the execution in local mode
        model_full_path = "./" + modelname + ".pkl"
        
        # Creates a directory for the files that are going to be extracted locally
        if not os.path.exists(ldir):       
            os.mkdir(ldir)

        nltk.download('omw-1.4')



    STOPWORDS = set(stopwordlist)

    lm = nltk.WordNetLemmatizer()

    punctuations_list = string.punctuation

    # Substracts all characters in the punctiation list from the string
    translator = str.maketrans('', '', punctuations_list)


   
    # Sets the seed value to 0, initializing the random number generator used by the language detector with a constant value. 
    DetectorFactory.seed = 0


	# Loads the csv file to a pyspark dataframe
    data_to_df = spark.read.option("header", True).csv(dir + filename + ".csv")

    # The name of the dataframe header is saved separately.
    header = data_to_df.columns[0]


    # Renames the header to "rawData".
    data_to_df = data_to_df.withColumnRenamed(header, "rawData")
    # Removes any rows from the DataFrame where the value in the "rawData" column is equal to the value stored in the variable "header"
    data_to_df = data_to_df.filter(data_to_df.rawData != header)
    # Filters the null values
    data_to_df = data_to_df.filter(data_to_df.rawData.isNotNull())


    # Removes duplicate records and saves the dataframe to cache memory
    data_to_df = data_to_df.distinct().persist(cache_settings)




    data_to_df.show(5)



    # The function "cleanData" is applied to each row of the data, creating a new column named "cleanedData." The values in the "cleanedData" column are of type String
    cleaned = F.udf(lambda q: cleanData(q), StringType())
    data_to_df = data_to_df.withColumn("cleanedData", cleaned(F.col("rawData")))


    data_to_df.show(5)


    # EXPERIMENTAL USE
    # Checks if the Pyspark dataframe rows are not more than 1000
    if data_to_df.count() <= 1000:

        
        try:
           
            # A very small number of records is selected from "data_to_df" (specifically the first 5) for testing using the Google Translate API
            data_to_df2 = data_to_df.head(5).persist(cache_settings)
            data_to_df2 = spark.createDataFrame(data_to_df2)

            print("Data for testing: ")
            data_to_df2.show()

            # The function "langDetect" is applied to the PySpark DataFrame "data_to_df2," creating a new column named "cleanedData_new."
            translated = F.udf(lambda q: langDetect(q), StringType())
            data_to_df2 = data_to_df2.withColumn("cleanedData_new", translated(F.col("cleanedData")))

            # The column "cleanedData" is dropped, and the "cleanedData_new" is renamed to "cleanedData" and takes its place with the translated data
            data_to_df2 = data_to_df2.drop("cleanedData")
            data_to_df2 = data_to_df2.withColumnRenamed("cleanedData_new", "cleanedData")


            print("Translated data for testing: ")
            data_to_df2.show()

            # The "data_to_df2" is released from cache memory.
            data_to_df2.unpersist()


        except:


            print("Could not apply translation on the dataframe")




    # Filters the cleaned data from "data_to_df" to remove any empty (null) values
    data_to_df = data_to_df.filter(data_to_df.cleanedData.isNotNull())



    # The functions "getSubjectivity" and "getPolarity" are applied to each row of the data, creating two new columns named "Subjectivity" and "Polarity," respectively. The data type of both columns is Float
    subjectivity = F.udf(lambda q: getSubjectivity(q), FloatType())
    polarity = F.udf(lambda q: getPolarity(q), FloatType())


    data_to_df = data_to_df.withColumn("Subjectivity", subjectivity(F.col("cleanedData")))
    data_to_df.show(5)


    data_to_df = data_to_df.withColumn("Polarity", polarity(F.col("cleanedData")))
    data_to_df.show(5)


    data_to_df = data_to_df.drop("rawData")



    # The function "getAnalysis" is applied to each row of the data, creating a new column named "Analysis." The data type of the values in the "Analysis" column is String
    analysis = F.udf(lambda q: getAnalysis(q), StringType())
    data_to_df = data_to_df.withColumn("Analysis", analysis(F.col("Polarity")))

    data_to_df.show(5)

    # Filtering of the data is performed, keeping the records where the value of "Analysis" is "Positive," meaning the positive results are retained
    positiveData = data_to_df.filter(data_to_df.Analysis == "Positive").select(F.col("cleanedData"))

    # Calculation of the percentage (%) of the total positive results with respect to the entire dataset
    percentage = round((positiveData.count() / data_to_df.count()) * 100, 1)
    positiveData = str(percentage) + "%"

    # Filtering of the data is performed, keeping the records where the value of "Analysis" is "Negative," meaning the positive results are retained
    negativeData = data_to_df.filter(data_to_df.Analysis == "Negative").select(F.col("cleanedData"))

    # Calculation of the percentage (%) of the total negative results with respect to the entire dataset
    percentage = round((negativeData.count() / data_to_df.count()) * 100, 1)
    negativeData = str(percentage) + "%"


    # The function "formatData" is applied to format and prepare the text for lemmatization, and it creates a new column named "formattedData" with the processed text
    formatted = F.udf(lambda q: formatData(q))
    data_to_df = data_to_df.withColumn("formattedData", formatted(F.col("cleanedData")))

    # From the "formattedData" column of "data_to_df," an RDD (Resilient Distributed Dataset) is created, which consists of a list of lemmatized words
    data_to_rdd = data_to_df.select("formattedData").rdd.flatMap(lambda list:[lemmatizeData(item) for item in list])
    # Transforms the RDD into a python list
    wordlist = data_to_rdd.mapPartitions(lambda list:flatten(list)).collect()

    data_to_df = data_to_df.drop("formattedData")


    # Creates a PySpark DataFrame named "train_df" using the columns "cleanedData" and "Polarity," where the value of "Polarity" is different from 0, and save it in a new column named "text"
    train_df = data_to_df.select(data_to_df["cleanedData"], data_to_df["Polarity"]) \
            .filter(data_to_df.Polarity != 0.0) \
            .withColumnRenamed("cleanedData", "text") \
            .persist(cache_settings)

    # Creates a new column in "train_df" named "target."
    # where the values of the "Polarity" column are greater than 0, the new value in the "target" column becomes 1.
    # Otherwise, the value becomes 0.
    # The data type of the values in the "target" column is Integer.
    train_df = train_df.withColumn("target", \
                        F.when(train_df["Polarity"] > 0.0, 1).otherwise(0).cast(IntegerType()))


    train_df = train_df.drop("Polarity")
   


#-----------------------------------------------------------------------------------------------------------



    # Converts the PySpark DataFrame to a Pandas DataFrame to create the result analysis files.
    data_pdf = data_to_df.toPandas()
    train_pdf = train_df.toPandas()



    # Releases the PySpark DataFrames that were stored in the cache memory
    data_to_df.unpersist()
    data_to_rdd.unpersist()
    train_df.unpersist()



    # Designs a bar chart to display the number of data emotion
    fig = plt.figure(figsize=(7,5))
    xlabel = ['Positive','Negative','Neutral']
    plt.bar(xlabel,data_pdf['Analysis'].value_counts())
    data_pdf['Analysis'].value_counts().plot(kind='bar')
    plt.title('Value count of data polarity')
    plt.ylabel('Count')
    plt.xlabel('Polarity')
    plt.grid(False)
    if "local" not in local_check:
        plt.savefig(filename + '_sentiment_count.png', bbox_inches='tight')
        client.upload(dir, filename + '_sentiment_count.png', overwrite=True)
    else:
        plt.savefig(ldir + filename + '_sentiment_count.png', bbox_inches='tight')



    # Designs a bar chart to display the number of data polarity
    fig = plt.figure(figsize=(7,7))
    colors = ('green', 'grey', 'red')
    wp={'linewidth':2, 'edgecolor': 'black'}
    tags=data_pdf['Analysis'].value_counts()
    explode = (0.1,0.1,0.1)
    tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode, label='')
    plt.title('Distribution of polarity')
    if "local" not in local_check:
        plt.savefig(filename + '_sentiment_percentage.png', bbox_inches='tight')
        client.upload(dir, filename + '_sentiment_percentage.png', overwrite=True)
    else:
        plt.savefig(ldir + filename + '_sentiment_percentage.png', bbox_inches='tight')



    # Designs the polarity and subjectivity in a scatter plot diagram
    plt.figure(figsize=(9,7))
    for i in range(0,data_pdf.shape[0]):
        plt.scatter(data_pdf['Polarity'][i],data_pdf['Subjectivity'][i], color='blue')
    plt.title('Sentiment Analysis')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    if "local" not in local_check:
        plt.savefig(filename + '_polarity_subjectivity.png', bbox_inches='tight')
        client.upload(dir, filename + '_polarity_subjectivity.png', overwrite=True)
    else:
        plt.savefig(ldir + filename + '_polarity_subjectivity.png', bbox_inches='tight')



    # Designs a wordcloud for the positive results
    posData = data_pdf.loc[data_pdf['Analysis']=='Positive', 'cleanedData']
    create_wordcloud(posData)


    # Designs a wordcloud for the negative results
    negData = data_pdf.loc[data_pdf['Analysis']=='Negative', 'cleanedData']
    create_wordcloud(negData)



    # The resulting Series object contains the frequency count of each word in the original input data of the word list
    lem = pd.DataFrame(wordlist)
    lem = lem[0].value_counts()



    # Designs the plot for the 10 most frequently used words
    lem = lem[:10]
    plt.figure(figsize=(10,5))
    plt.plot(lem.index, lem.values)
    plt.title('Top Words Overall')
    plt.xlabel('Count of words', fontsize=12)
    plt.ylabel('Words from sentences', fontsize=12)
    if "local" not in local_check:
        plt.savefig(filename + '_top_words.png', bbox_inches='tight')
        client.upload(dir, filename + '_top_words.png', overwrite=True)
    else:
        plt.savefig(ldir + filename + '_top_words.png', bbox_inches='tight')




    # EXPERIMENTAL USE
    # Checks if the "train_pdf" dataframe rows are more than 200
    if len(train_pdf.index) > 200:


        # The input features and the target are separated
        X=train_pdf.text
        y=train_pdf.target


        # The data is split, with 95% used for training and 5% for testing
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)



        # Creates an instance of the TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer and then fits the training data into it
        vectorizer = TfidfVectorizer(use_idf=True, max_features=500000)
        vectorizer.fit(X_train)



        # Transforms the data using the TF-IDF Vectorizer
        X_train = vectorizer.transform(X_train)
        X_test  = vectorizer.transform(X_test)


        # Checking the format of the labels on the y-axis
        if y_train.dtype != 'int' or y_train.dtype != 'float':
            # Encodes the labels into numerical format
            label_encoder = preprocessing.LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test  = label_encoder.fit_transform(y_test)
            print("Labels have been converted to numerical format using label encoding")
        

        feature_words = len(vectorizer.get_feature_names_out()) # Calculates the number of unique words used by the vectorizer



        if "local" not in local_check:

            # Checks if the evaluation model file exists at the specified location that was set at the beginning of the program
            file_exists = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration()).exists(sc._jvm.org.apache.hadoop.fs.Path(model_full_path))


            if file_exists:
                # The Bernoulli Naive Bayes model stored in the pkl file is loaded into memory
                model_data = spark.sparkContext.binaryFiles(model_full_path).take(1)[0][1]
                model_data = io.BytesIO(model_data)
                BNBmodel = pickle.load(model_data)
            else:
                BNBmodel = BernoulliNB() # If the pkl file is not found, the Bernoulli Naive Bayes model is created from scratch


        
        else:
 
            # Checks if the evaluation model file exists at the specified location that was set at the beginnin of the program
            file_exists = os.path.exists(model_full_path)
            if file_exists:
                 # The Bernoulli Naive Bayes model stored in the pkl file is loaded into memory
                with open(model_full_path, 'rb') as file:
                    model_data = file.read()

                model_data = io.BytesIO(model_data)
                BNBmodel = pickle.load(model_data)
            else:
                BNBmodel = BernoulliNB() # If the pkl file is not found, the Bernoulli Naive Bayes model is created from scratch




        # Application of the Bernoulli Naive Bayes model and training with the data
        BNBmodel = BernoulliNB()
        BNBmodel.fit(X_train, y_train)
        model_Evaluate(BNBmodel)
        y_pred1 = BNBmodel.predict(X_test)




        # Designs the ROC-AUC curve for the model.
        fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC CURVE')
        plt.legend(loc="lower right")
        if "local" not in local_check:
            plt.savefig(filename + '_roc_curve.png', bbox_inches='tight')
            client.upload(dir, filename + '_roc_curve.png', overwrite=True)
        else:
            plt.savefig(ldir + filename + '_roc_curve.png', bbox_inches='tight')



        # Serializes the model to bytes.
        model_bytes = pickle.dumps(BNBmodel)


        # Writes the serialized data to a .pkl file
        if "local" not in local_check:
            output_stream = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration()).create(spark._jvm.org.apache.hadoop.fs.Path(model_full_path), True)
            output_stream.write(model_bytes)
            output_stream.close()
        else:
            with open(model_full_path, 'ab') as file:
                output_stream = pickle.Pickler(file)
                output_stream.dump(model_bytes)

        


        print(f"Trained model saved to:{model_full_path}")


	    

    else:
        feature_words = 0




    # An Excel file is created with the final results
    data = np.array([[data_pdf.shape[0], positiveData, negativeData, feature_words]])
    data_sum = pd.DataFrame(data, columns=['Total Rows', 'Positive Data', 'Negative Data', 'Trained Words'])
    if "local" not in local_check:
        data_sum.to_excel(filename + '_data_sum.xlsx')
        client.upload(dir, filename + '_data_sum.xlsx', overwrite=True)
    else:
        data_sum.to_excel(ldir + filename + '_data_sum.xlsx')



    print('Sentiment Analysis finished.')



