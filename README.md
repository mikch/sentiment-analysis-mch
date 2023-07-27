# Sentiment Analysis MCh

![GitHub Logo](apache_spark_logo.png)


## Description
Sentiment Analysis MCh is an algorithm for academic purposes that is written in Python and runs on a Spark cluster. It is designed to analyze one-dimensional datasets in CSV format, containing only text data (strings). The program performs data cleaning and sentiment extraction on each record, followed by generating diagrams based on the analyzed data. Additionally, the program includes the training of a machine learning evaluation model.
This project is part of my thesis with the subject "Sentiment Analysis and Big Data Management using Apache Spark".


## Installation
1) The program requires a spark on hadoop installation.
   The versions that I used were the following:
    * Spark: Pre-built for Apache Hadoop 3.3 and later
      [https://spark.apache.org/downloads.html]
    * Hadoop: 3.3.3
      [https://hadoop.apache.org/releases.html]

3) It also requires the following python packages to be installed on all cluster nodes in order to run:
   

      |               |
      |---------------|
      | os            | 
      | pyspark       |
      | numpy         | 
      | textblob      |
      | wordcloud     |
      | re            |
      | matplotlib    |
      | seaborn       |
      | nltk          |
      | collections   |
      | sklearn       |
      | langdetect    |
      | translators   |
      | string        |
      | pickle        |
      | io            |
      | hdfs          |

## Usage
To run the program, follow these steps:
1. Prepare your dataset in CSV format with a single column containing text data.
2. Ensure you have a Spark cluster set up and running with access to hdfs.
3. Submit the program to the cluster using the following spark-submit command:
  > spark-submit --master [cluster-manager] --deploy-mode [mode] sentiment_analysis.py 


## Contributing
This project is currently maintained by me.
(https://github.com/mikch)


## License
The program is licensed under the Apache License 2.0


## Contact
For any questions or ideas regarding the project, you can reach me at m.xoust1992@gmail.com
