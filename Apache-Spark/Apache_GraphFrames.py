# Databricks notebook source
from pyspark.sql import SparkSession
from graphframes import *
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, BooleanType

# COMMAND ----------


sc = StructType([StructField('Id', IntegerType(), True),
                     StructField('ProductId', StringType(), True),
                     StructField('UserId', StringType(), True),
                     StructField('ProfileName', StringType(), True),
                     StructField('HelpfulnessNumerator', IntegerType(), True),                         
                     StructField('HelpfulnessDenominator', IntegerType(), True),       
                     StructField('Score', IntegerType(), True),       
                     StructField('Time', IntegerType(), True),                 
                     StructField('Summary', StringType(), True),
                     StructField('Text', StringType(), True) ])
                                 

# COMMAND ----------

data = spark.read.csv('/FileStore/tables/', header=True, schema=sc)
# spark_df.cache() # Cache data for faster reuse
data = data.dropna() # drop rows with missing values
data.dropna()

# Register table so it is accessible via SQL Context
# For Apache Spark = 2.0
data.createOrReplaceTempView("reviews_csv")

# COMMAND ----------

product = sqlContext.sql("SELECT * FROM reviews_csv")
customers=sqlContext.sql("SELECT * FROM vertex1_csv")

# COMMAND ----------

customers.dropDuplicates().show()

# COMMAND ----------

product.printSchema()

# COMMAND ----------

customers.printSchema()

# COMMAND ----------

data_Vertices = customers.withColumnRenamed("ProductId", "id").distinct()

data_Edges = product.withColumnRenamed("ProductId", "src")
data_Edges = data_Edges.withColumnRenamed("Score" , "dst")

# COMMAND ----------

g = GraphFrame(data_Vertices, data_Edges)
print (g)


# COMMAND ----------

print("Total Number of products: ")
print(g.vertices.count())
print("Scores of each product " )
print(g.edges.count())


# COMMAND ----------

display(g.vertices)

# COMMAND ----------

display(g.edges)

# COMMAND ----------

product.show()

# COMMAND ----------

display(g.inDegrees)

# COMMAND ----------

display(g.outDegrees)

# COMMAND ----------

display(g.degrees)

# COMMAND ----------

display(g.edges.filter("dst = '4' and HelpfulnessDenominator > 60"))

# COMMAND ----------

result = g.stronglyConnectedComponents(maxIter = 10)
display(result.select("id", "Component"))

# COMMAND ----------

ranks = g.pageRank(resetProbability= 0.15, maxIter = 5)
display(ranks.vertices.orderBy(ranks.vertices.pagerank.desc()).limit(100))
# display(ranks.vertices.orderBy(desc("pagerank")))

# COMMAND ----------

from pyspark.sql.functions import desc
display(g.edges.filter("Score = '4'").groupBy("src", "dst").avg("HelpfulnessDenominator").sort(desc("avg(HelpfulnessDenominator)")).limit(100))

# COMMAND ----------


