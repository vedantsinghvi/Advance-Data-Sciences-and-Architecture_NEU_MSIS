# Databricks notebook source
import pandas as pd
import numpy as np

# COMMAND ----------

spark

# COMMAND ----------


df = sqlContext.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/Reviews.csv')

display(df)


# COMMAND ----------

# MAGIC %sql show tables

# COMMAND ----------

display(df.limit(5))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM reviews_csv limit 5

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   Score, 
# MAGIC   Summary,
# MAGIC   Text,
# MAGIC  
# MAGIC  HelpfulnessDenominator as VotesTotal
# MAGIC FROM reviews_csv 
# MAGIC WHERE not(score = 0 or score = 1)
# MAGIC order by score

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT 'Good' REGEXP '.*';

# COMMAND ----------

# MAGIC %sql
# MAGIC SElect ProfileName, Score, Summary, Text from reviews_csv where Text like '_good%' order by ProfileName

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(Distinct ProfileName) from reviews_csv

# COMMAND ----------

# MAGIC %sql
# MAGIC select ProfileName, first(Score) from reviews_csv where score=5
# MAGIC group by ProfileName

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(Distinct ProfileName) from reviews_csv

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   avg(score) as scoretype
# MAGIC  FROM reviews_csv 
# MAGIC WHERE score = 5 or score = 4

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC case when Score > 3 then 'positive' else 'negative' end as scoretype,  ProfileName, Score
# MAGIC from reviews_csv
# MAGIC group by ProfileName, score

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(score) from amazon2_csv group by score

# COMMAND ----------

df.head(5)
