from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, min as min_,  split, monotonically_increasing_id, slice as slice_, to_date, max as max_
from pyspark.sql.types import StringType, ArrayType
import os

# see https://stackoverflow.com/questions/42051184/latent-dirichlet-allocation-lda-in-spark

print(os.getcwd())
spark = SparkSession.builder.config("spark.worker.cleanup.enabled", "true") .config("spark.worker.cleanup.interval", 60) .getOrCreate() 
df = spark.read.json("/user/hadoop/cleaned_tweets/")
# df = df.sample(0.5)
df = df.withColumn("dt", to_date("created_at", "EEE MMM dd HH:mm:ss +SSSS yyy")) 
df = df.withColumn("id", monotonically_increasing_id())
df = df.withColumn("split_text", split(df.cleaned_text, " "))
cv = CountVectorizer(inputCol="split_text", outputCol="raw_features") 
cvmodel = cv.fit(df)
vocab = cvmodel.vocabulary
result_cv = cvmodel.transform(df)
# online optimizer is critical, algorithm doesn't converge with em optimizer :(
lda = LDA(featuresCol = "raw_features", k=5, optimizer="online")
lda_model = lda.fit(dataset=result_cv)
num_words_per_topic = 50
# default is 10 terms per topic
described = lda_model.describeTopics(maxTermsPerTopic=num_words_per_topic)

def topic_render(topic):
    result = []
    for i in range(len(topic)):
        term = vocab[topic[i]]
        result.append(term)
    return result

topic_render_udf = udf(topic_render, ArrayType(StringType()))

described = described.withColumn("translated", topic_render_udf(described.termIndices))
# described.repartition(1).write.mode("overwrite").json("/user/hadoop/results")

described.select("translated").show()
