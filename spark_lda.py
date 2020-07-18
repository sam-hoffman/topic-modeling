from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, split, monotonically_increasing_id
from pyspark.sql.types import StringType, ArrayType

# see https://stackoverflow.com/questions/42051184/latent-dirichlet-allocation-lda-in-spark

spark = SparkSession.builder.config("spark.worker.cleanup.enabled", "true") .config("spark.worker.cleanup.interval", 60) .getOrCreate() 
df = spark.read.json("cleaned_tweets")
df = df.sample(0.05)
df = df.withColumn("id", monotonically_increasing_id())
df = df.withColumn("split_text", split(df.cleaned_text, " "))
cv = CountVectorizer(inputCol="split_text", outputCol="raw_features", vocabSize=2**18, maxDF=0.2)
cvmodel = cv.fit(df)
vocab = cvmodel.vocabulary
result_cv = cvmodel.transform(df)
lda = LDA(featuresCol = "raw_features", k=50, seed=1, optimizer="em")
lda_model = lda.fit(dataset=result_cv)
num_words_per_topic = 20
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
described.toPandas().to_json("lda_results.json", orient="records")




