
from pyspark.sql import SparkSession
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, FloatType, IntegerType, LongType,StructField
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import when, col
import pandas as pd
import seaborn as sns
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import lit
from IPython.display import display
from pyspark.sql.functions import split, explode
from pyspark.sql.functions import avg

pd.options.display.max_columns = 20
spark = SparkSession.builder \
        .master("local[6]") \
        .appName("COM6012 Assignment") \
        .config("spark.local.dir","/fastdata/acq22jj/") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("OFF")


raw_data = spark.read.csv('Data/ratings.csv',inferSchema=True,header=True)


raw_data.printSchema()


raw_sorted = raw_data.orderBy('timestamp')


# Get the earliest and latest timestamps
timestamp_col = "timestamp"
earliest_timestamp = raw_sorted.select(col(timestamp_col)).first()[0]
latest_timestamp = raw_sorted.select(col(timestamp_col)).orderBy(col(timestamp_col).desc()).first()[0]

# Split the data based on the timestamps
train_40 = raw_sorted.filter(col(timestamp_col) < earliest_timestamp + (latest_timestamp - earliest_timestamp) * 0.4)
test_60 = raw_sorted.filter(col(timestamp_col) >= earliest_timestamp + (latest_timestamp - earliest_timestamp) * 0.4)

train_60 = raw_sorted.filter(col(timestamp_col) < earliest_timestamp + (latest_timestamp - earliest_timestamp) * 0.6)
test_40 = raw_sorted.filter(col(timestamp_col) >= earliest_timestamp + (latest_timestamp - earliest_timestamp) * 0.6)

train_80 = raw_sorted.filter(col(timestamp_col) < earliest_timestamp + (latest_timestamp - earliest_timestamp) * 0.8)
test_20 = raw_sorted.filter(col(timestamp_col) >= earliest_timestamp + (latest_timestamp - earliest_timestamp) * 0.8)



seed = 6 #seed =06

train_40.cache()
train_60.cache()
train_80.cache()

test_60.cache()
test_40.cache()
test_20.cache()

#-----------------Setting-1 ALS---------------------------------


als_1 = ALS(userCol="userId", itemCol="movieId", seed=seed, coldStartStrategy="drop")
models_1_40 = als_1.fit(train_40)


predictions_1_1 = models_1_40.transform(test_60)
evaluator_1 = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse_1 = evaluator_1.evaluate(predictions_1_1)
print(f"Root-mean-square error for size 40 ALS_1 = " + str(rmse_1))


models_1_60 = als_1.fit(train_60)
models_1_80 = als_1.fit(train_80)

predictions_1_2 = models_1_60.transform(test_40)
evaluator_1 = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse_2 = evaluator_1.evaluate(predictions_1_2)
print(f"Root-mean-square error for size 60 ALS_1 = " + str(rmse_2))
predictions_1_3 = models_1_80.transform(test_20)
evaluator_1 = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse_3 = evaluator_1.evaluate(predictions_1_3)
print(f"Root-mean-square error for size 80 ALS_1= " + str(rmse_3))

als_2 = ALS(userCol="userId", itemCol="movieId", seed=seed, coldStartStrategy="drop",maxIter=20)

models_2_40 = als_2.fit(train_40)
models_2_60 = als_2.fit(train_60)
models_2_80 = als_2.fit(train_80)

predictions_2_1 = models_2_40.transform(test_60)
evaluator_1 = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse1 = evaluator_1.evaluate(predictions_2_1)
print(f"Root-mean-square error for size 40 ALS_2 = " + str(rmse1))

predictions_2_2 = models_2_60.transform(test_40)
evaluator_1 = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse2 = evaluator_1.evaluate(predictions_2_2)
print(f"Root-mean-square error for size 60 ALS_2 = " + str(rmse2))

predictions_2_3 = models_2_80.transform(test_20)
evaluator_1 = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse3 = evaluator_1.evaluate(predictions_2_3)
print(f"Root-mean-square error for size 80 ALS_2= " + str(rmse3))

evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")


eval_list = [evaluator_rmse, evaluator_mse, evaluator_mae]
train_list = [train_40, train_60, train_80]
test_list = [test_60, test_40, test_20]
models = [als_1, als_2]

results = []

for model in models:
    model_name = type(model).__name__
    for i, train_data in enumerate(train_list):
        split_name = ""
        if i == 0:
            split_name = "Split 40:60"
        elif i == 1:
            split_name = "Split 60:40"
        else:
            split_name = "Split 80:20"
        test_data = test_list[i]
        model_fit = model.fit(train_data)
        for evaluator in eval_list:
            metric = evaluator.evaluate(model_fit.transform(test_data).filter(col("prediction").isNotNull()))
            results.append((model_name, evaluator.getMetricName(), split_name, metric))

columns = ["Model", "Metric", "Split", "Score"]
df_results = spark.createDataFrame(results, columns)


data = df_results.toPandas()

# updating the 'Model' column
data.loc[data['Model'] == 'ALS', 'Model'] = ['ALS1', 'ALS1', 'ALS1', 'ALS1', 'ALS1', 'ALS1', 'ALS1', 'ALS1', 'ALS1', 'ALS2', 'ALS2', 'ALS2', 'ALS2', 'ALS2', 'ALS2', 'ALS2', 'ALS2', 'ALS2']

# print the updated dataframe
print('\n')
print(data)
print('\n')

colors = {"rmse": "red", "mse": "green", "mae": "blue"}

# Create the plot
sns.set_style("whitegrid")
g = sns.catplot(data=data, x="Split", y="Score", hue="Metric", col="Model", kind="bar", height=4, aspect=.7, palette=colors)
g.set_axis_labels("Splits", "Errors")
g.set_xticklabels(rotation=45)
g.legend.set_title("")
plt.savefig('Output/3_a_C.png',dpi=400)
plt.clf()


user_factors_split1 = als_2.fit(train_40).userFactors
user_factors_split2 = als_2.fit(train_60).userFactors
user_factors_split3 = als_2.fit(train_80).userFactors


# Get the user factors for each split

# Define a function to cluster the users with k-means and return the top five largest clusters
def get_top_five_clusters(user_factors):
    # Run k-means with k=25
    kmeans = KMeans(k=25, seed=seed)
    model = kmeans.fit(user_factors)

    # Add a cluster ID column to the user factors DataFrame
    user_factors_clustered = model.transform(user_factors)

    # Count the number of users in each cluster
    cluster_sizes = user_factors_clustered.groupBy("prediction").count().orderBy("count", ascending=False)

    # Return the top five largest clusters
    return cluster_sizes.limit(5),user_factors_clustered 

# Cluster the users for each split and get the top five largest clusters
top_five_clusters_split1,user_factors_clustered_1 = get_top_five_clusters(user_factors_split1)
top_five_clusters_split2,user_factors_clustered_2 = get_top_five_clusters(user_factors_split2)
top_five_clusters_split3,user_factors_clustered_3 = get_top_five_clusters(user_factors_split3)

# Print the sizes of the top five clusters for each split
print("Top five largest clusters for split 1:")
top_five_clusters_split1.show()
print("Top five largest clusters for split 2:")
top_five_clusters_split2.show()
print("Top five largest clusters for split 3:")
top_five_clusters_split3.show()


clus_pd_1 = top_five_clusters_split1.toPandas()
clus_pd_2 = top_five_clusters_split2.toPandas()
clus_pd_3 = top_five_clusters_split3.toPandas()

df_new = pd.DataFrame()
df_new["Split 1 Prediction"] = clus_pd_1["prediction"]
df_new["Split 1 Count"] = clus_pd_1["count"]
df_new["Split 2 Prediction"] = clus_pd_2["prediction"]
df_new["Split 2 Count"] = clus_pd_2["count"]
df_new["Split 3 Prediction"] = clus_pd_3["prediction"]
df_new["Split 3 Count"] = clus_pd_3["count"]

display(df_new)

from pyspark.sql.functions import lit

# Create a DataFrame with the sizes of the top five clusters for each split
cluster_sizes = (
    top_five_clusters_split1.select("prediction", "count")
    .withColumnRenamed("count", "size")
    .withColumn("split", lit("Split 1"))
    .union(
        top_five_clusters_split2.select("prediction", "count")
        .withColumnRenamed("count", "size")
        .withColumn("split", lit("Split 2"))
    )
    .union(
        top_five_clusters_split3.select("prediction", "count")
        .withColumnRenamed("count", "size")
        .withColumn("split", lit("Split 3"))
    )
)

# Display the sizes of the top five clusters for each split in a table
cluster_sizes.orderBy("split", "prediction").show()


def plot_clusters(top_five_clusters, split_num, ax):
    # Convert the top five clusters DataFrame to a Pandas DataFrame for plotting
    top_five_clusters_pd = top_five_clusters.toPandas()

    # Define a list of colors to use for the clusters
    colors = ["b", "g", "r", "c", "m"]

    # Create a bar chart of the cluster sizes
    ax.bar(top_five_clusters_pd["prediction"], top_five_clusters_pd["count"], color=colors[:len(top_five_clusters_pd)])

    # Set the title and axis labels
    ax.set_title(f"Top Five Largest User Clusters - Split {split_num}")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Users")
    ax.set_xticks(top_five_clusters_pd["prediction"])
    # Create a legend for the cluster colors
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(top_five_clusters_pd))]
    labels = [f"Cluster {i+1}" for i in range(len(top_five_clusters_pd))]
    ax.legend(handles, labels)

# Create a figure with three subplots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# Plot the top five largest clusters for each split on a separate subplot
plot_clusters(top_five_clusters_split1, 1, axs[0])
plot_clusters(top_five_clusters_split2, 2, axs[1])
plot_clusters(top_five_clusters_split3, 3, axs[2])

# Adjust the spacing between subplots
fig.tight_layout()

# Show the plot
plt.savefig("Output/Top_Five_Largest_User_Clusters.png", dpi=400)
plt.clf()


largest_cluster_1 = top_five_clusters_split1.head()
largest_cluster_2 = top_five_clusters_split2.head()
largest_cluster_3 = top_five_clusters_split3.head()

c= user_factors_clustered_1.filter(user_factors_clustered_1.prediction == largest_cluster_1[0]).select("id").collect()
c_1 = set()
for i in range(len(c)):
    c_1.add(c[i][0])


largest_cluster_ratings_1 = train_40.filter(train_40.userId.isin(c_1))

d= user_factors_clustered_2.filter(user_factors_clustered_2.prediction == largest_cluster_2[0]).select("id").collect()
e= user_factors_clustered_3.filter(user_factors_clustered_3.prediction == largest_cluster_3[0]).select("id").collect()
c_2 = set()
c_3 = set()
for i in range(len(d)):
    c_2.add(d[i][0])
for i in range(len(e)):
    c_3.add(e[i][0])


largest_cluster_ratings_2 = train_60.filter(train_60.userId.isin(c_2))
largest_cluster_ratings_3 = train_80.filter(train_80.userId.isin(c_3))

movies_largest_cluster_1 = largest_cluster_ratings_1.groupBy("movieId").agg(avg("rating").alias("avg_rating"))
movies_largest_cluster_2 = largest_cluster_ratings_2.groupBy("movieId").agg(avg("rating").alias("avg_rating"))
movies_largest_cluster_3 = largest_cluster_ratings_3.groupBy("movieId").agg(avg("rating").alias("avg_rating"))

movies_largest_cluster_1.show(5,False)

# Filter movies with average rating >= 4 for each split
top_movies_1 = movies_largest_cluster_1.filter(movies_largest_cluster_1.avg_rating >= 4).select("movieId").distinct()
top_movies_2 = movies_largest_cluster_2.filter(movies_largest_cluster_2.avg_rating >= 4).select("movieId").distinct()
top_movies_3 = movies_largest_cluster_3.filter(movies_largest_cluster_3.avg_rating >= 4).select("movieId").distinct()


movies = spark.read.csv('Data/movies.csv',header=True,inferSchema=True)


top_genres_1 = (
    top_movies_1
    .join(movies, on="movieId")
)

# Count the occurrence of each genre and show the top 10
top_genres_1.groupBy("genres").count().orderBy("count", ascending=False).limit(10).show()
top_genres_2 = (
    top_movies_2
    .join(movies, on="movieId")
)

# Count the occurrence of each genre and show the top 10
top_genres_2.groupBy("genres").count().orderBy("count", ascending=False).limit(10).show()
top_genres_3 = (
    top_movies_3
    .join(movies, on="movieId")
)

# Count the occurrence of each genre and show the top 10
top_genres_3.groupBy("genres").count().orderBy("count", ascending=False).limit(10).show()


top_1 = top_genres_1.groupBy("genres").count().orderBy("count", ascending=False).limit(10).toPandas()
top_2 =top_genres_2.groupBy("genres").count().orderBy("count", ascending=False).limit(10).toPandas()
top_3 =top_genres_3.groupBy("genres").count().orderBy("count", ascending=False).limit(10).toPandas()


df_f = pd.DataFrame()
df_f["Split 1 Genres"] = top_1["genres"]
df_f["Split 1 Count"] = top_1["count"]
df_f["Split 2 Genres"] = top_2["genres"]
df_f["Split 2 Count"] = top_2["count"]
df_f["Split 3 Genres"] = top_3["genres"]
df_f["Split 3 Count"] = top_3["count"]


display(df_f)
