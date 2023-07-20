
seed = 6 # seed =06

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.feature import PCA
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


spark = SparkSession.builder \
        .master("local[2]") \
        .appName("COM6012 Assignment") \
        .config("spark.local.dir","/fastdata/acq22jj/") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("OFF")


raw_df = spark.read.csv('Data/NIPS_1987-2015.csv',header=True,inferSchema=True)

paper_columns = raw_df.columns[1:]

# Transpose the DataFrame by converting to an RDD, transposing, and converting back to a DataFrame
rdd = raw_df.rdd.map(list).flatMap(lambda x: [(paper_columns[i], x[i+1]) for i in range(len(paper_columns))])
df_transposed = rdd.toDF(["papers", "words"])

# Aggregates the data by summing up the word counts for each paper, resulting in a new DataFrame with 5811 rows and 158 columns (one for each paper).
#This is a much more manageable size for running PCA and visualizing the results.
df_pivot = df_transposed.groupBy("papers").pivot("words").sum("words")

# Fill null values with 0
df_pivot = df_pivot.na.fill(0)



assembler = VectorAssembler(inputCols=df_pivot.columns[1:], outputCol="features")
assembled = assembler.transform(df_pivot)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Fit and transform the assembled data
scaledData = scaler.fit(assembled).transform(assembled).select("papers", "scaledFeatures")


pca = PCA(k=2, inputCol="scaledFeatures", outputCol="pcaFeatures")
pcaModel = pca.fit(scaledData)
pcaResult = pcaModel.transform(scaledData)


PC1 = pcaResult.select("pcaFeatures").rdd.map(lambda x: x[0][0]).collect()
PC2 = pcaResult.select("pcaFeatures").rdd.map(lambda x: x[0][1]).collect()
papers = pcaResult.select("papers").rdd.map(lambda x: x[0]).collect()
explainedvariance = pcaModel.explainedVariance.toArray()
variance_retained = explainedvariance / sum(explainedvariance)


pca_array = pcaModel.pc.toArray()
pca_array = np.matrix(pca_array)
covariance_matrix = np.cov(pca_array,rowvar=False)
eigenvalues = np.linalg.eigvals(covariance_matrix)


print(f"First 10 entries of the 2 PCs:\n PC1:{*PC1[:11],}\n PC2: {*PC2[:11],}")
print("Eigenvalues:",eigenvalues)
print('Explained_variance: ',explainedvariance)
print("Retained_variance",variance_retained)



plt.scatter(PC1, PC2, c='red',s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig('Output/Q4_papers_vs_pc1_2.png',dpi=400)
plt.clf()


# PLot PC1 vs papers
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(PC1, papers, s=10)
plt.xlabel('PC1')
plt.ylabel('Papers')
ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
plt.savefig('Output/Q4_papers_vs_pc_1.png',dpi=400)
plt.clf()
# Plot PC2 vs papers

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(PC2, papers, s=10)
plt.xlabel('PC2')
plt.ylabel('Papers')
ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
plt.savefig('Output/Q4_papers_vs_pc_2.png',dpi=400)
plt.clf()



