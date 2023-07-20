
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler,StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression #poisson
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder,OneHotEncoderModel
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import log, when, col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import json
import matplotlib.pyplot as plt
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/acq22jj") \
    .getOrCreate()


sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

raw_data = spark.read.csv('Data/freMTPL2freq.csv',header=True).cache()

motor_data = raw_data.withColumn("ClaimNb", col("ClaimNb") + 0.5) #adding 0.5 to all values to take care of log(0)
motor_data = motor_data.withColumn("BonusMalus", when(col("BonusMalus") < 100, 'Bonus').otherwise(col("BonusMalus"))) # Converting To Bonus and Malus
motor_data = motor_data.withColumn("BonusMalus", when(col("BonusMalus") >= 100, 'Malus').otherwise(col("BonusMalus")))
motor_data = motor_data.withColumn("LogClaimNb", log(col("ClaimNb")))
motor_data = motor_data.withColumn("NZClaim", when(col("ClaimNb") > 0.5, 1).otherwise(0))

motor_data = motor_data.cache()


raw_ohe = motor_data.select('IDpol',
 'ClaimNb',
 'Exposure',
 'Area',
 'VehPower',
 'VehAge',
 'DrivAge',
 'BonusMalus',
 'VehBrand',
 'VehGas',
 'Density',
 'Region',
 'LogClaimNb',
 'NZClaim')

raw_ohe = raw_ohe.cache()

#String Indexing to prepare for One Hot Encoding
# Define the categorical columns to be one-hot encoded
cat_cols = ['VehBrand','Region','Area','VehGas','BonusMalus','VehPower']

# Create a StringIndexer for each categorical column
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(raw_ohe) for col in cat_cols]

# Apply the StringIndexer to the dataframe
df_indexed = raw_ohe
for indexer in indexers:
    df_indexed = indexer.transform(df_indexed)

#df_indexed.show(truncate=False)



df_indexed = df_indexed.drop('VehBrand','Region','Area','VehGas','BonusMalus','VehPower','IDpol').cache()

# One Hot Encoding

encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),
            outputCol="{0}_encoded".format(indexer.getOutputCol())) 
    for indexer in indexers
]


ohe = OneHotEncoder(inputCols=['VehBrand_index',
 'Region_index',
 'Area_index',
 'VehGas_index',
 'BonusMalus_index',
 'VehPower_index'],
                    outputCols=['VehBrand_ohe',
 'Region_ohe',
 'Area_ohe',
 'VehGas_ohe',
 'BonusMalus_ohe',
 'VehPower_ohe'])
ohe_model = ohe.fit(df_indexed)




ohe_rawdata = ohe_model.transform(df_indexed).drop(*['VehBrand_index',
 'Region_index',
 'Area_index',
 'VehGas_index',
 'BonusMalus_index',
 'VehPower_index'])
#ohe_rawdata.show(truncate=False)



# Casting to DOuble Type
for column in ohe_rawdata.columns[1:5]:
    ohe_rawdata = ohe_rawdata.withColumn(column, col(column).cast("double"))

input_cols = ['Exposure',
 'VehAge',
 'DrivAge',
 'Density']


# Vectorising our one-hot-encoded and Numerical Columns

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
assembled_df = assembler.transform(ohe_rawdata)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(assembled_df)

scaled_df = scaler_model.transform(assembled_df)


final_df = scaled_df.drop('Exposure',
 'VehAge',
 'DrivAge',
 'Density','features')


final_df = final_df.cache()


seed = 6 #06 last two digits 

#Performing Stratified Split by using Pyspark sampleBy and using ClaimNb distinct values by taking 70 % of every distinct value
train = final_df.sampleBy("ClaimNb", fractions={0.5: 0.7, 1.5: 0.7,5.5: 0.7, 8.5: 0.7,2.5: 0.7, 16.5: 0.7,6.5: 0.7, 4.5: 0.7,9.5: 0.7, 3.5: 0.7,11.5:0.7}, seed=seed)

test = final_df.subtract(train)


train = train.cache()
test = test.cache()


input_cols_f = ['VehBrand_ohe',
 'Region_ohe',
 'Area_ohe',
 'VehGas_ohe',
 'BonusMalus_ohe',
 'VehPower_ohe',
 'scaled_features']


assembler_t = VectorAssembler(inputCols=input_cols_f, outputCol="features_")
train = assembler_t.transform(train)


assembler_tt = VectorAssembler(inputCols=input_cols_f, outputCol="features_")
test = assembler_tt.transform(test)



#Performing Poisson Regression--------------------------------------------------------------------------


glm_poisson = GeneralizedLinearRegression(featuresCol='features_', labelCol='ClaimNb', maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')


stages_poisson = [glm_poisson]
pipeline_poisson = Pipeline(stages=stages_poisson)
pipelineModel_poisson = pipeline_poisson.fit(train)


predictions_poisson = pipelineModel_poisson.transform(test)
evaluator_poisson = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")
rmse_poisson = evaluator_poisson.evaluate(predictions_poisson)


print("\nPoisson RMSE = %g \n" % rmse_poisson)
print('\n Model Coefficients')
print(pipelineModel_poisson.stages[-1].coefficients)

#Performing Linear Regression---------------------------------------------------------------------------


evaluator_linear = RegressionEvaluator(labelCol="LogClaimNb", predictionCol="prediction", metricName="rmse")



linearL1 = LinearRegression(featuresCol='features_', labelCol='LogClaimNb', maxIter=50, regParam=0.01,
                          elasticNetParam=1, solver='normal')

# Pipeline for the model with L1 regularisation
stageslrL1 = [linearL1]
pipelinelrL1 = Pipeline(stages=stageslrL1)
pipelineModellrL1 = pipelinelrL1.fit(train)
predictions_linear_L1 = pipelineModellrL1.transform(test)
# With Predictions
rmse_l1_qn = evaluator_linear.evaluate(predictions_linear_L1)


print(f'RMSE Linear Regression_L1 = {rmse_l1_qn:.3f}')
print('Coefficients: ',pipelineModellrL1.stages[-1].coefficients)


linearL2 = LinearRegression(featuresCol='features_', labelCol='LogClaimNb', maxIter=50, regParam=0.01,
                            elasticNetParam=0, solver='normal')



# Pipeline for the model with L2 regularisation
stageslrL2 = [linearL2]
pipelinelrL2 = Pipeline(stages=stageslrL2)
pipelineModellrL2 = pipelinelrL2.fit(train)
predictions_linear_L2 = pipelineModellrL2.transform(test)
# With Predictions
rmse_l2_qn = evaluator_linear.evaluate(predictions_linear_L2)



print(f'RMSE Linear Regression_L2 = {rmse_l2_qn:.3f}')
print('Coefficients: ',pipelineModellrL2.stages[-1].coefficients)



#---------Performing Logistic Regression------------------------------------------------------------


logisticL1 = LogisticRegression(featuresCol='features_', labelCol='NZClaim', maxIter=50, regParam=0.01, \
                          elasticNetParam=1, family="binomial")


# Pipeline for the model with L1 regularisation
stageslgL1 = [logisticL1]
pipelinelgL1 = Pipeline(stages=stageslgL1)
pipelineModellgL1 = pipelinelgL1.fit(train)

predictions_lg_L1 = pipelineModellgL1.transform(test)
# With Predictions
evaluator = MulticlassClassificationEvaluator\
      (labelCol="NZClaim", predictionCol="prediction", metricName="accuracy")
accuracy_lg_l1 = evaluator.evaluate(predictions_lg_L1)


#w_L1 now holds the weights for each feature
w_L1 = pipelineModellgL1.stages[-1].coefficients.values



print(f'Logistic Regression With L1 Regularisation Accuracy : {accuracy_lg_l1:.3f}')
print('Coefficients: ',w_L1)


logisticL2 = LogisticRegression(featuresCol='features_', labelCol='NZClaim', maxIter=50, regParam=0.01, \
                          elasticNetParam=0, family="binomial")




# Pipeline for the model with L1 regularisation
stageslgL2 = [logisticL2]
pipelinelgL2 = Pipeline(stages=stageslgL2)
pipelineModellgL2 = pipelinelgL2.fit(train)

predictions_lg_L2 = pipelineModellgL2.transform(test)
# With Predictions
evaluator = MulticlassClassificationEvaluator\
      (labelCol="NZClaim", predictionCol="prediction", metricName="accuracy")
accuracy_lg_l2 = evaluator.evaluate(predictions_lg_L2)


#w_L1 now holds the weights for each feature
w_L2 = pipelineModellgL2.stages[-1].coefficients.values




print(f'Logistic Regression With L2 Regularisation Accuracy : {accuracy_lg_l2:.3f}')
print('Coefficients: ',w_L2)


#Part-b Complete -----------------------------------------------------------------------

#Part-C -------------------------------------------------------------------------------


#Creating ParamGrids for our models-----------------

glm_poisson_params = ParamGridBuilder()\
    .addGrid(glm_poisson.maxIter, [50])\
    .addGrid(glm_poisson.regParam, [0.001, 0.01, 0.1, 1, 10])\
    .build()


linearL1_params = ParamGridBuilder()\
    .addGrid(linearL1.maxIter, [50])\
    .addGrid(linearL1.regParam, [0.001, 0.01, 0.1, 1, 10])\
    .build()


linearL2_params = ParamGridBuilder()\
    .addGrid(linearL2.maxIter, [50])\
    .addGrid(linearL2.regParam, [0.001, 0.01, 0.1, 1, 10])\
    .build()

logisticL1_params = ParamGridBuilder()\
    .addGrid(logisticL1.maxIter, [50])\
    .addGrid(logisticL1.regParam, [0.001, 0.01, 0.1, 1, 10])\
    .build()


logisticL2_params = ParamGridBuilder()\
    .addGrid(logisticL2.maxIter, [50])\
    .addGrid(logisticL2.regParam, [0.001, 0.01, 0.1, 1, 10])\
    .build()




#-------Crossvalidation for Models:

glm_poisson_cv = CrossValidator(estimator=glm_poisson,
                                estimatorParamMaps=glm_poisson_params,
                                evaluator=evaluator_poisson,
                                numFolds=10)

linearL1_cv = CrossValidator(estimator=linearL1,
                             estimatorParamMaps=linearL1_params,
                             evaluator=evaluator_linear,
                             numFolds=10)

linearL2_cv = CrossValidator(estimator=linearL2,
                             estimatorParamMaps=linearL2_params,
                             evaluator=evaluator_linear,
                             numFolds=10)

logisticL1_cv = CrossValidator(estimator=logisticL1,
                               estimatorParamMaps=logisticL1_params,
                               evaluator=evaluator,
                               numFolds=10)

logisticL2_cv = CrossValidator(estimator=logisticL2,
                               estimatorParamMaps=logisticL2_params,
                               evaluator=evaluator,
                               numFolds=10)


#Getting best models from Cross Validation:

best_poisson =glm_poisson_cv.fit(train)
best_linearL1_cv =linearL1_cv.fit(train)
best_linearL2_cv =linearL2_cv.fit(train)
best_logisticL1_cv =logisticL1_cv.fit(train)
best_logisticL2_cv =logisticL2_cv.fit(train)


#Plotting the Validation Curves : 

# Define a function to plot the validation curve
def plot_validation_curve(cv_model, param_name,Metric=''):
    # Get the values of the hyperparameter
    param_values = [float(param_map[param_name]) for param_map in cv_model.getEstimatorParamMaps()]

    # Get the mean and standard deviation of the cross-validation metrics for each hyperparameter value
    mean_vals = cv_model.avgMetrics
    std_vals = cv_model.stdMetrics

    # Plot the validation curve
    #plt.errorbar(param_values, mean_vals, yerr=std_vals, fmt='-o')
    plt.plot(param_values, mean_vals, '-o')
    #plt.fill_between(param_values, [mean - std for mean, std in zip(mean_vals, std_vals)], [mean + std for mean, std in zip(mean_vals, std_vals)], alpha=0.2)
    plt.xlabel(param_name)
    plt.xscale('log')
    plt.ylabel(f'{Metric}')
    plt.title('Validation Curve')

# Plot the validation curves for each model
plot_validation_curve(best_poisson, glm_poisson.regParam,'Score')
plt.savefig('Output/glm_poisson_validation_curve.png')
plt.clf()
plot_validation_curve(best_linearL1_cv, linearL1.regParam,'Score')
plt.savefig('Output/linearL1_validation_curve.png')
plt.clf()
plot_validation_curve(best_linearL2_cv, linearL2.regParam,'Score')
plt.savefig('Output/linearL2_validation_curve.png')
plt.clf()
plot_validation_curve(best_logisticL1_cv, logisticL1.regParam,'Score')
plt.savefig('Output/logisticL1_validation_curve.png')
plt.clf()
plot_validation_curve(best_logisticL2_cv,logisticL2.regParam,'Score')
plt.savefig('Output/logisticL2_validation_curve.png')
plt.clf()



