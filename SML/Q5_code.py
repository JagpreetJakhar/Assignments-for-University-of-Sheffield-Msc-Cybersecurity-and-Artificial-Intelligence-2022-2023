seed = 6 #06 seed
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import udf
import json

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("COM6012 Assignment") \
        .config("spark.local.dir","/fastdata/acq22jj/") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

raw_df = spark.read.csv('Data/HIGGS.csv.gz')

boson_column_names = ('label','lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb')


raw_columns = raw_df.columns

raw = raw_df.select([col(raw_columns[i]).alias(boson_column_names[i]) if i < len(boson_column_names) else col(raw_columns[i]) for i in range(len(raw_columns))])


df = raw.select([col(c).cast(DoubleType()) for c in raw.columns])

assembler = VectorAssembler(inputCols = boson_column_names[1:len(df.columns)], outputCol = 'features') 
vectorised = assembler.transform(df)
final= vectorised.select('features','label')

final.printSchema()

(train_1, val) = final.randomSplit([0.99, 0.01], seed)
(val_train,val_test) = val.randomSplit([0.7, 0.3], seed)
val_train.cache()
val_test.cache()
gbt = GBTClassifier(labelCol="label", featuresCol="features", \
                   maxDepth=2, maxBins=3, lossType='logistic', subsamplingRate= 0.5, seed=seed,maxIter=5,minInstancesPerNode=1)

stages = [gbt]
pipeline = Pipeline(stages=stages)
pipelineModelg = pipeline.fit(val_train)


predictions_gbt = pipelineModelg.transform(val_test)

# For accuracy
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_gbt = evaluator_acc.evaluate(predictions_gbt)

# For area under ROC curve
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
auc_gbt = evaluator_auc.evaluate(predictions_gbt)

print(f'Accuracy{accuracy_gbt}, AUC: {auc_gbt}')

gbt_param_grid = (ParamGridBuilder()
                  .addGrid(gbt.maxDepth, [4,8,12])
                  .addGrid(gbt.maxBins, [7, 15,20])
                  .addGrid(gbt.minInstancesPerNode, [1])
                  .addGrid(gbt.maxIter,[5,10,15])
                  .build())

gbt_crossval_auc = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=gbt_param_grid,
                              evaluator=evaluator_auc,
                              numFolds=3)
gbt_crossval_acc = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=gbt_param_grid,
                              evaluator=evaluator_acc,
                              numFolds=3)


cv_model_gbt_acc = gbt_crossval_acc.fit(val_train)
cv_model_gbt_auc = gbt_crossval_auc.fit(val_train)
prediction_gbt_acc = cv_model_gbt_acc.transform(val_test)
accuracy_gbt_cv = evaluator_acc.evaluate(prediction_gbt_acc)
prediction_gbt_auc = cv_model_gbt_auc.transform(val_test)
area_under_curve_gbt_cv = evaluator_auc.evaluate(prediction_gbt_auc)


print(f'accuracy_gbt_cv{accuracy_gbt_cv}\n area_under_curve_gbt_cv{area_under_curve_gbt_cv}')
paramDict_gbt_auc = {param[0].name: param[1] for param in cv_model_gbt_auc.bestModel.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict_gbt_auc, indent = 4))
paramDict_gbt_acc = {param[0].name: param[1] for param in cv_model_gbt_acc.bestModel.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict_gbt_acc, indent = 4))


#-------------------------------------Random Forest-------------------------------------------------------------------

rf = RandomForestClassifier(numTrees=5, maxDepth=2, labelCol="label", featuresCol="features",maxBins=3)

stages_rf = [rf]
pipeline_rf = Pipeline(stages=stages_rf)
pipelineModelrf = pipeline.fit(val_train)
predictions = pipelineModelrf.transform(val_test)

# Evaluate the model using BinaryClassificationEvaluator
evaluator_rf_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
auc_rf = evaluator_rf_auc.evaluate(predictions)

# Evaluate the model using MulticlassClassificationEvaluator for accuracy
evaluator_rf_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_rf = evaluator_rf_acc.evaluate(predictions)
print(f"RF Accuracy: {accuracy_rf:.4f}")
print(f"RF AUC: {auc_rf:.4f}")


paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [4, 8, 12]) \
    .addGrid(rf.maxBins, [10, 20,30]) \
    .addGrid(rf.numTrees, [15, 25, 30]) \
    .build()
    
crossval_rf_acc = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=evaluator_rf_acc,
                          numFolds=3)
crossval_rf_auc = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=evaluator_rf_auc,
                          numFolds=3)


cv_model_rf_acc = crossval_rf_acc.fit(val_train)
cv_model_rf_auc = crossval_rf_acc.fit(val_train)
prediction_rf_acc = cv_model_rf_acc.transform(val_test)
accuracy_rf_cv = evaluator_rf_acc.evaluate(prediction_rf_acc)
prediction_rf_auc = cv_model_rf_auc.transform(val_test)
area_under_curve_rf_cv = evaluator_rf_auc.evaluate(prediction_rf_auc)
print(f'area_under_curve_rf_cv{area_under_curve_rf_cv}\n accuracy_rf_cv{accuracy_rf_cv}')

paramDict_rf_auc = {param[0].name: param[1] for param in cv_model_rf_auc.bestModel.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict_rf_auc, indent = 4))


paramDict_rf_auc = {param[0].name: param[1] for param in cv_model_rf_acc.bestModel.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict_rf_auc, indent = 4))

print('#One Percent Run Complete --------------------------------------------------\n')
print(f"Best Parameters for GBT maxDepth : {paramDict_gbt_auc['maxDepth']} \n MaxBins={paramDict_gbt_auc['maxBins']} \n maxIter{paramDict_gbt_auc['maxIter']}\n ")
print(f"Best Parameters for RF maxDepth : {paramDict_rf_auc['maxDepth']} \n MaxBins={paramDict_rf_auc['maxBins']} \n numTrees{paramDict_rf_auc['numTrees']} \n")





(train, test) = final.randomSplit([0.7, 0.3], seed)
train.cache()
test.cache()
gbt_final = GBTClassifier(labelCol="label", featuresCol="features", \
                   maxDepth=paramDict_gbt_auc['maxDepth'], maxBins=paramDict_gbt_auc['maxBins'], lossType='logistic', subsamplingRate= 0.5, seed=seed,maxIter=paramDict_gbt_auc['maxIter'])


stages_gbt = [gbt_final]
pipeline_gbt_final = Pipeline(stages=stages_gbt)
pipelineModelg_final = pipeline_gbt_final.fit(train)
predictions_gbt_final = pipelineModelg_final.transform(test)


evaluator_acc_final = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_gbt_final = evaluator_acc_final.evaluate(predictions_gbt_final)

# For area under ROC curve
evaluator_auc_final = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
auc_gbt_final = evaluator_auc_final.evaluate(predictions_gbt_final)

print(f'Accuracy_final_GBT:{accuracy_gbt_final}\n AUC_final_GBT: {auc_gbt_final}')

rf_final = RandomForestClassifier(numTrees=paramDict_rf_auc['numTrees'], maxDepth=paramDict_rf_auc['maxDepth'], maxBins=paramDict_rf_auc['maxBins'],labelCol="label", featuresCol="features")

model_final = rf_final.fit(train)

# Make predictions on the test data
predictions_final_rf = model_final.transform(test)

# Evaluate the model using BinaryClassificationEvaluator
evaluator_rf_auc_final = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
auc_rf_final = evaluator_rf_auc_final.evaluate(predictions_final_rf)

# Evaluate the model using MulticlassClassificationEvaluator for accuracy
evaluator_rf_acc_final = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy_rf_final = evaluator_rf_acc_final.evaluate(predictions_final_rf)

# Print the accuracy and AUC
print(f"Accuracy_final_RF: {accuracy_rf_final:.4f}")
print(f"AUC_final_RF: {auc_rf_final:.4f}")





