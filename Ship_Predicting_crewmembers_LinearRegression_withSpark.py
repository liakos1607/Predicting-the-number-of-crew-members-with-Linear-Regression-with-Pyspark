
import pyspark

from pyspark.ml.regression import LinearRegression

from pyspark.sql import SparkSession

spark=SparkSession.builder.appName('Linear').getOrCreate()

df=spark.read.csv('cruise_ship_info.csv',inferSchema=True,header=True)

from pyspark.ml.regression import LinearRegression

df.printSchema()

df.groupBy('Cruise_line').count().show()

from pyspark.ml.feature import StringIndexer

indexer=StringIndexer(inputCol='Cruise_line',outputCol='cruise_cat')
indexed=indexer.fit(df).transform(df)


from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler


indexed.columns


assembler=VectorAssembler(inputCols=['Age',
 'Tonnage',
 'passengers',
 'length',
 'cabins',
 'passenger_density',
 'cruise_cat'],outputCol='features')



output=assembler.transform(indexed)

final_data=output.select('features','crew')


train_data,test_data=final_data.randomSplit([0.7,0.3])


train_data.describe().show()

test_data.describe().show()

ship_lr=LinearRegression(labelCol='crew')

trained_ship_model=ship_lr.fit(train_data)

ship_results=trained_ship_model.evaluate(test_data)


ship_results.rootMeanSquaredError


ship_results.r2


from pyspark.sql.functions import corr

df.select(corr('crew','passengers')).show()


df.select(corr('crew','cabins')).show()




