package com.bs.bicycle

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, RandomForestRegressor}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

/**
  * Created by hadoop on 2017/8/1.
  */
object BikeModel {

  def datetimeEtl(df: DataFrame, spark: SparkSession): DataFrame = {
    import spark.implicits._
    val df2 = df.withColumn("month", month($"datetime")).withColumn("hour", hour($"datetime")).withColumn("dow", BikeUDF.week2Int(date_format($"datetime", "EEEE")))
      .withColumn("day", dayofmonth($"datetime")).withColumn("woy", weekofyear($"datetime")).withColumn("date", to_date($"datetime").cast("string")).withColumn("year", year($"datetime"))
    df2
  }




  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().appName(getClass.getSimpleName).master("local[2]")/*.config("spark.sql.shuffle.partitions", "1")*/.getOrCreate()

    import spark.implicits._

    val dataPath = "data/bike/bike_train.csv"
    val testPath = "data/bike/bike_test.csv"
    val savePath = "data/bike/result"

    //  datetime|season|holiday|workingday|weather|temp |atemp |humidity|windspeed|casual|registered|count|
    val df_train = spark.read.option("header", true).option("inferSchema", true).csv(dataPath)
    val df_test = spark.read.option("header", true).option("inferSchema", true).csv(testPath)

    val df_train2 = datetimeEtl(df_train, spark)
    val df_test2 = datetimeEtl(df_test, spark)

    val df_train3 =df_train2.withColumn("casual_log", log1p($"casual")).withColumn("registered_log", log1p($"registered")).withColumn("count_log", log1p($"count"))
    val df_season = df_train3.groupBy("season").agg("count" -> "sum").withColumnRenamed("sum(count)", "count_season")
    val df_train4 = df_train3.as("t1").join(df_season.as("t2"), $"t1.season" === $"t2.season").select("t1.*", "t2.count_season")
    val df_test4 = df_test2.as("t1").join(df_season.as("t2"), $"t1.season" === $"t2.season").select("t1.*", "t2.count_season")

    //    df_train4.filter($"date" === "2011-04-15" || $"date" === "2012-04-16" || $"date" === "2011-11-25" || $"date" === "2012-11-23").select("date", "workingday", "holiday").dropDuplicates("date").show(false)

    val df_train5 = df_train4.withColumn("workingday", BikeUDF.workingday($"date", $"workingday")).withColumn("holiday", BikeUDF.holiday($"date", $"holiday"))
    val df_test5 = df_test4.withColumn("workingday", BikeUDF.workingday($"date", $"workingday")).withColumn("holiday", BikeUDF.holiday($"date", $"holiday"))

    //    df_train5.filter($"date" === "2011-04-15" || $"date" === "2012-04-16" || $"date" === "2011-11-25" || $"date" === "2012-11-23").select("date", "workingday", "holiday").dropDuplicates("date").show(false)

    val df_train6 = df_train5.withColumn("peak", BikeUDF.peak($"hour", $"workingday")).withColumn("ideal", BikeUDF.ideal($"temp", $"windspeed")).withColumn("sticky", BikeUDF.sticky($"humidity", $"workingday"))
    val df_test6 = df_test5.withColumn("peak", BikeUDF.peak($"hour", $"workingday")).withColumn("ideal", BikeUDF.ideal($"temp", $"windspeed")).withColumn("sticky", BikeUDF.sticky($"humidity", $"workingday"))

    //    df_train6.show(10, false)
    //    df_test6.show(10, false)

    val rfCol = Array("weather", "temp", "atemp", "windspeed","workingday", "season", "holiday", "sticky", "hour", "dow", "woy", "peak")
    val gbtCol = Array("weather", "temp", "atemp", "humidity", "windspeed", "holiday", "workingday", "season", "hour", "dow", "year", "ideal","count_season")

    val va_rf = new VectorAssembler().setInputCols(rfCol).setOutputCol("features")
    val va_gbt = new VectorAssembler().setInputCols(gbtCol).setOutputCol("features")

    val rfr1 = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("casual_log").setMinInstancesPerNode(5).setMaxDepth(15)
    val rfr2 = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("registered_log").setMinInstancesPerNode(5).setMaxDepth(15)
    val pipeline1 = new Pipeline().setStages(Array(va_rf, rfr1))
    val pipeline2 = new Pipeline().setStages(Array(va_rf, rfr2))

    //    val param1 = new ParamGridBuilder().addGrid(rfr1.maxDepth, Array(12, 15)).build()
    //    val param2 = new ParamGridBuilder().addGrid(rfr2.maxDepth, Array(12, 15)).build()
    //
    //    val eval1 = new RegressionEvaluator().setLabelCol("casual_log").setPredictionCol("prediction").setMetricName("rmse")
    //    val eval2 = new RegressionEvaluator().setLabelCol("registered_log").setPredictionCol("prediction").setMetricName("rmse")
    //
    //    val cv1 = new CrossValidator().setEstimator(pipeline1).setEvaluator(eval1).setEstimatorParamMaps(param1).setNumFolds(3)
    //    val cv2 = new CrossValidator().setEstimator(pipeline2).setEvaluator(eval2).setEstimatorParamMaps(param2).setNumFolds(3)
    //
    //    val cvModel1 = cv1.fit(df_train6)
    //    val cvModel2 = cv2.fit(df_train6)
    //
    //    val cv_pred1 = cvModel1.transform(df_test6).withColumn("res1", expm1("prediction")).select("datetime", "res1")
    //    val cv_pred2 = cvModel2.transform(df_test6).withColumn("res2", expm1("prediction")).select("datetime", "res2")
    //    val res_rf = cv_pred1.as("t1").join(cv_pred2.as("t2"), $"t1.datetime" === $"t2.datetime").select($"t1.*", $"t2.res2").withColumn("count", BikeUDF.sumCount($"res1", $"res2")).select(date_format($"datetime", "yyyy-MM-dd HH:mm:ss").as("datetime"), $"count")

    val model1 = pipeline1.fit(df_train6)
    val model2 = pipeline2.fit(df_train6)
    val pred1 = model1.transform(df_test6)
    val pred2 = model2.transform(df_test6)
    val res1 = pred1.withColumn("res1", expm1("prediction")).select("datetime", "res1")
    val res2 = pred2.withColumn("res2", expm1("prediction")).select("datetime", "res2")
    val res_rf = res1.as("t1").join(res2.as("t2"), $"t1.datetime" === $"t2.datetime").select($"t1.*", $"t2.res2").withColumn("count", BikeUDF.sumCount($"res1", $"res2")).select(date_format($"datetime", "yyyy-MM-dd HH:mm:ss").as("datetime"), $"count")

    val gbt1 = new GBTRegressor().setFeaturesCol("features").setLabelCol("casual_log").setMaxIter(100).setMaxDepth(5)
    val gbt2 = new GBTRegressor().setFeaturesCol("features").setLabelCol("registered_log").setMaxIter(100).setMaxDepth(5)
    val pipeline3 = new Pipeline().setStages(Array(va_gbt, gbt1))
    val pipeline4 = new Pipeline().setStages(Array(va_gbt, gbt2))

    //    val param3 = new ParamGridBuilder().addGrid(gbt1.maxDepth, Array(7, 10)).build()
    //    val param4 = new ParamGridBuilder().addGrid(gbt2.maxDepth, Array(7, 10)).build()
    //
    //    val eval3 = new RegressionEvaluator().setLabelCol("casual_log").setPredictionCol("prediction").setMetricName("rmse")
    //    val eval4 = new RegressionEvaluator().setLabelCol("registered_log").setPredictionCol("prediction").setMetricName("rmse")
    //
    //    val cv3 = new CrossValidator().setEstimator(pipeline3).setEvaluator(eval3).setEstimatorParamMaps(param3).setNumFolds(3)
    //    val cv4 = new CrossValidator().setEstimator(pipeline4).setEvaluator(eval4).setEstimatorParamMaps(param4).setNumFolds(3)
    //
    //    val cvModel3 = cv3.fit(df_train6)
    //    val cvModel4 = cv4.fit(df_train6)
    //
    //    val cv_pred3 = cvModel3.transform(df_test6).withColumn("res1", expm1("prediction")).select("datetime", "res1")
    //    val cv_pred4 = cvModel4.transform(df_test6).withColumn("res2", expm1("prediction")).select("datetime", "res2")
    //
    //    val res_gbt = cv_pred3.as("t1").join(cv_pred4.as("t2"), $"t1.datetime" === $"t2.datetime").select($"t1.*", $"t2.res2").withColumn("count", BikeUDF.sumCount($"res1", $"res2")).select(date_format($"datetime", "yyyy-MM-dd HH:mm:ss").as("datetime"), $"count")




    val model3 = pipeline3.fit(df_train6)
    val model4 = pipeline4.fit(df_train6)
    val pred3 = model3.transform(df_test6)
    val pred4 = model4.transform(df_test6)
    val res3 = pred3.withColumn("res1", expm1("prediction")).select("datetime", "res1")
    val res4 = pred4.withColumn("res2", expm1("prediction")).select("datetime", "res2")
    val res_gbt = res3.as("t1").join(res4.as("t2"), $"t1.datetime" === $"t2.datetime").select($"t1.*", $"t2.res2").withColumn("count", BikeUDF.sumCount($"res1", $"res2")).select(date_format($"datetime", "yyyy-MM-dd HH:mm:ss").as("datetime"), $"count")

    val res = res_rf.as("t1").join(res_gbt.as("t2"), $"t1.datetime" === $"t2.datetime").select($"t1.datetime", $"t1.count".as("c1"), $"t2.count".as("c2")).withColumn("count", BikeUDF.sumWeightCount($"c1", $"c2")).select("datetime", "count")

    res.coalesce(1).sort("datetime").write.option("header", "true").csv(savePath)


  }

}
