package com.bs.titanic

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{GBTClassifier, MultilayerPerceptronClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col, when}
import org.apache.spark.sql.types.IntegerType

/**
  * Created by hadoop on 2017/8/1.
  */
object TitanicModel {

  def main(args: Array[String]): Unit = {

    val path = "data/titanic/train.csv"
    val path2 = "data/titanic/test.csv"
    val savePath = "data/titanic/result"

    val spark = SparkSession.builder().appName(getClass.getSimpleName).master("local[2]").getOrCreate()
    import spark.implicits._

    val df = spark.read.option("header", true).option("inferSchema", true).csv(path)
    val df_test = spark.read.option("header", true).option("inferSchema", true).csv(path2)
    val meanFare = df_test.agg(avg("Fare")).first().getDouble(0)

    //    df_test.describe().show(false)
    //    df.printSchema()
    //    df.describe().show(false)

    val known_age = df.filter($"Age".isNotNull)
    val unknown_age = df.filter($"Age".isNull)
    val known_age_test = df_test.filter($"Age".isNotNull)
    val unknown_age_test = df_test.filter($"Age".isNull)

    val va = new VectorAssembler().setInputCols(Array("Pclass", "SibSp", "Parch", "Fare")).setOutputCol("age_feature")
    val rfr = new RandomForestRegressor().setLabelCol("Age").setFeaturesCol("age_feature")
    val pipeline = new Pipeline().setStages(Array(va, rfr))

    val model = pipeline.fit(known_age)
    val pred = model.transform(unknown_age).withColumn("Age", $"prediction").select(known_age.columns.map(col): _*)
    val pred2 = model.transform(unknown_age_test).withColumn("Age", $"prediction").select(known_age_test.columns.map(col): _*)

    val df2 = known_age.union(pred)
    val df_test2 = known_age_test.union(pred2)
    val df3 = df2.withColumn("Cabin", when($"Cabin".isNull, "NO").otherwise("YES")).na.fill(Map("Embarked" -> "S")).withColumn("Title", TitanicUDF.title($"Name", $"Sex"))
      .withColumn("IsChild", TitanicUDF.isChild($"Age"))
      .withColumn("FamilySize", TitanicUDF.familySize($"SibSp", $"Parch"))
      .withColumn("PclassSex", TitanicUDF.pclassSex($"Pclass", $"Sex"))
    val df_test3 = df_test2.withColumn("Cabin", when($"Cabin".isNull, "NO").otherwise("YES")).na.fill(Map("Embarked" -> "S", "Fare" -> meanFare)).withColumn("Title", TitanicUDF.title($"Name", $"Sex"))
      .withColumn("IsChild", TitanicUDF.isChild($"Age"))
      .withColumn("FamilySize", TitanicUDF.familySize($"SibSp", $"Parch"))
      .withColumn("PclassSex", TitanicUDF.pclassSex($"Pclass", $"Sex"))


    val catFeature = Array("Pclass", "Sex", "Cabin", /*"Embarked",*/ "Title", "PclassSex", "IsChild")
    val catFeatureIndexed = catFeature.map(x => x + "Indexed")
    val catFeatureIndexedoh = catFeatureIndexed.map(x => x + "oh")
    val numFeature = Array("Age", "SibSp", "Parch", "Fare", "FamilySize")

    val si = catFeature.map(c => new StringIndexer().setInputCol(c).setOutputCol(c + "Indexed").setHandleInvalid("skip").fit(df3))
    val oh = catFeatureIndexed.map(c => new OneHotEncoder().setInputCol(c).setOutputCol(c + "oh"))

    val allFeature = catFeatureIndexedoh ++ numFeature

    val va2 = new VectorAssembler().setInputCols(allFeature).setOutputCol("features")
    //    val vi = new VectorIndexer().setInputCol("features").setOutputCol("featuresIdx").setMaxCategories(10)

    val rf = new RandomForestClassifier().setLabelCol("Survived").setFeaturesCol("features")

    val gbt = new GBTClassifier().setLabelCol("Survived").setFeaturesCol("features")

    val mlpc = new MultilayerPerceptronClassifier().setFeaturesCol("features").setLabelCol("Survived").setLayers(Array(allFeature.length, 30, allFeature.length, 2)).setSeed(1234567L)

    val pipeline2 = new Pipeline().setStages(si ++ oh ++ Array(va2, rf))

    val params = new ParamGridBuilder().addGrid(rf.numTrees, Array(30, 40)).addGrid(rf.maxDepth, Array(6, 8)).addGrid(rf.maxBins, Array(15, 25)).build()
    val params_gbt = new ParamGridBuilder().addGrid(gbt.maxIter, Array(15, 20)).addGrid(gbt.maxDepth, Array(6, 9)).addGrid(gbt.maxBins, Array(32, 40)).build()
    val params_mlpc = new ParamGridBuilder().addGrid(mlpc.maxIter, Array(100, 150)).build()

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("Survived")
    val evaluator2 = new MulticlassClassificationEvaluator().setLabelCol("Survived")

    val cv = new CrossValidator().setEstimator(pipeline2).setEvaluator(evaluator2).setEstimatorParamMaps(params).setNumFolds(5)

    val cvModel = cv.fit(df3)

    val predictions = cvModel.transform(df_test3)

    val best = cvModel.bestModel.asInstanceOf[PipelineModel]
    val rfm = best.stages(si.length + oh.length + 1).asInstanceOf[RandomForestClassificationModel]
    val bestparam1 = rfm.extractParamMap().getOrElse(rfm.numTrees, 1)
    val bestparam2 = rfm.extractParamMap().getOrElse(rfm.maxDepth, 1)
    val bestparam3 = rfm.extractParamMap().getOrElse(rfm.maxBins, 1)
    println(s"numTrees: $bestparam1")
    println(s"maxDepth: $bestparam2")
    println(s"maxBins: $bestparam3")

    //    val gbtm = best.stages(si.length + oh.length + 1).asInstanceOf[GBTClassificationModel]
    //    val bestparam11 = gbtm.extractParamMap().getOrElse(gbtm.maxIter, 1)
    //    val bestparam12 = gbtm.extractParamMap().getOrElse(gbtm.maxDepth, 1)
    //    val bestparam13 = gbtm.extractParamMap().getOrElse(gbtm.maxBins, 1)
    //    println(s"maxIter: $bestparam11")
    //    println(s"maxDepth: $bestparam12")
    //    println(s"maxBins: $bestparam13")
    //    println(gbtm.toDebugString)


    //    predictions.printSchema()
    //    predictions.show(10, false)

    predictions.select($"PassengerId", $"prediction".as("Survived").cast(IntegerType)).repartition(1).write.options(Map("header" -> "true")).csv(savePath)

  }
}
