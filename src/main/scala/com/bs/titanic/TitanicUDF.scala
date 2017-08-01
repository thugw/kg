package com.bs.titanic

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

import scala.util.matching.Regex

/**
  * Created by hadoop on 2017/8/1.
  */
object TitanicUDF {

  val title: UserDefinedFunction = udf[String, String, String]((name, sex) => {
    val titles = Map(
      "Mrs" -> "Mrs",
      "Lady" -> "Mrs",
      "Mme" -> "Mrs",
      "Ms" -> "Ms",
      "Miss" -> "Miss",
      "Mlle" -> "Miss",
      "Master" -> "Master",
      "Rev" -> "Rev",
      "Don" -> "Mr",
      "Sir" -> "Sir",
      "Dr" -> "Dr",
      "Col" -> "Col",
      "Capt" -> "Col",
      "Major" -> "Col"
    )

    val p = new Regex(""".*,(.*?)\.""")
    p.findFirstMatchIn(name) match {
      case Some(m) => titles.get(m.group(1).trim) match {
        case Some(t) => t
        case _ => if (sex == "male") "Mr" else "Mrs"
      }
      case _ => if (sex == "male") "Mr" else "Mrs"
    }
  })


  val isChild: UserDefinedFunction = udf[String, Int](age => {
    if (age <= 12) "child"
    else if (age <= 30) "adult"
    else if (age <= 60) "senior"
    else "aged"
  })

  val familySize: UserDefinedFunction = udf[Int, Int, Int]((sibSp, parch) => {
    sibSp + parch
  })

  val pclassSex: UserDefinedFunction = udf[String, Int, String]((pclass, sex) => {
    pclass + sex
  })

}
