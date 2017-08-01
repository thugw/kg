package com.bs.bicycle

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

/**
  * Created by hadoop on 2017/8/1.
  */
object BikeUDF {

  val week2Int: UserDefinedFunction = udf[Int, String] {
    case "星期一" => 1
    case "星期二" => 2
    case "星期三" => 3
    case "星期四" => 4
    case "星期五" => 5
    case "星期六" => 6
    case "星期七" => 7
    case _ => 0
  }

  val sumCount: UserDefinedFunction = udf[Int, Int, Int]((c1, c2) =>{
    c1 + c2
  })

  val sumWeightCount: UserDefinedFunction = udf[Int, Int, Int]((c1, c2) =>{
    (0.2 * c1 + 0.8 * c2).round.toInt
  })

  val workingday: UserDefinedFunction = udf[Int, String, Int]((date, workingday) =>{
    if(date.equals("2011-04-15") || date.equals("2012-04-16")) 1
    else if (date.equals("2011-11-25") || date.equals("2012-11-23")) 0
    //    else if (date.equals("2012-05-21") || date.equals("2012-06-01") || date.equals("2012-10-30")) 0
    else if (date.contains("12-24") /*|| date.contains("12-26") */|| date.contains("12-31")) 0
    else workingday
  })

  val holiday: UserDefinedFunction = udf[Int, String, Int]((date, holiday) =>{
    if(date.equals("2011-04-15") || date.equals("2012-04-16")) 0
    else if (date.equals("2011-11-25") || date.equals("2012-11-23")) 1
    else if (date.equals("2012-05-21") || date.equals("2012-06-01") || date.equals("2012-10-30")) 1 // storms tornado sandy
    else if (date.contains("12-24") || date.contains("12-26") || date.contains("12-31")) 1
    else holiday
  })

  val peak: UserDefinedFunction = udf[Int, Int, Int]((hour, workingday) =>{
    if(workingday == 1 && (hour == 8 || hour == 12 || hour == 17 || hour == 18)) 1
    else if(workingday == 0 && (hour >= 10 && hour <= 19)) 1
    else 0
  })

  val ideal: UserDefinedFunction = udf[Int, Double, Double]((temp, windspeed) =>{
    if (temp > 27 && windspeed < 30) 1 else 0
  })

  val sticky: UserDefinedFunction = udf[Int, Int, Int]((humidity, workingday) =>{
    if (workingday == 1 && humidity >= 60) 1 else 0
  })

}
