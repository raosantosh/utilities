package com.yahoo.spike

import java.util.ArrayList

object TestingMain {
  def main(args: Array[String]): Unit =
    {
      println("hi")

      doSomething("Hi Santosha")
      println(Math.abs(2.6))
      var hello = new ArrayList[String];
      hello.add("this")
      var hello1 = List(1, 2, 3)
      var hello2 = hello1.filter(dd => dd != 1)
      println(hello2)
    }

  def doSomething(value: String) = {
    println("hello")
  }
} 

