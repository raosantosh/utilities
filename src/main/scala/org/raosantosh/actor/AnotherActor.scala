package com.yahoo.spike

import akka.actor.Actor

class TomHanks extends Actor {
  def receive = {
    case "actor" => println("received test in hanks")
    case _      => println("received unknown message")
  }
}