package com.yahoo.spike

import akka.actor.Actor
import akka.event.Logging
import akka.actor.ActorSystem
import akka.actor.Props
 
class TomCruise extends Actor {
  def receive = {
    case "actor" => println("received test in cruise")
    case _      => println("received unknown message")
  }
}

class VanDam extends Actor {
  
  def receive = {
    case "actor1" => println("received test in dam")
    case _      => println("received unknown message")
    
    val system = ActorSystem("HelloSystem")
    val anotherActor = system.actorOf(Props[TomCruise], name = "helloactor")
    
    anotherActor ! "actor"
  }
}

object Main extends App {
  val system = ActorSystem("HelloSystem")
  // default Actor constructor
  val cruise = system.actorOf(Props[TomCruise], name = "cruise")
  val dam = system.actorOf(Props[VanDam], name = "dam")
  val hanks = system.actorOf(Props[TomHanks], name = "hanks")
  
  cruise ! "actor"
  dam ! "actor1"
  hanks ! "actor"
}