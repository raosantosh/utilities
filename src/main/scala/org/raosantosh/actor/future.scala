import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.concurrent.Await
import scala.util.{Success, Failure}
import scala.concurrent._
import scala.concurrent.duration._

class temp {

  val f: Future[String] = Future {
    "Santosh"
  }
  
  Await.result(f, 0 nanos)
  
 async {
    
  }
  

  
  f onComplete {
    case Success(values) => println (values)
    case Failure(t) => println("Failed")
  }
}