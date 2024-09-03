import scalismo.common.PointId
import scalismo.geometry.{Landmark, Point, _3D}
import scalismo.ui.api._
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel._
import scalismo.statisticalmodel.dataset._

object PCAmodel {

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  // create a visualization window
  val ui = ScalismoUI()

  def main(args: Array[String]): Unit = {

    // We are building a PCA model using all fitted meshes in registration step
    // refer to last part("An easier way to build a model") of the tutorial for the method https://unibas-gravis.github.io/scalismo-tutorial/tutorials/tutorial6.html


    val meshFiles = new java.io.File("D:\\UNIBAS\\Scalismo\\scalismoseed\\data\\datasets\\PosteriorFitted\\").listFiles // load all fitted 46 meshes into a list
    val meshes = meshFiles.map{f => MeshIO.readMesh(f).get}

    val toAlign : IndexedSeq[TriangleMesh[_3D]] = meshes.tail
    val reference = meshes.head  //setting the reference mesh

    val pointIds = IndexedSeq(2214, 6341, 10008, 14129, 8156, 17000)
    val refLandmarks = pointIds.map{id => Landmark(s"L_$id", reference.pointSet.point(PointId(id))) }
    val alignedMeshes = toAlign.map { mesh =>
      val landmarks = pointIds.map{id => Landmark("L_"+id, mesh.pointSet.point(PointId(id)))}
      val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(landmarks, refLandmarks, center = Point(0,0,0))
      mesh.transform(rigidTrans)
      }
    val dc = DataCollection.fromTriangleMesh3DSequence(reference, alignedMeshes)// PCA from tutorial 6 of scalismo.org
    val pcaModel = PointDistributionModel.createUsingPCA(dc)
    val pcaGroup = ui.createGroup("PCA_Group")
    ui.show(pcaGroup, pcaModel, "PCA")

    StatisticalModelIO.writeStatisticalTriangleMeshModel3D(pcaModel, new java.io.File("D:\\alignedfemurs\\NewPCA-Model.h5"))// Saving PCA model
    println("Model has been saved")
  }

}

