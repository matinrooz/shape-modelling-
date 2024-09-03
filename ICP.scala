package example

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry.{Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.UniformMeshSampler3D
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.{ScalismoUI, ScalismoUIHeadless}

import java.io.File

object ICP {

  scalismo.initialize()
  implicit val rng = scalismo.utils.Random(42)

  val ui = ScalismoUI()
  def main(args: Array[ String ]): Unit = {

    // load the GP model that was created previously
    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("D:\\UNIBAS\\Scalismo\\scalismoseed\\data\\datasets\\gpmodel.h5")).get


    // load reference mesh
    val referenceMesh = MeshIO.readMesh(new File("D:\\UNIBAS\\Scalismo\\scalismoseed\\data\\datasets\\femur.stl")).get
    val meshGroup = ui.createGroup("Meshes")
    ui.show(meshGroup, referenceMesh, "reference_mesh")


    // load reference landmarks
    val referenceLandmarks = LandmarkIO.readLandmarksJson[ _3D ](new java.io.File("D:\\UNIBAS\\Scalismo\\scalismoseed\\data\\datasets\\femur.json")).get
    val points: Seq[ Point[ _3D ] ] = referenceLandmarks.map(lm => lm.point)
    val ptIds = points.map(point => referenceMesh.pointSet.findClosestPoint(point).id)

    for (i <- 0 to 46) {
      //loading the aligned meshes and landmarks
      val targetMesh = MeshIO.readMesh(new File("D:\\UNIBAS\\Scalismo\\scalismoseed\\data\\datasets\\alignedfemurs\\meshes\\" + i + ".stl")).get
      ui.show(meshGroup, targetMesh, "target_mesh")

      val targetLandmarks = LandmarkIO.readLandmarksJson[ _3D ](new java.io.File("D:\\UNIBAS\\Scalismo\\scalismoseed\\data\\datasets\\alignedfemurs\\landmarks/" + i + ".json")).get
      val pointsTarget: Seq[ Point[ _3D ] ] = targetLandmarks.map(lm => lm.point)
      val posteriorModel = model.posterior(ptIds.zip(pointsTarget).toIndexedSeq, 0.1) //posterior has been created here
      ui.show(ui.createGroup("posterior"), posteriorModel, "posterior")


      def attributeCorrespondences(movingMesh: TriangleMesh[ _3D ], referenceMesh: TriangleMesh[ _3D ], ptIds: Seq[ PointId ]): Seq[ (PointId, Point[ _3D ]) ] = {
        ptIds.map { id: PointId =>
          val pt = movingMesh.pointSet.point(id)
          val closestPointOnMesh2 = referenceMesh.pointSet.findClosestPoint(pt).point
          (id, closestPointOnMesh2)
        }
      }

      //finds for each point of interest the closest point on the target. From tutorial 10
      def fitModel(correspondences: Seq[ (PointId, Point[ _3D ]) ]): TriangleMesh[ _3D ] = {

        val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[ Double ](3), DenseMatrix.eye[ Double ](3))
        val regressionData = correspondences.map(correspondence =>
          (correspondence._1, correspondence._2, littleNoise)
        )
        val posterior = posteriorModel.posterior(regressionData.toIndexedSeq)
        posterior.mean
      }

      //we find the corresponding points in the other mesh
      def nonrigidICP(movingMesh: TriangleMesh[ _3D ], targetMesh: TriangleMesh[ _3D ], posteriorModel: StatisticalMeshModel, targetPoints: Seq[ Point[ _3D ] ], numberOfIterations: Int): TriangleMesh[ _3D ] = {
        if (numberOfIterations == 0) movingMesh
        else {
          val ptIds = targetPoints.map(point => posteriorModel.referenceMesh.pointSet.findClosestPoint(point).id)
          val correspondences = attributeCorrespondences(movingMesh, targetMesh, ptIds)
          val transformed = fitModel(correspondences)  //fitting posterior to model using non rigid ICP

          nonrigidICP(transformed, targetMesh, posteriorModel, targetPoints, numberOfIterations - 1)
        }
      }

      val sampler = UniformMeshSampler3D(model.referenceMesh, 1000)
      val points: Seq[ Point[ _3D ] ] = sampler.sample.map(pointWithProbability => pointWithProbability._1)


      val resultGroup = ui.createGroup("Fitted Mesh")
      val finalFit = nonrigidICP(posteriorModel.mean, targetMesh, posteriorModel, points, 150)  // We perform iteration 150 times
      ui.show(resultGroup, finalFit, "Final Fit")


      MeshIO.writeSTL(finalFit, new File("D:\\UNIBAS\\Scalismo\\scalismoseed\\data\\datasets\\PosteriorFitted\\Fitted-" + i + ".stl")) //saving fitted mesh


    }

  }
}
