import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.{EuclideanSpace, Field, PointId}
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, PDKernel}
import scalismo.mesh.TriangleMesh
import scalismo.numerics.{RandomMeshSampler3D, UniformMeshSampler3D}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

import java.io.File

object Reconstruction {
  scalismo.initialize()

  val ui = ScalismoUI()

  implicit val rng = scalismo.utils.Random(42)

  def main(args: Array[ String ]): Unit = {


    // Loading GP model
    var PCAmodel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("D:\\alignedfemurs\\gpmodel.h5")).get
    ui.show(ui.createGroup("Loaded PCA"), PCAmodel, "LoadedPCA")


    // Loading all Partial Femurs
        (47 until 57).foreach { i =>
      val partialMesh = MeshIO.readMesh(new java.io.File("D\\partial-femurs\\" + i  + ".stl")).get
      val targetGroup = ui.createGroup("partial_meshes")
      ui.show(targetGroup, partialMesh, "partial_femur")

      // Setting up correspondences
      val sampler = UniformMeshSampler3D(partialMesh, 1000)
      val points: Seq[ Point[ _3D ] ] = sampler.sample.map(pointWithProbability => pointWithProbability._1)
      val ptIds = points.map(point => PCAmodel.referenceMesh.pointSet.findClosestPoint(point).id)

      def attributeCorrespondences(movingMesh: TriangleMesh[ _3D ], referenceMesh: TriangleMesh[ _3D ], ptIds: Seq[ PointId ]): Seq[ (PointId, Point[ _3D ]) ] = {
        ptIds.map { id: PointId =>
          val pt = movingMesh.pointSet.point(id)
          val closestPointOnMesh2 = referenceMesh.pointSet.findClosestPoint(pt).point
          (id, closestPointOnMesh2)
        }
      }

      // Finding correspondences
      val correspondences = attributeCorrespondences(PCAmodel.mean, partialMesh, ptIds)

      val referencePoints = correspondences.map(pointPair => pointPair._1)
      val targetPoints = correspondences.map(pointPair => pointPair._2)


      val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[ Double ](3), DenseMatrix.eye[ Double ](3) * 0.5)

      def fitModel(correspondences: Seq[ (PointId, Point[ _3D ]) ]): TriangleMesh[ _3D ] = {

        val regressionData = correspondences.map(correspondence =>
          (correspondence._1, correspondence._2, littleNoise)
        )
        val posterior = PCAmodel.posterior(regressionData.toIndexedSeq)
        posterior.mean
      }


    val resultGroup = ui.createGroup("ReconstructionSet")

    def nonrigidICP(movingMesh: TriangleMesh[ _3D ], targetMesh: TriangleMesh[ _3D ], targetPoints: Seq[ Point[ _3D ] ], Iterations: Int): TriangleMesh[ _3D ] = {
      if (Iterations == 0) movingMesh
      else {
        val ptIds = targetPoints.map(point => PCAmodel.referenceMesh.pointSet.findClosestPoint(point).id)
        val correspondences = attributeCorrespondences(movingMesh, targetMesh, ptIds)
        val transformedModel = fitModel(correspondences)
        nonrigidICP(transformedModel, targetMesh, targetPoints, Iterations - 1)
      }
    }


    val finalFit = nonrigidICP(PCAmodel.mean, partialMesh, points, 100)
    ui.show(resultGroup, finalFit, "Reconstructed Femur")

    MeshIO.writeSTL(finalFit, new File("D:\\alignedfemurs\\results\\" + i + ".stl"))
          println("Femure"+(i+46)+"is done")
  }
}

}


