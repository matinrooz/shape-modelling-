package example

import scalismo.common.{EuclideanSpace3D, Field, RealSpace}
import scalismo.common.interpolation.{NearestNeighborInterpolator, TriangleMeshInterpolator3D}
import scalismo.geometry.{EuclideanVector, EuclideanVector3D, Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, DiagonalKernel3D, GaussianKernel, GaussianKernel3D, PDKernel}
import scalismo.mesh.TriangleMesh3D
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{GaussianProcess3D, LowRankGaussianProcess, PointDistributionModel, StatisticalMeshModel}
import scalismo.transformations._
import scalismo.ui.api.ScalismoUI
import breeze.linalg.DenseVector

import java.awt.Color
import java.io.File
import scala.swing.Color

object GPmodel {

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)
    val ui = ScalismoUI()

    //loading reference mesh
    val reference = MeshIO.readMesh(new java.io.File("data/datasets/femur.stl")).get

    //Loading reference Landmarks
    val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new File("data/datasets/femur.json")).get

    val modelGroup = ui.createGroup("model")
    val refMeshView = ui.show(modelGroup, reference, "referenceMesh")
    val rigidFemurs = ui.createGroup("RigidFemurs")
    val alignedFemurs = ui.createGroup("RigidFemurs")

    //loading a rigid mesh
    val meshRigid: TriangleMesh3D = MeshIO.readMesh(new File("data/datasets/full-femurs/meshes/1.stl")).get
    //loading a rigid landmark
    val rigidLandmarks = LandmarkIO.readLandmarksJson[ _3D ](new File("data/datasets/full-femurs/landmarks/1.json")).get

    //performing rigid alignment
    val bestTransform: RigidTransformation[ _3D ] = LandmarkRegistration.rigid3DLandmarkRegistration(rigidLandmarks, referenceLandmarks, center = Point(0, 0, 0))
    val alignedFemur = meshRigid.transform(bestTransform)
    val alignedLandmarks = rigidLandmarks.map(lm => lm.copy(point = bestTransform(lm.point)))
    val rigidFemurView = ui.show(rigidFemurs, meshRigid, "Rigid_Femur")
    val alignedFemurView = ui.show(alignedFemurs, alignedFemur, "Aligned_Femur")
    rigidFemurView.color = java.awt.Color.red
    alignedFemurView.color = java.awt.Color.green

    val refMeshColor = new Color(115,238,70)
    val defMeshColor = new Color(101,70,108)
    refMeshView.color = refMeshColor


    //initializing mean for Gaussian Process
    val zeroMean = Field(EuclideanSpace3D, (pt:Point[_3D]) => EuclideanVector3D(0,0,0))

    //setting up the kernels with combination of scalar valued Kernels and matrix valued kernels
    val scalarValuedGaussianKernel1 : PDKernel[_3D]= GaussianKernel3D(sigma = 40.0)//scalar kernel
    val scalarValuedGaussianKernel2 : PDKernel[_3D]= GaussianKernel3D(sigma = 60.0)
    val scalarValuedGaussianKernel3 : PDKernel[_3D]= GaussianKernel3D(sigma = 100.0)
    val kernel1 = DiagonalKernel3D(scalarValuedGaussianKernel1,scalarValuedGaussianKernel2,scalarValuedGaussianKernel3) //matrix kernel

    val scalarValuedGaussianKernel4 : PDKernel[_3D]= GaussianKernel3D(sigma = 50.0) //scalar kernel
   val scalarValuedGaussianKernel5 : PDKernel[_3D]= GaussianKernel3D(sigma = 60.0)
    val scalarValuedGaussianKernel6 : PDKernel[_3D]= GaussianKernel3D(sigma = 100.0)
    val kernel2 = DiagonalKernel3D(scalarValuedGaussianKernel4,scalarValuedGaussianKernel5,scalarValuedGaussianKernel6)//matrix kernel
    val kernel = kernel1+kernel2  //summing up the kernels
    val gp = GaussianProcess3D [EuclideanVector[_3D]](zeroMean,kernel) //gaussian process
    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(alignedFemur,  gp,0.07,  NearestNeighborInterpolator()) //retreiving low rank for generating the model
    val deformedGroup = ui.createGroup("deformed")
    val refMeshView2 = ui.show(deformedGroup, reference, "referenceMesh")
    refMeshView2.color = defMeshColor
    val gpDefView = ui.addTransformation(deformedGroup, lowRankGP, "RefMeshDeformedByGp")

    val model = PointDistributionModel(reference,lowRankGP)  //creating the model
     ui.show(modelGroup , model , "gp model")

   val gpModel = StatisticalModelIO.writeStatisticalTriangleMeshModel3D(model, new java.io.File("data/datasets/gpmodel.h5")) //saving the model

      }



}