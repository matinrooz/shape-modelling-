package example

import jdk.internal.org.jline.keymap.KeyMap.display
import scalismo.geometry.{EuclideanVector, Landmark, Point, _3D, _}
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.mesh.TriangleMesh3D
import scalismo.registration.LandmarkRegistration
import scalismo.transformations._
import scalismo.ui.api.ScalismoUI

import java.io.File

object FemurRigidAlignment {

  def main(args: Array[String]): Unit = {



    scalismo.initialize()

    implicit val rng = scalismo.utils.Random(42)
    val ui = ScalismoUI()
    //Loading the rigid meshes and Landmarks
    val meshDir = "data/datasets/full-femurs/meshes/"
    val LMDir = "data/datasets/full-femurs/landmarks/"

    //loading the reference Mesh
    val referenceMesh: TriangleMesh3D = MeshIO.readMesh(new File("data/datasets/femur.stl")).get
    val meshGroup = ui.createGroup("RefMesh")
    val refmeshview = ui.show(meshGroup, referenceMesh, "Reference_Mesh")

    //loading reference landmarks
     val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new File("data/datasets/femur.json")).get
     val rigidFemurs = ui.createGroup("RigidFemurs")
     val alignedFemurs = ui.createGroup("AlignedFemurs")
     val femurFiles = new java.io.File(meshDir).listFiles()
     val meshSize = femurFiles.size

     //performing rigid alignment of all the meshes with corresponding landmarks
     (0 until meshSize).foreach {i=>
     val meshRigid: TriangleMesh3D = MeshIO.readMesh(new File(meshDir + i + ".stl")).get
     val rigidLandmarks = LandmarkIO.readLandmarksJson[ _3D ](new File(LMDir + i + ".json")).get
     val rigidTransform: RigidTransformation[ _3D ] = LandmarkRegistration.rigid3DLandmarkRegistration(rigidLandmarks, referenceLandmarks, center = Point(0, 0, 0))
     val alignedFemur = meshRigid.transform(rigidTransform)  // Storing Aligned Femurs
     val alignedLandmarks = rigidLandmarks.map(lm => lm.copy(point = rigidTransform(lm.point)))
     val rigidFemurView = ui.show(rigidFemurs, meshRigid, "Rigid_Femur")
     val alignedFemurView = ui.show(alignedFemurs, alignedFemur, "Aligned_Femur")
     rigidFemurView.color = java.awt.Color.red
     alignedFemurView.color = java.awt.Color.green

       //saving aligned meshes
    MeshIO.writeMesh(alignedFemur, new File("/data/datasets/alignedfemurs/meshes/" +i+".stl"))
       //saving aligned landmarks
    LandmarkIO.writeLandmarksJson[_3D](alignedLandmarks, new File("/data/datasets/alignedfemurs/landmarks/" + i + ".json"))


  }


}}