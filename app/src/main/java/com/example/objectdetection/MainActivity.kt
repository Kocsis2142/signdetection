package com.example.objectdetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.pdf.PdfDocument
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var imageCapture: ImageCapture? = null
    private var saveNextDetection = mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        setContent {
            CameraScreen(
                saveNextDetection = saveNextDetection.value,
                onCaptureRequest = { saveNextDetection.value = true },
                resetCaptureFlag = { saveNextDetection.value = false }
            )
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (!allPermissionsGranted()) {
            Toast.makeText(this, "Permissions not granted", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    fun findFrameRect(bitmap: Bitmap): RotatedRect? {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)

        val binary = Mat()
        Imgproc.adaptiveThreshold(
            gray, binary, 255.0,
            Imgproc.ADAPTIVE_THRESH_MEAN_C,
            Imgproc.THRESH_BINARY_INV,
            15, 10.0
        )

        // 🔥 Szignó eltüntetése → csak a keret maradjon
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(5.0, 5.0))
        Imgproc.erode(binary, binary, kernel)
        Imgproc.dilate(binary, binary, kernel)

        val contours = mutableListOf<MatOfPoint>()
        Imgproc.findContours(binary, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        if (contours.isEmpty()) return null

        val imageArea = bitmap.width * bitmap.height
        var bestRect: RotatedRect? = null
        var bestArea = 0.0

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < imageArea * 0.05) continue

            val rect = Imgproc.minAreaRect(MatOfPoint2f(*c.toArray()))
            if (area > bestArea) {
                bestArea = area
                bestRect = rect
            }
        }
        return bestRect
    }
    fun isSignatureInsideFrame(bitmap: Bitmap, frameRect: RotatedRect): Boolean {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(3.0, 3.0), 0.0)

        val binary = Mat()
        Imgproc.threshold(gray, binary, 150.0, 255.0, Imgproc.THRESH_BINARY_INV)

        // Keret mask
        val mask = Mat.zeros(binary.size(), CvType.CV_8UC1)
        val boxPoints = arrayOf(Point(), Point(), Point(), Point())
        frameRect.points(boxPoints)
        val poly = MatOfPoint(*boxPoints)
        Imgproc.fillConvexPoly(mask, poly, Scalar(255.0))

        // Megnézzük, van-e fekete pixel a kereten kívül
        val outside = Mat()
        Core.bitwise_and(binary, Scalar(255.0), outside, Core.bitwise_not(mask))

        val nonZero = Core.countNonZero(outside)
        return nonZero < 500  // állítható küszöb
    }
    fun cropRotatedRect(bitmap: Bitmap, rect: RotatedRect): Bitmap {
        val srcMat = Mat()
        Utils.bitmapToMat(bitmap, srcMat)

        val points = arrayOf(Point(), Point(), Point(), Point())
        rect.points(points)

        val srcPts = MatOfPoint2f(*points)
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(rect.size.width, 0.0),
            Point(rect.size.width, rect.size.height),
            Point(0.0, rect.size.height)
        )

        val matrix = Imgproc.getPerspectiveTransform(srcPts, dstPts)
        val dstMat = Mat(Size(rect.size.width, rect.size.height), srcMat.type())

        Imgproc.warpPerspective(srcMat, dstMat, matrix, dstMat.size())

        val cropped = Bitmap.createBitmap(dstMat.cols(), dstMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(dstMat, cropped)
        return cropped
    }
    fun drawSignatureDebug(bitmap: Bitmap, frameRect: RotatedRect, maxOutsideRatio: Double = 0.1): Bitmap {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        Imgproc.threshold(gray, gray, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)

        // ✅ Keret pontok átalakítása
        val boxPoints = MatOfPoint()
        val rectPoints = Array(4) { Point() }
        frameRect.points(rectPoints)
        boxPoints.fromArray(*rectPoints)

        // ✅ Maszk a keretre
        val mask = Mat.zeros(mat.size(), CvType.CV_8UC1)
        Imgproc.fillConvexPoly(mask, boxPoints, Scalar(255.0))

        // ✅ Kívül eső pixelek
        val outside = Mat()
        Core.bitwise_and(gray, gray, outside, mask.inv())

        // ✅ Színes debug kép
        val debugMat = Mat.zeros(mat.size(), CvType.CV_8UC3)

        val blue = Mat(debugMat.size(), debugMat.type(), Scalar(255.0, 0.0, 0.0))
        blue.copyTo(debugMat, gray)

        val red = Mat(debugMat.size(), debugMat.type(), Scalar(0.0, 0.0, 255.0))
        red.copyTo(debugMat, outside)

        Imgproc.polylines(debugMat, listOf(boxPoints), true, Scalar(0.0, 255.0, 0.0), 3)

        // ✅ Outside ratio számítás
        val totalSignaturePixels = Core.countNonZero(gray)
        val outsidePixels = Core.countNonZero(outside)
        val outsideRatio = if (totalSignaturePixels == 0) 0.0 else outsidePixels.toDouble() / totalSignaturePixels

        // ✅ Szöveg ráírás
        Imgproc.putText(
            debugMat,
            "Outside: %.2f (max: %.2f)".format(outsideRatio, maxOutsideRatio),
            Point(30.0, 50.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            1.2,
            if (outsideRatio > maxOutsideRatio) Scalar(0.0, 0.0, 255.0) else Scalar(0.0, 255.0, 0.0),
            2
        )

        val result = Bitmap.createBitmap(debugMat.width(), debugMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(debugMat, result)
        return result
    }



}
