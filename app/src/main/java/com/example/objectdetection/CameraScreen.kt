package com.example.objectdetection

import android.graphics.*
import android.graphics.pdf.PdfDocument
import android.util.Log
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

@Composable
fun CameraScreen(
    saveNextDetection: Boolean,
    onCaptureRequest: () -> Unit,
    resetCaptureFlag: () -> Unit
) {
    val context = LocalContext.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    val previewView = remember { PreviewView(context) }

    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            val options = ObjectDetectorOptions.Builder()
                .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
                .enableMultipleObjects()
                .build()

            val objectDetector = ObjectDetection.getClient(options)

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val mediaImage = imageProxy.image
                if (mediaImage != null) {
                    val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

                    objectDetector.process(inputImage)
                        .addOnSuccessListener { objects ->
                            if (saveNextDetection && objects.isNotEmpty()) {
                                val box = objects.first().boundingBox
                                val bitmap = imageProxy.toBitmap()
                                val cropped = cropBitmap(bitmap, box)
                                saveBitmapAsPDF(context.cacheDir, cropped, context)
                                resetCaptureFlag()
                            }
                        }
                        .addOnCompleteListener { imageProxy.close() }
                } else {
                    imageProxy.close()
                }
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    context as androidx.lifecycle.LifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalysis
                )
            } catch (exc: Exception) {
                Log.e("CameraScreen", "Camera binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    Column(modifier = Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.weight(1f))

        Button(
            onClick = { onCaptureRequest() },
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
                .padding(8.dp)
        ) {
            Text("Capture PDF")
        }
    }
}

private fun cropBitmap(bitmap: Bitmap, rect: Rect): Bitmap {
    val safeLeft = rect.left.coerceAtLeast(0)
    val safeTop = rect.top.coerceAtLeast(0)
    val safeWidth = rect.width().coerceAtMost(bitmap.width - safeLeft)
    val safeHeight = rect.height().coerceAtMost(bitmap.height - safeTop)
    return Bitmap.createBitmap(bitmap, safeLeft, safeTop, safeWidth, safeHeight)
}

private fun saveBitmapAsPDF(dir: File, bitmap: Bitmap, context: android.content.Context) {
    val pdfDocument = PdfDocument()
    val pageInfo = PdfDocument.PageInfo.Builder(bitmap.width, bitmap.height, 1).create()
    val page = pdfDocument.startPage(pageInfo)
    page.canvas.drawBitmap(bitmap, 0f, 0f, null)
    pdfDocument.finishPage(page)

    val file = File(dir, "detected_rectangle.pdf")
    FileOutputStream(file).use { fos ->
        pdfDocument.writeTo(fos)
    }
    pdfDocument.close()

    Toast.makeText(context, "PDF saved: ${file.absolutePath}", Toast.LENGTH_LONG).show()
}
