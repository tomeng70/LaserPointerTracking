package com.example.laserpointertracking

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.SoundPool
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.laserpointertracking.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var soundPool: SoundPool? = null
    private var hitSoundId: Int = 0
    private var soundLoaded = false

    private val params = DetectionParams()
    private var prevState = 0
    private var lastHitTimeMs = Long.MIN_VALUE / 2

    companion object {
        private const val TAG = "LaserTracker"
        private const val REQUEST_CODE_CAMERA = 10
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialisation failed")
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_LONG).show()
            return
        }

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initSoundPool()
        initControls()

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (hasCameraPermission()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), REQUEST_CODE_CAMERA
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        soundPool?.release()
    }

    // ── Sound ─────────────────────────────────────────────────────────────────

    private fun initSoundPool() {
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_GAME)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
        soundPool = SoundPool.Builder().setMaxStreams(3).setAudioAttributes(attrs).build()
        soundPool?.setOnLoadCompleteListener { _, _, status -> soundLoaded = (status == 0) }

        val resId = resources.getIdentifier("hit", "raw", packageName)
        if (resId != 0) hitSoundId = soundPool?.load(this, resId, 1) ?: 0
    }

    private fun playHit() {
        if (soundLoaded && hitSoundId != 0) soundPool?.play(hitSoundId, 1f, 1f, 0, 0, 1f)
    }

    // ── UI ────────────────────────────────────────────────────────────────────

    private fun initControls() {
        binding.btnClear.setOnClickListener { binding.overlayView.clearPoints() }

        binding.fabSettings.setOnClickListener {
            binding.settingsPanel.visibility =
                if (binding.settingsPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        // SeekBar: Min Area  (range 0–50, default 3)
        bindSeekBar(binding.sbMinArea, min = 0, max = 50, initial = params.minArea.toInt(),
            label = binding.tvMinArea, fmt = { "Min Area: $it px²" }
        ) { params.minArea = it.toDouble() }

        // SeekBar: Max Area  (range 10–1000, default 300)
        bindSeekBar(binding.sbMaxArea, min = 10, max = 1000, initial = params.maxArea.toInt(),
            label = binding.tvMaxArea, fmt = { "Max Area: $it px²" }
        ) { params.maxArea = it.toDouble() }

        // SeekBar: Min Circularity  (0–100 maps to 0.00–1.00, default 35)
        bindSeekBar(binding.sbMinCirc, min = 0, max = 100,
            initial = (params.minCircularity * 100).toInt(),
            label = binding.tvMinCirc, fmt = { "Min Circularity: ${"%.2f".format(it / 100.0)}" }
        ) { params.minCircularity = it / 100.0 }

        // SeekBar: Min Redness Abs  (0–50, default 10)
        bindSeekBar(binding.sbMinRedness, min = 0, max = 50, initial = params.minRednessAbs,
            label = binding.tvMinRedness, fmt = { "Min Redness Abs: $it" }
        ) { params.minRednessAbs = it }
    }

    private fun bindSeekBar(
        sb: SeekBar,
        min: Int, max: Int, initial: Int,
        label: TextView,
        fmt: (Int) -> String,
        onChange: (Int) -> Unit
    ) {
        sb.max = max - min
        sb.progress = initial - min
        label.text = fmt(initial)
        sb.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, progress: Int, fromUser: Boolean) {
                val value = progress + min
                label.text = fmt(value)
                onChange(value)
            }
            override fun onStartTrackingTouch(sb: SeekBar) = Unit
            override fun onStopTrackingTouch(sb: SeekBar) = Unit
        })
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().apply {
                surfaceProvider = binding.viewFinder.surfaceProvider
            }

            val analyzer = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .apply { setAnalyzer(cameraExecutor, ::processFrame) }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ── Frame analysis ────────────────────────────────────────────────────────

    private fun processFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        val imgW = bitmap.width
        val imgH = bitmap.height

        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        bitmap.recycle()

        val result = LaserDetector.detect(mat, params)
        mat.release()

        val now = SystemClock.elapsedRealtime()

        if (result.center != null) {
            if (prevState == 0 && (now - lastHitTimeMs) >= params.hitCooldownMs) {
                prevState = 1
                lastHitTimeMs = now
                playHit()

                // Scale image coords → view coords and record the hit point
                val cx = result.center.x.toFloat()
                val cy = result.center.y.toFloat()
                runOnUiThread {
                    val vw = binding.overlayView.width.toFloat()
                    val vh = binding.overlayView.height.toFloat()
                    if (vw > 0 && vh > 0) {
                        binding.overlayView.addPoint(cx * vw / imgW, cy * vh / imgH)
                    }
                }
            } else {
                prevState = 1
            }
        } else {
            prevState = 0
        }

        imageProxy.close()
    }

    // ── Permissions ───────────────────────────────────────────────────────────

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_CAMERA) {
            if (hasCameraPermission()) startCamera()
            else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }
}
