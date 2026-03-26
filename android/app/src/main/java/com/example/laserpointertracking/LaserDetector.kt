package com.example.laserpointertracking

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.max

/**
 * Result of processing a single camera frame.
 *
 * @param center Best laser-dot centre in image-pixel coordinates, or null when nothing detected.
 * @param contour Winning contour (useful for debug overlays), or null.
 */
data class DetectionResult(
    val center: Point?,
    val contour: MatOfPoint?
)

/**
 * Stateless image-processing pipeline ported directly from track_laser.py (splitHSV branch).
 *
 * Input: RGBA [Mat] as produced by CameraX OUTPUT_IMAGE_FORMAT_RGBA_8888.
 * All intermediate Mats are released before returning.
 */
object LaserDetector {

    /**
     * Run the full detection pipeline on one frame.
     *
     * Python equivalent:
     *   b, g, r = cv2.split(frame)
     *   max_gb  = cv2.max(g, b)
     *   redness = cv2.subtract(r, max_gb)
     *   thr     = max(percentile(redness, PCT), MIN_REDNESS_ABS)
     *   laser   = threshold + morphology
     *   center  = findCenterBest(laser)
     */
    fun detect(rgbaMat: Mat, params: DetectionParams): DetectionResult {
        // ── 1. Split RGBA → individual channels ──────────────────────────────
        val channels = ArrayList<Mat>()
        Core.split(rgbaMat, channels)
        val r = channels[0]
        val g = channels[1]
        val b = channels[2]
        // channels[3] = alpha – ignored

        // ── 2. redness = R − max(G, B) ────────────────────────────────────────
        //    Suppresses white/grey highlights and non-red colours.
        val maxGB = Mat()
        Core.max(g, b, maxGB)
        val redness = Mat()
        Core.subtract(r, maxGB, redness)

        // ── 3. Dynamic threshold (percentile) ────────────────────────────────
        val thr = max(
            computePercentile(redness, params.percentileThresh),
            params.minRednessAbs.toDouble()
        )
        val laser = Mat()
        Imgproc.threshold(redness, laser, thr, 255.0, Imgproc.THRESH_BINARY)

        // ── 4. Morphological cleanup ──────────────────────────────────────────
        val ksize = if (params.kernelSize % 2 == 0) params.kernelSize + 1 else params.kernelSize
        val k = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(ksize.toDouble(), ksize.toDouble())
        )
        if (params.morphOpenIters > 0) {
            Imgproc.morphologyEx(laser, laser, Imgproc.MORPH_OPEN,
                k, Point(-1.0, -1.0), params.morphOpenIters)
        }
        if (params.morphCloseIters > 0) {
            Imgproc.morphologyEx(laser, laser, Imgproc.MORPH_CLOSE,
                k, Point(-1.0, -1.0), params.morphCloseIters)
        }

        // ── 5. Find best blob ─────────────────────────────────────────────────
        val result = findBestContour(laser, params)

        // ── Cleanup ───────────────────────────────────────────────────────────
        channels.forEach { it.release() }
        maxGB.release()
        redness.release()
        laser.release()
        k.release()

        return result
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    /**
     * Compute the pixel value at [percentile] (0–100) across a single-channel [Mat].
     * Equivalent to numpy.percentile(redness, PERCENTILE_THRESH).
     */
    private fun computePercentile(mat: Mat, percentile: Double): Double {
        val total = mat.rows() * mat.cols()
        if (total == 0) return 0.0
        val raw = ByteArray(total)
        mat.get(0, 0, raw)
        val values = IntArray(total) { raw[it].toInt() and 0xFF }
        values.sort()
        val idx = ((percentile / 100.0) * total).toInt().coerceIn(0, total - 1)
        return values[idx].toDouble()
    }

    /**
     * Port of findCenterBest() from track_laser.py.
     *
     * Scores each passing contour as:  area × (0.5 + circularity)
     * and returns the highest-scoring one.
     */
    private fun findBestContour(mask: Mat, params: DetectionParams): DetectionResult {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        // Clone: some OpenCV builds modify the source during findContours
        Imgproc.findContours(
            mask.clone(), contours, hierarchy,
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
        )
        hierarchy.release()

        var bestCenter: Point? = null
        var bestContour: MatOfPoint? = null
        var bestScore = -1.0

        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < params.minArea || area > params.maxArea) continue

            val pts2f = MatOfPoint2f(*c.toArray())
            val perim = Imgproc.arcLength(pts2f, true)
            pts2f.release()
            if (perim <= 0.0) continue

            val circularity = (4.0 * PI * area) / (perim * perim)
            if (circularity < params.minCircularity) continue

            val m = Imgproc.moments(c)
            val center = if (m.m00 > 0) {
                Point(m.m10 / m.m00, m.m01 / m.m00)
            } else {
                val rect = Imgproc.boundingRect(c)
                Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0)
            }

            val score = area * (0.5 + circularity)
            if (score > bestScore) {
                bestScore = score
                bestCenter = center
                bestContour = c
            }
        }

        return DetectionResult(bestCenter, bestContour)
    }
}
