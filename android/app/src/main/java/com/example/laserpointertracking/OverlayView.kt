package com.example.laserpointertracking

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Transparent overlay that records and draws laser-hit circles.
 *
 * Mirrors the blue dots drawn by cv2.circle() in track_laser.py.
 * Points are stored in view coordinates; the caller must scale from
 * image coordinates before calling [addPoint].
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint().apply {
        color = Color.BLUE
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    private val hitPoints = mutableListOf<Pair<Float, Float>>()

    /** Record a new hit in view coordinates and redraw. Must be called on the main thread. */
    fun addPoint(x: Float, y: Float) {
        hitPoints.add(x to y)
        invalidate()
    }

    /** Clear all recorded hits and redraw. Must be called on the main thread. */
    fun clearPoints() {
        hitPoints.clear()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for ((x, y) in hitPoints) {
            canvas.drawCircle(x, y, CIRCLE_RADIUS, paint)
        }
    }

    companion object {
        private const val CIRCLE_RADIUS = 8f
    }
}
