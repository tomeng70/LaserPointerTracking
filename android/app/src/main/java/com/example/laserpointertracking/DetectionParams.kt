package com.example.laserpointertracking

/**
 * Tunable detection parameters – mirrors the global constants in track_laser.py.
 * All fields are vars so they can be adjusted live from the UI.
 */
data class DetectionParams(
    var minArea: Double = 3.0,
    var maxArea: Double = 300.0,
    var minCircularity: Double = 0.35,
    var percentileThresh: Double = 99.55,
    var minRednessAbs: Int = 10,
    var kernelSize: Int = 3,
    var morphOpenIters: Int = 1,
    var morphCloseIters: Int = 1,
    /** Minimum milliseconds between two registered hits (mirrors HIT_COOLDOWN_SEC). */
    var hitCooldownMs: Long = 60L
)
