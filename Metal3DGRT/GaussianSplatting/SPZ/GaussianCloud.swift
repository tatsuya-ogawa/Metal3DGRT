// spz-swift
// Swift port of https://github.com/nianticlabs/spz
// MIT License - Copyright (c) 2024 Niantic Labs

import Foundation

/// A point cloud composed of Gaussians.
///
/// Each gaussian is represented by:
/// - xyz position
/// - xyz scales (on log scale, compute exp(x) to get scale factor)
/// - xyzw quaternion (stored as x, y, z, w)
/// - alpha (before sigmoid activation, compute sigmoid(a) to get alpha value between 0 and 1)
/// - rgb color (as SH DC component, compute 0.5 + 0.282095 * x to get color value between 0 and 1)
/// - 0 to 45 spherical harmonics coefficients
public struct GaussianCloud: Sendable {
    /// Total number of points (gaussians) in this splat.
    public var numPoints: Int32 = 0

    /// Degree of spherical harmonics for this splat (0-3).
    public var shDegree: Int32 = 0

    /// Whether the gaussians should be rendered in antialiased mode (mip splatting).
    public var antialiased: Bool = false

    /// XYZ positions for each gaussian. Length = numPoints * 3.
    public var positions: [Float] = []

    /// XYZ scales on log scale. Compute exp(x) to get actual scale. Length = numPoints * 3.
    public var scales: [Float] = []

    /// XYZW quaternions for each gaussian (x, y, z, w ordering). Length = numPoints * 4.
    public var rotations: [Float] = []

    /// Alpha values before sigmoid activation. Length = numPoints.
    public var alphas: [Float] = []

    /// RGB colors as SH DC component. Length = numPoints * 3.
    public var colors: [Float] = []

    /// Spherical harmonics coefficients.
    ///
    /// The number of coefficients per point depends on shDegree:
    /// - 0 -> 0
    /// - 1 -> 9  (3 coeffs x 3 channels)
    /// - 2 -> 24 (8 coeffs x 3 channels)
    /// - 3 -> 45 (15 coeffs x 3 channels)
    ///
    /// The color channel is the inner (fastest varying) axis, and the coefficient is the outer
    /// (slower varying) axis. For degree 1, the order of the 9 values is:
    /// sh1n1_r, sh1n1_g, sh1n1_b, sh10_r, sh10_g, sh10_b, sh1p1_r, sh1p1_g, sh1p1_b
    public var sh: [Float] = []

    public init() {}

    public init(
        numPoints: Int32,
        shDegree: Int32,
        antialiased: Bool = false,
        positions: [Float] = [],
        scales: [Float] = [],
        rotations: [Float] = [],
        alphas: [Float] = [],
        colors: [Float] = [],
        sh: [Float] = []
    ) {
        self.numPoints = numPoints
        self.shDegree = shDegree
        self.antialiased = antialiased
        self.positions = positions
        self.scales = scales
        self.rotations = rotations
        self.alphas = alphas
        self.colors = colors
        self.sh = sh
    }

    /// Convert between two coordinate systems (performed in-place).
    public mutating func convertCoordinates(from: CoordinateSystem, to: CoordinateSystem) {
        guard numPoints > 0 else { return }

        let c = coordinateConverter(from: from, to: to)
        let pointCount = Int(numPoints)

        // Convert positions
        if c.flipP != SIMD3<Float>(1, 1, 1) {
            let flipP = c.flipP
            positions.withUnsafeMutableBufferPointer { ptr in
                DispatchQueue.concurrentPerform(iterations: pointCount) { i in
                    ptr[i * 3 + 0] *= flipP.x
                    ptr[i * 3 + 1] *= flipP.y
                    ptr[i * 3 + 2] *= flipP.z
                }
            }
        }

        // Convert rotations
        if c.flipQ != SIMD3<Float>(1, 1, 1) {
            let flipQ = c.flipQ
            rotations.withUnsafeMutableBufferPointer { ptr in
                DispatchQueue.concurrentPerform(iterations: pointCount) { i in
                    ptr[i * 4 + 0] *= flipQ.x
                    ptr[i * 4 + 1] *= flipQ.y
                    ptr[i * 4 + 2] *= flipQ.z
                }
            }
        }

        // Convert spherical harmonics
        if !sh.isEmpty {
            let numCoeffsPerPoint = sh.count / 3 / pointCount
            if numCoeffsPerPoint > 0 {
                let flipSh = c.flipSh
                sh.withUnsafeMutableBufferPointer { ptr in
                    DispatchQueue.concurrentPerform(iterations: pointCount) { i in
                        let baseIdx = i * numCoeffsPerPoint * 3
                        for j in 0..<numCoeffsPerPoint {
                            let f = flipSh[j]
                            if f != 1.0 {
                                ptr[baseIdx + j * 3 + 0] *= f
                                ptr[baseIdx + j * 3 + 1] *= f
                                ptr[baseIdx + j * 3 + 2] *= f
                            }
                        }
                    }
                }
            }
        }
    }

    /// Rotates the GaussianCloud by 180 degrees about the x axis.
    /// Converts from RUB to RDF coordinates and vice versa. Performed in-place.
    public mutating func rotate180DegAboutX() {
        convertCoordinates(from: .rub, to: .rdf)
    }

    /// Compute the median volume of the gaussian ellipsoids.
    public func medianVolume() -> Float {
        guard numPoints > 0 else { return 0.01 }

        // Volume of ellipsoid is 4/3 * pi * x * y * z, where x, y, z are radii.
        // Scales are on log scale: exp(x) * exp(y) * exp(z) = exp(x + y + z)
        // So we sort by (x + y + z) and compute volume later.
        var scaleSums: [Float] = []
        scaleSums.reserveCapacity(Int(numPoints))

        for i in stride(from: 0, to: scales.count, by: 3) {
            let sum = scales[i] + scales[i + 1] + scales[i + 2]
            scaleSums.append(sum)
        }

        scaleSums.sort()
        let median = scaleSums[scaleSums.count / 2]
        return (Float.pi * 4.0 / 3.0) * exp(median)
    }

    /// Validate that all arrays have the correct sizes.
    public func checkSizes() -> Bool {
        guard numPoints >= 0 else { return false }
        guard shDegree >= 0 && shDegree <= 3 else { return false }
        guard positions.count == Int(numPoints) * 3 else { return false }
        guard scales.count == Int(numPoints) * 3 else { return false }
        guard rotations.count == Int(numPoints) * 4 else { return false }
        guard alphas.count == Int(numPoints) else { return false }
        guard colors.count == Int(numPoints) * 3 else { return false }
        guard sh.count == Int(numPoints) * Int(dimForDegree(shDegree)) * 3 else { return false }
        return true
    }
}
