// spz-swift
// Swift port of https://github.com/nianticlabs/spz
// MIT License - Copyright (c) 2024 Niantic Labs

import Foundation

/// Coordinate system conventions for 3D Gaussian splats.
/// The naming convention is: first letter is X direction, second is Y, third is Z.
/// R/L = Right/Left, U/D = Up/Down, F/B = Front/Back
public enum CoordinateSystem: Int, Sendable {
    case unspecified = 0
    case ldb = 1  // Left Down Back
    case rdb = 2  // Right Down Back
    case lub = 3  // Left Up Back
    case rub = 4  // Right Up Back (Three.js coordinate system)
    case ldf = 5  // Left Down Front
    case rdf = 6  // Right Down Front (PLY coordinate system)
    case luf = 7  // Left Up Front (GLB coordinate system)
    case ruf = 8  // Right Up Front (Unity coordinate system)
}

/// Coordinate conversion helper that stores axis flips for position, quaternion, and SH coefficients.
public struct CoordinateConverter: Sendable {
    /// Position axis flips (x, y, z)
    public var flipP: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 1.0)

    /// Quaternion component flips (x, y, z - w is never flipped)
    public var flipQ: SIMD3<Float> = SIMD3<Float>(1.0, 1.0, 1.0)

    /// Flips for the 15 spherical harmonics coefficients
    public var flipSh: [Float] = Array(repeating: 1.0, count: 15)

    public init() {}

    public init(flipP: SIMD3<Float>, flipQ: SIMD3<Float>, flipSh: [Float]) {
        self.flipP = flipP
        self.flipQ = flipQ
        self.flipSh = flipSh
    }
}

/// Determine which axes match between two coordinate systems
internal func axesMatch(_ a: CoordinateSystem, _ b: CoordinateSystem) -> (Bool, Bool, Bool) {
    let aNum = a.rawValue - 1
    let bNum = b.rawValue - 1

    if aNum < 0 || bNum < 0 {
        return (true, true, true)
    }

    return (
        ((aNum >> 0) & 1) == ((bNum >> 0) & 1),
        ((aNum >> 1) & 1) == ((bNum >> 1) & 1),
        ((aNum >> 2) & 1) == ((bNum >> 2) & 1)
    )
}

/// Create a coordinate converter between two coordinate systems
public func coordinateConverter(from: CoordinateSystem, to: CoordinateSystem) -> CoordinateConverter {
    let (xMatch, yMatch, zMatch) = axesMatch(from, to)
    let x: Float = xMatch ? 1.0 : -1.0
    let y: Float = yMatch ? 1.0 : -1.0
    let z: Float = zMatch ? 1.0 : -1.0

    return CoordinateConverter(
        flipP: SIMD3<Float>(x, y, z),
        flipQ: SIMD3<Float>(y * z, x * z, x * y),
        flipSh: [
            y,          // 0
            z,          // 1
            x,          // 2
            x * y,      // 3
            y * z,      // 4
            1.0,        // 5
            x * z,      // 6
            1.0,        // 7
            y,          // 8
            x * y * z,  // 9
            y,          // 10
            z,          // 11
            x,          // 12
            z,          // 13
            x           // 14
        ]
    )
}
