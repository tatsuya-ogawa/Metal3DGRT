// spz-swift
// Swift port of https://github.com/nianticlabs/spz
// MIT License - Copyright (c) 2024 Niantic Labs

import Foundation
import simd

// MARK: - Type Aliases

public typealias Vec3f = SIMD3<Float>
public typealias Quat4f = SIMD4<Float>  // w, x, y, z ordering
public typealias Half = UInt16

// MARK: - Constants

internal let sqrt1_2: Float = 0.7071067811865476  // 1/sqrt(2)
internal let colorScale: Float = 0.15

// MARK: - Half Precision Conversion

/// Convert a half-precision float (Float16 as UInt16) to Float
public func halfToFloat(_ h: Half) -> Float {
    let sgn = (h >> 15) & 0x1
    let exponent = (h >> 10) & 0x1f
    let mantissa = h & 0x3ff

    let signMul: Float = sgn == 1 ? -1.0 : 1.0

    if exponent == 0 {
        // Subnormal numbers
        return signMul * powf(2.0, -14.0) * Float(mantissa) / 1024.0
    }

    if exponent == 31 {
        // Infinity or NaN
        return mantissa != 0 ? Float.nan : signMul * Float.infinity
    }

    // Normal numbers
    return signMul * powf(2.0, Float(exponent) - 15.0) * (1.0 + Float(mantissa) / 1024.0)
}

/// Convert a Float to half-precision (Float16 as UInt16)
public func floatToHalf(_ f: Float) -> Half {
    let f32 = f.bitPattern
    let sign = Int32((f32 >> 31) & 0x01)
    let exponent = Int32((f32 >> 23) & 0xff)
    let mantissa = Int32(f32 & 0x7fffff)

    // Handle inf and nan
    if exponent == 0xFF {
        if mantissa == 0 {
            return Half(sign << 15) | 0x7C00  // Inf
        }
        return Half(sign << 15) | 0x7C01  // NaN
    }

    let centeredExp = exponent - 127

    // If exponent is greater than half range, return +/- Inf
    if centeredExp > 15 {
        return Half(sign << 15) | 0x7C00
    }

    // Normal numbers
    if centeredExp > -15 {
        return Half(sign << 15) | Half((centeredExp + 15) << 10) | Half(mantissa >> 13)
    }

    // Subnormal numbers
    let fullMantissa = 0x800000 | mantissa
    let shift = -(centeredExp + 14)
    let newMantissa = fullMantissa >> shift
    return Half(sign << 15) | Half(newMantissa >> 13)
}

// MARK: - Vector Helpers

public func squaredNorm(_ v: Vec3f) -> Float {
    return simd_dot(v, v)
}

public func norm(_ v: Vec3f) -> Float {
    return simd_length(v)
}

public func normalized(_ v: Vec3f) -> Vec3f {
    return simd_normalize(v)
}

// MARK: - Quaternion Helpers

/// Quaternion norm (w, x, y, z ordering)
public func norm(_ q: Quat4f) -> Float {
    return sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
}

/// Normalize quaternion (w, x, y, z ordering)
public func normalized(_ q: Quat4f) -> Quat4f {
    let n = norm(q)
    return Quat4f(q.x / n, q.y / n, q.z / n, q.w / n)
}

/// Rotate vector by quaternion (w, x, y, z ordering)
public func times(_ q: Quat4f, _ p: Vec3f) -> Vec3f {
    let w = q.w, x = q.x, y = q.y, z = q.z
    let vx = p.x, vy = p.y, vz = p.z

    let x2 = x + x
    let y2 = y + y
    let z2 = z + z
    let wx2 = w * x2
    let wy2 = w * y2
    let wz2 = w * z2
    let xx2 = x * x2
    let xy2 = x * y2
    let xz2 = x * z2
    let yy2 = y * y2
    let yz2 = y * z2
    let zz2 = z * z2

    return Vec3f(
        vx * (1.0 - (yy2 + zz2)) + vy * (xy2 - wz2) + vz * (xz2 + wy2),
        vx * (xy2 + wz2) + vy * (1.0 - (xx2 + zz2)) + vz * (yz2 - wx2),
        vx * (xz2 - wy2) + vy * (yz2 + wx2) + vz * (1.0 - (xx2 + yy2))
    )
}

/// Multiply quaternions (w, x, y, z ordering)
public func times(_ a: Quat4f, _ b: Quat4f) -> Quat4f {
    let w = a.w, x = a.x, y = a.y, z = a.z
    let qw = b.w, qx = b.x, qy = b.y, qz = b.z

    return normalized(Quat4f(
        w * qx + x * qw + y * qz - z * qy,
        w * qy - x * qz + y * qw + z * qx,
        w * qz + x * qy - y * qx + z * qw,
        w * qw - x * qx - y * qy - z * qz
    ))
}

/// Scale quaternion
public func times(_ q: Quat4f, _ s: Float) -> Quat4f {
    return Quat4f(q.x * s, q.y * s, q.z * s, q.w * s)
}

/// Add quaternions
public func plus(_ a: Quat4f, _ b: Quat4f) -> Quat4f {
    return Quat4f(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
}

/// Scale vector
public func times(_ v: Vec3f, _ s: Float) -> Vec3f {
    return v * s
}

/// Add vectors
public func plus(_ a: Vec3f, _ b: Vec3f) -> Vec3f {
    return a + b
}

/// Element-wise multiply vectors
public func times(_ a: Vec3f, _ b: Vec3f) -> Vec3f {
    return a * b
}

/// Convert scaled axis representation to quaternion
public func axisAngleQuat(_ scaledAxis: Vec3f) -> Quat4f {
    let a0 = scaledAxis.x
    let a1 = scaledAxis.y
    let a2 = scaledAxis.z
    let thetaSquared = a0 * a0 + a1 * a1 + a2 * a2

    if thetaSquared > 0.0 {
        let theta = sqrt(thetaSquared)
        let halfTheta = theta * 0.5
        let k = sin(halfTheta) / theta
        return normalized(Quat4f(a0 * k, a1 * k, a2 * k, cos(halfTheta)))
    }

    // Taylor series approximation for small angles
    let k: Float = 0.5
    return normalized(Quat4f(a0 * k, a1 * k, a2 * k, 1.0))
}

// MARK: - Clamping and Conversion Helpers

internal func toUint8(_ x: Float) -> UInt8 {
    return UInt8(clamping: Int(round(x).clamped(to: 0...255)))
}

internal func sigmoid(_ x: Float) -> Float {
    return 1.0 / (1.0 + exp(-x))
}

internal func invSigmoid(_ x: Float) -> Float {
    return log(x / (1.0 - x))
}

/// Quantize spherical harmonic coefficient to 8 bits with bucket rounding
internal func quantizeSH(_ x: Float, bucketSize: Int32) -> UInt8 {
    var q = Int32(round(x * 128.0) + 128.0)
    q = (q + bucketSize / 2) / bucketSize * bucketSize
    return UInt8(clamping: q.clamped(to: 0...255))
}

/// Unquantize spherical harmonic coefficient from 8 bits
internal func unquantizeSH(_ x: UInt8) -> Float {
    return (Float(x) - 128.0) / 128.0
}

// MARK: - SH Degree Helpers

internal func degreeForDim(_ dim: Int32) -> Int32 {
    if dim < 3 { return 0 }
    if dim < 8 { return 1 }
    if dim < 15 { return 2 }
    return 3
}

internal func dimForDegree(_ degree: Int32) -> Int32 {
    switch degree {
    case 0: return 0
    case 1: return 3
    case 2: return 8
    case 3: return 15
    default: return 0
    }
}

// MARK: - Extensions

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}

extension Int32 {
    func clamped(to range: ClosedRange<Int32>) -> Int32 {
        return Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
