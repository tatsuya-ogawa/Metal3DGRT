// spz-swift
// Swift port of https://github.com/nianticlabs/spz
// MIT License - Copyright (c) 2024 Niantic Labs

import Foundation
import simd

/// Header structure for SPZ file format.
internal struct PackedGaussiansHeader {
    static let magic: UInt32 = 0x5053474e  // "NGSP" = Niantic Gaussian Splat
    static let currentVersion: UInt32 = 3

    var magic: UInt32 = PackedGaussiansHeader.magic
    var version: UInt32 = currentVersion
    var numPoints: UInt32 = 0
    var shDegree: UInt8 = 0
    var fractionalBits: UInt8 = 0
    var flags: UInt8 = 0
    var reserved: UInt8 = 0

    static let size = 16  // Total bytes: 4 + 4 + 4 + 1 + 1 + 1 + 1

    static let flagAntialiased: UInt8 = 0x1
}

/// Represents a single unpacked gaussian with full precision.
/// Each gaussian has 236 bytes of data.
public struct UnpackedGaussian: Sendable {
    public var position: SIMD3<Float> = .zero  // x, y, z
    public var rotation: SIMD4<Float> = SIMD4<Float>(0, 0, 0, 1)  // x, y, z, w
    public var scale: SIMD3<Float> = .zero     // log(scale)
    public var color: SIMD3<Float> = .zero     // rgb sh0 encoding
    public var alpha: Float = 0                 // inverse logistic
    public var shR: [Float] = Array(repeating: 0, count: 15)
    public var shG: [Float] = Array(repeating: 0, count: 15)
    public var shB: [Float] = Array(repeating: 0, count: 15)

    public init() {}
}

/// Represents a single packed gaussian in low precision format.
/// Each gaussian has at most 65 bytes.
public struct PackedGaussian: Sendable {
    public var position: [UInt8] = Array(repeating: 0, count: 9)
    public var rotation: [UInt8] = Array(repeating: 0, count: 4)
    public var scale: [UInt8] = Array(repeating: 0, count: 3)
    public var color: [UInt8] = Array(repeating: 0, count: 3)
    public var alpha: UInt8 = 0
    public var shR: [UInt8] = Array(repeating: 0, count: 15)
    public var shG: [UInt8] = Array(repeating: 0, count: 15)
    public var shB: [UInt8] = Array(repeating: 0, count: 15)

    public init() {}

    /// Unpack this gaussian to full precision.
    public func unpack(
        usesFloat16: Bool,
        usesQuaternionSmallestThree: Bool,
        fractionalBits: Int32,
        converter c: CoordinateConverter = CoordinateConverter()
    ) -> UnpackedGaussian {
        var result = UnpackedGaussian()

        // Decode position
        if usesFloat16 {
            // Legacy float16 format
            for i in 0..<3 {
                let halfValue = UInt16(position[i * 2]) | (UInt16(position[i * 2 + 1]) << 8)
                result.position[i] = c.flipP[i] * halfToFloat(halfValue)
            }
        } else {
            // 24-bit fixed point coordinates
            let scale = 1.0 / Float(1 << fractionalBits)
            for i in 0..<3 {
                var fixed32 = Int32(position[i * 3 + 0])
                fixed32 |= Int32(position[i * 3 + 1]) << 8
                fixed32 |= Int32(position[i * 3 + 2]) << 16
                // Sign extension
                if (fixed32 & 0x800000) != 0 {
                    fixed32 |= Int32(bitPattern: 0xff000000)
                }
                result.position[i] = c.flipP[i] * Float(fixed32) * scale
            }
        }

        // Decode scale
        for i in 0..<3 {
            result.scale[i] = Float(self.scale[i]) / 16.0 - 10.0
        }

        // Decode rotation
        if usesQuaternionSmallestThree {
            result.rotation = unpackQuaternionSmallestThree(rotation, converter: c)
        } else {
            result.rotation = unpackQuaternionFirstThree(Array(rotation.prefix(3)), converter: c)
        }

        // Decode alpha
        result.alpha = invSigmoid(Float(alpha) / 255.0)

        // Decode color
        for i in 0..<3 {
            result.color[i] = ((Float(color[i]) / 255.0) - 0.5) / colorScale
        }

        // Decode spherical harmonics
        for i in 0..<15 {
            result.shR[i] = c.flipSh[i] * unquantizeSH(shR[i])
            result.shG[i] = c.flipSh[i] * unquantizeSH(shG[i])
            result.shB[i] = c.flipSh[i] * unquantizeSH(shB[i])
        }

        return result
    }
}

/// Represents a full splat in packed low-precision format.
/// Data is stored non-interleaved for better compression.
public struct PackedGaussians: Sendable {
    public var numPoints: Int32 = 0
    public var shDegree: Int32 = 0
    public var fractionalBits: Int32 = 0
    public var antialiased: Bool = false
    public var usesQuaternionSmallestThree: Bool = true

    public var positions: [UInt8] = []
    public var scales: [UInt8] = []
    public var rotations: [UInt8] = []
    public var alphas: [UInt8] = []
    public var colors: [UInt8] = []
    public var sh: [UInt8] = []

    public init() {}

    /// Whether this uses legacy float16 position encoding.
    public var usesFloat16: Bool {
        return positions.count == Int(numPoints) * 3 * 2
    }

    /// Get a single packed gaussian at the given index.
    public func at(_ i: Int32) -> PackedGaussian {
        var result = PackedGaussian()

        let positionBytes = usesFloat16 ? 6 : 9
        let start3 = Int(i) * 3
        let posStart = Int(i) * positionBytes

        // Copy position bytes
        for j in 0..<positionBytes {
            result.position[j] = positions[posStart + j]
        }

        // Copy scale
        for j in 0..<3 {
            result.scale[j] = scales[start3 + j]
        }

        // Copy rotation
        let rotationBytes = usesQuaternionSmallestThree ? 4 : 3
        let rotStart = Int(i) * rotationBytes
        for j in 0..<rotationBytes {
            result.rotation[j] = rotations[rotStart + j]
        }

        // Copy color and alpha
        for j in 0..<3 {
            result.color[j] = colors[start3 + j]
        }
        result.alpha = alphas[Int(i)]

        // Copy spherical harmonics
        let shDim = Int(dimForDegree(shDegree))
        let shStart = Int(i) * shDim * 3
        for j in 0..<shDim {
            result.shR[j] = sh[shStart + j * 3 + 0]
            result.shG[j] = sh[shStart + j * 3 + 1]
            result.shB[j] = sh[shStart + j * 3 + 2]
        }
        // Fill remaining SH with neutral value
        for j in shDim..<15 {
            result.shR[j] = 128
            result.shG[j] = 128
            result.shB[j] = 128
        }

        return result
    }

    /// Unpack a single gaussian at the given index.
    public func unpack(_ i: Int32, converter c: CoordinateConverter = CoordinateConverter()) -> UnpackedGaussian {
        return at(i).unpack(
            usesFloat16: usesFloat16,
            usesQuaternionSmallestThree: usesQuaternionSmallestThree,
            fractionalBits: fractionalBits,
            converter: c
        )
    }

    /// Validate that all arrays have correct sizes.
    public func checkSizes() -> Bool {
        let usesFloat16 = self.usesFloat16
        let shDim = dimForDegree(shDegree)

        guard positions.count == Int(numPoints) * 3 * (usesFloat16 ? 2 : 3) else { return false }
        guard scales.count == Int(numPoints) * 3 else { return false }
        guard rotations.count == Int(numPoints) * (usesQuaternionSmallestThree ? 4 : 3) else { return false }
        guard alphas.count == Int(numPoints) else { return false }
        guard colors.count == Int(numPoints) * 3 else { return false }
        guard sh.count == Int(numPoints) * Int(shDim) * 3 else { return false }
        return true
    }
}

// MARK: - Quaternion Packing/Unpacking

/// Pack quaternion using "smallest three" method (4 bytes output).
internal func packQuaternionSmallestThree(_ rotation: [Float], converter c: CoordinateConverter) -> [UInt8] {
    // Normalize and apply coordinate conversion
    var q = normalized(Quat4f(rotation[0], rotation[1], rotation[2], rotation[3]))
    q.x *= c.flipQ.x
    q.y *= c.flipQ.y
    q.z *= c.flipQ.z

    // Find largest component
    var iLargest = 0
    for i in 1..<4 {
        if abs(q[i]) > abs(q[iLargest]) {
            iLargest = i
        }
    }

    // Transform so largest is positive (avoids sending sign bit)
    let negate = q[iLargest] < 0

    // Compress using sign bit and 9-bit precision per element
    var comp: UInt32 = UInt32(iLargest)
    for i in 0..<4 {
        if i != iLargest {
            let negbit: UInt32 = (q[i] < 0) != negate ? 1 : 0
            let mag = UInt32(Float((1 << 9) - 1) * (abs(q[i]) / sqrt1_2) + 0.5)
            comp = (comp << 10) | (negbit << 9) | mag
        }
    }

    // Little-endian output
    return [
        UInt8(comp & 0xff),
        UInt8((comp >> 8) & 0xff),
        UInt8((comp >> 16) & 0xff),
        UInt8((comp >> 24) & 0xff)
    ]
}

/// Unpack quaternion from "smallest three" format (4 bytes input).
internal func unpackQuaternionSmallestThree(_ r: [UInt8], converter c: CoordinateConverter = CoordinateConverter()) -> SIMD4<Float> {
    var comp = UInt32(r[0])
    comp |= UInt32(r[1]) << 8
    comp |= UInt32(r[2]) << 16
    comp |= UInt32(r[3]) << 24

    let cMask: UInt32 = (1 << 9) - 1

    let iLargest = Int(comp >> 30)
    var sumSquares: Float = 0
    var rotation = SIMD4<Float>(0, 0, 0, 0)

    for i in stride(from: 3, through: 0, by: -1) {
        if i != iLargest {
            let mag = comp & cMask
            let negbit = (comp >> 9) & 0x1
            comp = comp >> 10
            rotation[i] = sqrt1_2 * Float(mag) / Float(cMask)
            if negbit == 1 {
                rotation[i] = -rotation[i]
            }
            sumSquares += rotation[i] * rotation[i]
        }
    }
    rotation[iLargest] = sqrt(1.0 - sumSquares)

    // Apply coordinate conversion
    rotation.x *= c.flipQ.x
    rotation.y *= c.flipQ.y
    rotation.z *= c.flipQ.z

    return rotation
}

/// Unpack quaternion from "first three" format (3 bytes input, legacy).
internal func unpackQuaternionFirstThree(_ r: [UInt8], converter c: CoordinateConverter = CoordinateConverter()) -> SIMD4<Float> {
    var xyz = SIMD3<Float>(
        Float(r[0]),
        Float(r[1]),
        Float(r[2])
    ) * (1.0 / 127.5) + SIMD3<Float>(-1, -1, -1)

    xyz *= c.flipQ

    // Compute w from normalization constraint
    let w = sqrt(max(0.0, 1.0 - simd_dot(xyz, xyz)))

    return SIMD4<Float>(xyz.x, xyz.y, xyz.z, w)
}
