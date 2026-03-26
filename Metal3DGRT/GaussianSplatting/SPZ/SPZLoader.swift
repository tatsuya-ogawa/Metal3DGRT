// spz-swift
// Swift port of https://github.com/nianticlabs/spz
// MIT License - Copyright (c) 2024 Niantic Labs

import Foundation

/// Options for unpacking SPZ data.
public struct UnpackOptions: Sendable {
    /// Target coordinate system for the unpacked data.
    public var to: CoordinateSystem

    public init(to: CoordinateSystem = .unspecified) {
        self.to = to
    }
}

// MARK: - Loading Functions

/// Load a GaussianCloud from SPZ data.
/// - Parameters:
///   - data: The SPZ file data (gzip compressed).
///   - options: Unpacking options including target coordinate system.
/// - Returns: The unpacked GaussianCloud.
public func loadSpz(_ data: Data, options: UnpackOptions = UnpackOptions()) throws -> GaussianCloud {
    let packed = try loadSpzPacked(data)
    return unpackGaussians(packed, options: options)
}

/// Load a GaussianCloud from an SPZ file.
/// - Parameters:
///   - url: URL to the SPZ file.
///   - options: Unpacking options including target coordinate system.
/// - Returns: The unpacked GaussianCloud.
public func loadSpz(from url: URL, options: UnpackOptions = UnpackOptions()) throws -> GaussianCloud {
    let data = try Data(contentsOf: url)
    return try loadSpz(data, options: options)
}

/// Load packed gaussians from SPZ data (without unpacking).
/// - Parameter data: The SPZ file data (gzip compressed).
/// - Returns: The packed gaussians structure.
public func loadSpzPacked(_ data: Data) throws -> PackedGaussians {
    let decompressed = try decompressGzip(data)
    return try deserializePackedGaussians(decompressed)
}

/// Load packed gaussians from an SPZ file.
/// - Parameter url: URL to the SPZ file.
/// - Returns: The packed gaussians structure.
public func loadSpzPacked(from url: URL) throws -> PackedGaussians {
    let data = try Data(contentsOf: url)
    return try loadSpzPacked(data)
}

// MARK: - Deserialization

/// Deserialize packed gaussians from uncompressed data.
internal func deserializePackedGaussians(_ data: Data) throws -> PackedGaussians {
    let maxPointsToRead: Int32 = 10_000_000

    guard data.count >= PackedGaussiansHeader.size else {
        throw SPZError.invalidHeader
    }

    // Read header
    var offset = 0

    let magic = data.withUnsafeBytes { ptr -> UInt32 in
        ptr.load(fromByteOffset: offset, as: UInt32.self)
    }
    offset += 4

    guard magic == PackedGaussiansHeader.magic else {
        throw SPZError.invalidHeader
    }

    let version = data.withUnsafeBytes { ptr -> UInt32 in
        ptr.load(fromByteOffset: offset, as: UInt32.self)
    }
    offset += 4

    guard version >= 1 && version <= 3 else {
        throw SPZError.unsupportedVersion(Int(version))
    }

    let numPoints = data.withUnsafeBytes { ptr -> UInt32 in
        ptr.load(fromByteOffset: offset, as: UInt32.self)
    }
    offset += 4

    guard numPoints <= maxPointsToRead else {
        throw SPZError.tooManyPoints(Int(numPoints))
    }

    let shDegree = data[offset]
    offset += 1

    guard shDegree <= 3 else {
        throw SPZError.unsupportedSHDegree(Int(shDegree))
    }

    let fractionalBits = data[offset]
    offset += 1

    let flags = data[offset]
    offset += 1

    // Skip reserved byte
    offset += 1

    // Calculate sizes
    let usesFloat16 = version == 1
    let usesQuaternionSmallestThree = version >= 3
    let shDim = dimForDegree(Int32(shDegree))

    let positionBytes = Int(numPoints) * 3 * (usesFloat16 ? 2 : 3)
    let scaleBytes = Int(numPoints) * 3
    let rotationBytes = Int(numPoints) * (usesQuaternionSmallestThree ? 4 : 3)
    let alphaBytes = Int(numPoints)
    let colorBytes = Int(numPoints) * 3
    let shBytes = Int(numPoints) * Int(shDim) * 3

    let totalBytes = offset + positionBytes + alphaBytes + colorBytes + scaleBytes + rotationBytes + shBytes

    guard data.count >= totalBytes else {
        throw SPZError.readError
    }

    // Read data arrays
    var result = PackedGaussians()
    result.numPoints = Int32(numPoints)
    result.shDegree = Int32(shDegree)
    result.fractionalBits = Int32(fractionalBits)
    result.antialiased = (flags & PackedGaussiansHeader.flagAntialiased) != 0
    result.usesQuaternionSmallestThree = usesQuaternionSmallestThree

    result.positions = Array(data[offset..<(offset + positionBytes)])
    offset += positionBytes

    result.alphas = Array(data[offset..<(offset + alphaBytes)])
    offset += alphaBytes

    result.colors = Array(data[offset..<(offset + colorBytes)])
    offset += colorBytes

    result.scales = Array(data[offset..<(offset + scaleBytes)])
    offset += scaleBytes

    result.rotations = Array(data[offset..<(offset + rotationBytes)])
    offset += rotationBytes

    result.sh = Array(data[offset..<(offset + shBytes)])

    return result
}

// MARK: - Unpacking

public func unpackGaussians(_ packed: PackedGaussians, options: UnpackOptions = UnpackOptions()) -> GaussianCloud {
    let numPoints = packed.numPoints
    let shDim = dimForDegree(packed.shDegree)
    let usesFloat16 = packed.usesFloat16
    let usesQuaternionSmallestThree = packed.usesQuaternionSmallestThree
    let fractionalBits = packed.fractionalBits

    let pointCount = Int(numPoints)
    let shDimInt = Int(shDim)

    var result = GaussianCloud()
    result.numPoints = numPoints
    result.shDegree = packed.shDegree
    result.antialiased = packed.antialiased

    // Allocate arrays efficiently without initialization overhead
    result.positions = Array(unsafeUninitializedCapacity: pointCount * 3) { _, count in count = pointCount * 3 }
    result.scales = Array(unsafeUninitializedCapacity: pointCount * 3) { _, count in count = pointCount * 3 }
    result.rotations = Array(unsafeUninitializedCapacity: pointCount * 4) { _, count in count = pointCount * 4 }
    result.alphas = Array(unsafeUninitializedCapacity: pointCount) { _, count in count = pointCount }
    result.colors = Array(unsafeUninitializedCapacity: pointCount * 3) { _, count in count = pointCount * 3 }
    
    let shCount = pointCount * shDimInt * 3
    result.sh = Array(unsafeUninitializedCapacity: shCount) { _, count in count = shCount }

    packed.positions.withUnsafeBufferPointer { pPos in
    packed.scales.withUnsafeBufferPointer { pScale in
    packed.rotations.withUnsafeBufferPointer { pRot in
    packed.alphas.withUnsafeBufferPointer { pAlpha in
    packed.colors.withUnsafeBufferPointer { pColor in
    packed.sh.withUnsafeBufferPointer { pSH in
        result.positions.withUnsafeMutableBufferPointer { outPos in
        result.scales.withUnsafeMutableBufferPointer { outScale in
        result.rotations.withUnsafeMutableBufferPointer { outRot in
        result.alphas.withUnsafeMutableBufferPointer { outAlpha in
        result.colors.withUnsafeMutableBufferPointer { outColor in
        result.sh.withUnsafeMutableBufferPointer { outSH in

            DispatchQueue.concurrentPerform(iterations: pointCount) { i in
                // Decode position
                if usesFloat16 {
                    let h1 = UInt16(pPos[i * 6 + 0]) | (UInt16(pPos[i * 6 + 1]) << 8)
                    let h2 = UInt16(pPos[i * 6 + 2]) | (UInt16(pPos[i * 6 + 3]) << 8)
                    let h3 = UInt16(pPos[i * 6 + 4]) | (UInt16(pPos[i * 6 + 5]) << 8)
                    outPos[i * 3 + 0] = halfToFloat(h1)
                    outPos[i * 3 + 1] = halfToFloat(h2)
                    outPos[i * 3 + 2] = halfToFloat(h3)
                } else {
                    let scale = 1.0 / Float(1 << fractionalBits)
                    for c in 0..<3 {
                        var fixed32 = Int32(pPos[i * 9 + c * 3 + 0])
                        fixed32 |= Int32(pPos[i * 9 + c * 3 + 1]) << 8
                        fixed32 |= Int32(pPos[i * 9 + c * 3 + 2]) << 16
                        if (fixed32 & 0x800000) != 0 { fixed32 |= Int32(bitPattern: 0xff000000) }
                        outPos[i * 3 + c] = Float(fixed32) * scale
                    }
                }

                // Decode scale
                outScale[i * 3 + 0] = Float(pScale[i * 3 + 0]) / 16.0 - 10.0
                outScale[i * 3 + 1] = Float(pScale[i * 3 + 1]) / 16.0 - 10.0
                outScale[i * 3 + 2] = Float(pScale[i * 3 + 2]) / 16.0 - 10.0

                // Decode rotation
                if usesQuaternionSmallestThree {
                    let r0 = pRot[i * 4 + 0]
                    let r1 = pRot[i * 4 + 1]
                    let r2 = pRot[i * 4 + 2]
                    let r3 = pRot[i * 4 + 3]
                    
                    var comp = UInt32(r0)
                    comp |= UInt32(r1) << 8
                    comp |= UInt32(r2) << 16
                    comp |= UInt32(r3) << 24

                    let cMask: UInt32 = (1 << 9) - 1
                    let iLargest = Int(comp >> 30)
                    var sumSquares: Float = 0
                    
                    for j in stride(from: 3, through: 0, by: -1) {
                        if j == iLargest { continue }
                        let mag = comp & cMask
                        let negbit = (comp >> 9) & 0x1
                        comp = comp >> 10
                        
                        var val = 0.7071067811865476 * Float(mag) / Float(cMask)
                        if negbit == 1 { val = -val }
                        outRot[i * 4 + j] = val
                        sumSquares += val * val
                    }
                    outRot[i * 4 + iLargest] = sqrt(1.0 - sumSquares)
                } else {
                    let px = Float(pRot[i * 3 + 0])
                    let py = Float(pRot[i * 3 + 1])
                    let pz = Float(pRot[i * 3 + 2])
                    let xyz = SIMD3<Float>(px, py, pz) * (1.0 / 127.5) + SIMD3<Float>(-1, -1, -1)
                    let w = sqrt(max(0.0, 1.0 - simd_dot(xyz, xyz)))
                    outRot[i * 4 + 0] = xyz.x
                    outRot[i * 4 + 1] = xyz.y
                    outRot[i * 4 + 2] = xyz.z
                    outRot[i * 4 + 3] = w
                }

                // Decode alpha
                let alphaVal = Float(pAlpha[i]) / 255.0
                outAlpha[i] = log(alphaVal / (1.0 - alphaVal)) // Inline invSigmoid

                // Decode color
                outColor[i * 3 + 0] = ((Float(pColor[i * 3 + 0]) / 255.0) - 0.5) * (1.0 / 0.15) // 0.15 is colorScale
                outColor[i * 3 + 1] = ((Float(pColor[i * 3 + 1]) / 255.0) - 0.5) * (1.0 / 0.15)
                outColor[i * 3 + 2] = ((Float(pColor[i * 3 + 2]) / 255.0) - 0.5) * (1.0 / 0.15)

                // Decode spherical harmonics
                for j in 0..<(shDimInt * 3) {
                    let shVal = pSH[i * shDimInt * 3 + j]
                    outSH[i * shDimInt * 3 + j] = (Float(shVal) - 128.0) * (1.0 / 128.0) // Inline unquantizeSH
                }
            }
        }}}}}}
    }}}}}}

    // Apply coordinate conversion
    result.convertCoordinates(from: .rub, to: options.to)

    return result
}
