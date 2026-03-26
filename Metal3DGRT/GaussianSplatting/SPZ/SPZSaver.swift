// spz-swift
// Swift port of https://github.com/nianticlabs/spz
// MIT License - Copyright (c) 2024 Niantic Labs

import Foundation

/// Options for packing SPZ data.
public struct PackOptions: Sendable {
    /// Source coordinate system of the input data.
    public var from: CoordinateSystem

    public init(from: CoordinateSystem = .unspecified) {
        self.from = from
    }
}

// MARK: - Saving Functions

/// Save a GaussianCloud to SPZ format.
/// - Parameters:
///   - cloud: The gaussian cloud to save.
///   - options: Packing options including source coordinate system.
/// - Returns: The compressed SPZ data.
public func saveSpz(_ cloud: GaussianCloud, options: PackOptions = PackOptions()) throws -> Data {
    let packed = packGaussians(cloud, options: options)
    let serialized = serializePackedGaussians(packed)
    return try compressGzip(serialized)
}

/// Save a GaussianCloud to an SPZ file.
/// - Parameters:
///   - cloud: The gaussian cloud to save.
///   - url: URL where to save the file.
///   - options: Packing options including source coordinate system.
public func saveSpz(_ cloud: GaussianCloud, to url: URL, options: PackOptions = PackOptions()) throws {
    let data = try saveSpz(cloud, options: options)
    try data.write(to: url)
}

// MARK: - Packing

/// Pack a GaussianCloud into compressed format.
public func packGaussians(_ cloud: GaussianCloud, options: PackOptions = PackOptions()) -> PackedGaussians {
    guard cloud.checkSizes() else {
        return PackedGaussians()
    }

    let numPoints = cloud.numPoints
    let shDim = dimForDegree(cloud.shDegree)
    let c = coordinateConverter(from: options.from, to: .rub)

    var packed = PackedGaussians()
    packed.numPoints = numPoints
    packed.shDegree = cloud.shDegree
    packed.fractionalBits = 12  // ~0.25mm resolution
    packed.antialiased = cloud.antialiased
    packed.usesQuaternionSmallestThree = true

    packed.positions = Array(repeating: 0, count: Int(numPoints) * 3 * 3)
    packed.scales = Array(repeating: 0, count: Int(numPoints) * 3)
    packed.rotations = Array(repeating: 0, count: Int(numPoints) * 4)
    packed.alphas = Array(repeating: 0, count: Int(numPoints))
    packed.colors = Array(repeating: 0, count: Int(numPoints) * 3)
    packed.sh = Array(repeating: 0, count: Int(numPoints) * Int(shDim) * 3)

    // Pack positions as 24-bit fixed point
    let scale = Float(1 << packed.fractionalBits)
    for i in 0..<(Int(numPoints) * 3) {
        let flipped = c.flipP[i % 3] * cloud.positions[i]
        let fixed32 = Int32(round(flipped * scale))
        packed.positions[i * 3 + 0] = UInt8(fixed32 & 0xff)
        packed.positions[i * 3 + 1] = UInt8((fixed32 >> 8) & 0xff)
        packed.positions[i * 3 + 2] = UInt8((fixed32 >> 16) & 0xff)
    }

    // Pack scales
    for i in 0..<(Int(numPoints) * 3) {
        packed.scales[i] = toUint8((cloud.scales[i] + 10.0) * 16.0)
    }

    // Pack rotations
    for i in 0..<Int(numPoints) {
        let rotation = [
            cloud.rotations[i * 4 + 0],
            cloud.rotations[i * 4 + 1],
            cloud.rotations[i * 4 + 2],
            cloud.rotations[i * 4 + 3]
        ]
        let packedRot = packQuaternionSmallestThree(rotation, converter: c)
        packed.rotations[i * 4 + 0] = packedRot[0]
        packed.rotations[i * 4 + 1] = packedRot[1]
        packed.rotations[i * 4 + 2] = packedRot[2]
        packed.rotations[i * 4 + 3] = packedRot[3]
    }

    // Pack alphas (apply sigmoid)
    for i in 0..<Int(numPoints) {
        packed.alphas[i] = toUint8(sigmoid(cloud.alphas[i]) * 255.0)
    }

    // Pack colors
    for i in 0..<(Int(numPoints) * 3) {
        packed.colors[i] = toUint8(cloud.colors[i] * (colorScale * 255.0) + (0.5 * 255.0))
    }

    // Pack spherical harmonics
    if cloud.shDegree > 0 {
        let sh1Bits: Int32 = 5
        let shRestBits: Int32 = 4
        let shPerPoint = Int(dimForDegree(cloud.shDegree)) * 3

        for i in stride(from: 0, to: Int(numPoints) * shPerPoint, by: shPerPoint) {
            var j = 0
            var k = 0
            // Degree 1 coefficients (9 values = 3 coeffs x 3 channels)
            while j < 9 {
                packed.sh[i + j + 0] = quantizeSH(c.flipSh[k] * cloud.sh[i + j + 0], bucketSize: 1 << (8 - sh1Bits))
                packed.sh[i + j + 1] = quantizeSH(c.flipSh[k] * cloud.sh[i + j + 1], bucketSize: 1 << (8 - sh1Bits))
                packed.sh[i + j + 2] = quantizeSH(c.flipSh[k] * cloud.sh[i + j + 2], bucketSize: 1 << (8 - sh1Bits))
                j += 3
                k += 1
            }
            // Higher degree coefficients
            while j < shPerPoint {
                packed.sh[i + j + 0] = quantizeSH(c.flipSh[k] * cloud.sh[i + j + 0], bucketSize: 1 << (8 - shRestBits))
                packed.sh[i + j + 1] = quantizeSH(c.flipSh[k] * cloud.sh[i + j + 1], bucketSize: 1 << (8 - shRestBits))
                packed.sh[i + j + 2] = quantizeSH(c.flipSh[k] * cloud.sh[i + j + 2], bucketSize: 1 << (8 - shRestBits))
                j += 3
                k += 1
            }
        }
    }

    return packed
}

// MARK: - Serialization

/// Serialize packed gaussians to binary data.
internal func serializePackedGaussians(_ packed: PackedGaussians) -> Data {
    var data = Data()

    // Write header
    var magic = PackedGaussiansHeader.magic
    var version = PackedGaussiansHeader.currentVersion
    var numPoints = UInt32(packed.numPoints)
    let shDegree = UInt8(packed.shDegree)
    let fractionalBits = UInt8(packed.fractionalBits)
    let flags: UInt8 = packed.antialiased ? PackedGaussiansHeader.flagAntialiased : 0
    let reserved: UInt8 = 0

    data.append(Data(bytes: &magic, count: 4))
    data.append(Data(bytes: &version, count: 4))
    data.append(Data(bytes: &numPoints, count: 4))
    data.append(shDegree)
    data.append(fractionalBits)
    data.append(flags)
    data.append(reserved)

    // Write data arrays in order
    data.append(contentsOf: packed.positions)
    data.append(contentsOf: packed.alphas)
    data.append(contentsOf: packed.colors)
    data.append(contentsOf: packed.scales)
    data.append(contentsOf: packed.rotations)
    data.append(contentsOf: packed.sh)

    return data
}
