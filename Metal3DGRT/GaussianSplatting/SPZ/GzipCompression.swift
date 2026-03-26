// spz-swift
// Swift port of https://github.com/nianticlabs/spz
// MIT License - Copyright (c) 2024 Niantic Labs

import Foundation
import Compression

/// Errors that can occur during SPZ operations.
public enum SPZError: Error, Sendable {
    case decompressionFailed
    case compressionFailed
    case invalidHeader
    case unsupportedVersion(Int)
    case tooManyPoints(Int)
    case unsupportedSHDegree(Int)
    case readError
    case writeError
    case invalidData
}

// MARK: - Gzip Compression using zlib

/// Decompress gzip-compressed data.
internal func decompressGzip(_ compressed: Data) throws -> Data {
    guard !compressed.isEmpty else {
        throw SPZError.decompressionFailed
    }

    // Check for gzip magic number (0x1f 0x8b)
    let isGzip = compressed.count >= 2 && compressed[compressed.startIndex] == 0x1f && compressed[compressed.startIndex + 1] == 0x8b

    if isGzip {
        return try decompressGzipManual(compressed)
    } else {
        return try decompressDeflate(compressed)
    }
}

/// Manual gzip decompression - skip gzip header and decompress deflate payload.
private func decompressGzipManual(_ compressed: Data) throws -> Data {
    // Gzip header: magic (2) + method (1) + flags (1) + mtime (4) + xfl (1) + os (1) = 10 bytes minimum
    guard compressed.count >= 10 else {
        throw SPZError.decompressionFailed
    }

    let flags = compressed[compressed.startIndex + 3]
    var offset = 10

    // FEXTRA
    if flags & 0x04 != 0 {
        guard compressed.count >= offset + 2 else { throw SPZError.decompressionFailed }
        let xlen = Int(compressed[compressed.startIndex + offset]) | (Int(compressed[compressed.startIndex + offset + 1]) << 8)
        offset += 2 + xlen
    }

    // FNAME
    if flags & 0x08 != 0 {
        while offset < compressed.count && compressed[compressed.startIndex + offset] != 0 {
            offset += 1
        }
        offset += 1
    }

    // FCOMMENT
    if flags & 0x10 != 0 {
        while offset < compressed.count && compressed[compressed.startIndex + offset] != 0 {
            offset += 1
        }
        offset += 1
    }

    // FHCRC
    if flags & 0x02 != 0 {
        offset += 2
    }

    guard offset < compressed.count - 8 else {
        throw SPZError.decompressionFailed
    }

    // The deflate data ends 8 bytes before the end (4 bytes CRC32 + 4 bytes original size)
    let startIndex = compressed.startIndex + offset
    let endIndex = compressed.endIndex - 8
    let deflateData = compressed[startIndex..<endIndex]

    return try decompressDeflate(Data(deflateData))
}

/// Decompress raw deflate data.
private func decompressDeflate(_ compressed: Data) throws -> Data {
    // Estimate decompressed size (SPZ typically compresses ~10x)
    let estimatedSize = max(compressed.count * 20, 65536)

    let sourceBytes = [UInt8](compressed)
    var destBytes = [UInt8](repeating: 0, count: estimatedSize)

    let decodedSize = compression_decode_buffer(
        &destBytes,
        destBytes.count,
        sourceBytes,
        sourceBytes.count,
        nil,
        COMPRESSION_ZLIB
    )

    if decodedSize == 0 || decodedSize == estimatedSize {
        // May need more space, try again with larger buffer
        var largerBuffer = [UInt8](repeating: 0, count: estimatedSize * 10)

        let decodedSize2 = compression_decode_buffer(
            &largerBuffer,
            largerBuffer.count,
            sourceBytes,
            sourceBytes.count,
            nil,
            COMPRESSION_ZLIB
        )

        if decodedSize2 == 0 {
            throw SPZError.decompressionFailed
        }

        return Data(largerBuffer.prefix(decodedSize2))
    }

    return Data(destBytes.prefix(decodedSize))
}

/// Compress data using gzip.
internal func compressGzip(_ data: Data) throws -> Data {
    guard !data.isEmpty else {
        throw SPZError.compressionFailed
    }

    // Compress using zlib deflate
    let compressedDeflate = try compressDeflate(data)

    // Wrap in gzip container
    var result = Data()

    // Gzip header
    result.append(contentsOf: [
        0x1f, 0x8b,  // Magic number
        0x08,        // Compression method (deflate)
        0x00,        // Flags
        0x00, 0x00, 0x00, 0x00,  // Modification time
        0x00,        // Extra flags
        0xff         // OS (unknown)
    ] as [UInt8])

    // Compressed data
    result.append(compressedDeflate)

    // CRC32 of original data
    let crc = crc32(data)
    result.append(UInt8(crc & 0xff))
    result.append(UInt8((crc >> 8) & 0xff))
    result.append(UInt8((crc >> 16) & 0xff))
    result.append(UInt8((crc >> 24) & 0xff))

    // Original size (mod 2^32)
    let size = UInt32(data.count & 0xffffffff)
    result.append(UInt8(size & 0xff))
    result.append(UInt8((size >> 8) & 0xff))
    result.append(UInt8((size >> 16) & 0xff))
    result.append(UInt8((size >> 24) & 0xff))

    return result
}

/// Compress data using raw deflate.
private func compressDeflate(_ data: Data) throws -> Data {
    let bufferSize = max(data.count + 1024, 1024)

    let sourceBytes = [UInt8](data)
    var destBytes = [UInt8](repeating: 0, count: bufferSize)

    let encodedSize = compression_encode_buffer(
        &destBytes,
        destBytes.count,
        sourceBytes,
        sourceBytes.count,
        nil,
        COMPRESSION_ZLIB
    )

    if encodedSize == 0 {
        throw SPZError.compressionFailed
    }

    return Data(destBytes.prefix(encodedSize))
}

// MARK: - CRC32

/// Calculate CRC32 checksum.
private func crc32(_ data: Data) -> UInt32 {
    var crc: UInt32 = 0xffffffff

    // CRC32 lookup table (IEEE 802.3 polynomial)
    let table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 {
            if c & 1 != 0 {
                c = 0xedb88320 ^ (c >> 1)
            } else {
                c = c >> 1
            }
        }
        return c
    }

    for byte in data {
        let index = Int((crc ^ UInt32(byte)) & 0xff)
        crc = table[index] ^ (crc >> 8)
    }

    return crc ^ 0xffffffff
}
