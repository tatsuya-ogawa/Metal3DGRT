//
//  GaussianBLASBufferBuilderTests.swift
//  Metal3DGRTUITests
//
//  Tests for GaussianBLASBufferBuilder — requires a real GPU device.
//

import XCTest
import Metal
import simd
@testable import Metal3DGRT

final class GaussianBLASBufferBuilderTests: XCTestCase {

    private var device: MTLDevice!
    private var commandQueue: MTLCommandQueue!
    private var builder: GaussianBLASBufferBuilder!

    // MARK: - Setup

    override func setUpWithError() throws {
        device = MTLCreateSystemDefaultDevice()
        try XCTSkipIf(device == nil, "Metal device unavailable")
        commandQueue = device.makeCommandQueue()!

        let library = try Self.findMetalLibrary(device: device)
        builder = try GaussianBLASBufferBuilder(device: device, library: library)
    }

    /// Locate the compiled Metal library.
    /// `device.makeDefaultLibrary()` looks in the *current* bundle, which is the
    /// test bundle — Metal shaders are compiled into the host app instead.
    /// Fall back to searching the host app bundle next to the test bundle.
    private static func findMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        let testBundle = Bundle(for: GaussianBLASBufferBuilderTests.self)

        // 1. Try the test bundle directly (works when .metal files are added to the test target)
        if let lib = try? device.makeDefaultLibrary(bundle: testBundle) {
            return lib
        }

        // 2. Try the main bundle (might be different in some test runners)
        if let lib = device.makeDefaultLibrary() {
            return lib
        }

        // 3. Search for the metallib inside the host application bundle
        // We might need to go several levels up if we are inside a Runner app's PlugIns folder
        var searchDir = testBundle.bundleURL
        for _ in 0..<5 {
            searchDir = searchDir.deletingLastPathComponent()

            let candidates = [
                // macOS: AppName.app/Contents/Resources/default.metallib
                searchDir.appendingPathComponent("Metal3DGRT.app/Contents/Resources/default.metallib"),
                // iOS / Catalyst: AppName.app/default.metallib
                searchDir.appendingPathComponent("Metal3DGRT.app/default.metallib"),
                // Alternative names
                searchDir.appendingPathComponent("Metal3DGRTiOS.app/default.metallib"),
            ]

            for url in candidates {
                if FileManager.default.fileExists(atPath: url.path) {
                    if let lib = try? device.makeLibrary(URL: url) {
                        return lib
                    }
                }
            }

            // Also look for ANY .app sibling
            if let items = try? FileManager.default.contentsOfDirectory(at: searchDir, includingPropertiesForKeys: nil) {
                for item in items where item.pathExtension == "app" {
                    let subs = ["Contents/Resources/default.metallib", "default.metallib"]
                    for sub in subs {
                        let url = item.appendingPathComponent(sub)
                        if let lib = try? device.makeLibrary(URL: url) {
                            return lib
                        }
                    }
                }
            }
        }

        throw GaussianBLASBufferBuilder.Error.defaultLibraryUnavailable
    }

    // MARK: - Helpers

    /// Identity-rotation gaussian at the origin with uniform scale.
    private func makeIdentityGaussian(
        position: SIMD3<Float> = .zero,
        scale: SIMD3<Float> = SIMD3<Float>(repeating: 1.0),
        density: Float = 1.0
    ) -> GaussianSplat {
        var sh = [Float](repeating: 0, count: 48)
        let c0: Float = 0.28209479177387814
        sh[0] = (1.0 - 0.5) / c0
        sh[1] = (1.0 - 0.5) / c0
        sh[2] = (1.0 - 0.5) / c0
        GaussianSplat(
            position: position,
            rotationWXYZ: SIMD4<Float>(1, 0, 0, 0), // identity quaternion
            scale: scale,
            density: density,
            sh: sh
        )
    }

    private func buildGeometry(
        gaussians: [GaussianSplat],
        primitive: GaussianEnclosingPrimitive,
        kernel: GaussianKernelConfiguration = GaussianKernelConfiguration(),
        storageMode: MTLResourceOptions = .storageModeShared
    ) throws -> GaussianBLASTriangleGeometry {
        try builder.makeTriangleGeometry(
            gaussians: gaussians,
            primitive: primitive,
            kernel: kernel,
            commandQueue: commandQueue,
            inputStorageMode: storageMode,
            outputStorageMode: storageMode,
            instanceStorageMode: storageMode
        )
    }

    private func readCanonicalVertices(from geometry: GaussianBLASPrimitiveGeometry) -> [SIMD3<Float>] {
        let count = geometry.vertexCount
        let pointer = geometry.vertexBuffer.contents()
            .bindMemory(to: SIMD3<Float>.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readCanonicalIndices(from geometry: GaussianBLASPrimitiveGeometry) -> [UInt32] {
        let count = geometry.triangleCount * 3
        let pointer = geometry.indexBuffer.contents()
            .bindMemory(to: UInt32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readInstanceDescriptors(from geometry: GaussianBLASInstanceGeometry) -> [MTLAccelerationStructureUserIDInstanceDescriptor] {
        let count = geometry.gaussianCount
        let pointer = geometry.instanceDescriptorBuffer.contents()
            .bindMemory(to: MTLAccelerationStructureUserIDInstanceDescriptor.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readInstanceTransforms(from geometry: GaussianBLASInstanceGeometry) -> [matrix_float4x4] {
        let count = geometry.gaussianCount
        let pointer = geometry.transformBuffer.contents()
            .bindMemory(to: matrix_float4x4.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func readInstancedSurfelNormalDensity(from geometry: GaussianBLASInstanceGeometry) -> [SIMD4<Float>]? {
        guard let buffer = geometry.surfelNormalDensityBuffer else { return nil }
        let count = geometry.gaussianCount
        let pointer = buffer.contents()
            .bindMemory(to: SIMD4<Float>.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private func transformedVertices(from geometry: GaussianBLASTriangleGeometry, instanceIndex: Int = 0) -> [SIMD3<Float>] {
        let canonicalVertices = readCanonicalVertices(from: geometry.primitiveGeometry)
        let transform = readInstanceTransforms(from: geometry.instances)[instanceIndex]
        return canonicalVertices.map { vertex in
            let world = transform * SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1.0)
            return SIMD3<Float>(world.x, world.y, world.z)
        }
    }

    // MARK: - Empty Input

    func testEmptyInputThrows() {
        XCTAssertThrowsError(try buildGeometry(gaussians: [], primitive: .octahedron))
    }

    // MARK: - Buffer Counts (All Primitives)

    func testOctahedronBufferCounts() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .octahedron)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 6)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 8)
        XCTAssertEqual(readCanonicalVertices(from: geo.primitiveGeometry).count, 6)
        XCTAssertEqual(readCanonicalIndices(from: geo.primitiveGeometry).count, 24) // 8 * 3
        XCTAssertNil(geo.instances.surfelNormalDensityBuffer)
    }

    func testIcosahedronBufferCounts() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .icosahedron)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 12)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 20)
        XCTAssertEqual(readCanonicalVertices(from: geo.primitiveGeometry).count, 12)
        XCTAssertEqual(readCanonicalIndices(from: geo.primitiveGeometry).count, 60) // 20 * 3
    }

    func testTriHexahedronBufferCounts() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .triHexahedron)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 6)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 6)
        XCTAssertEqual(readCanonicalIndices(from: geo.primitiveGeometry).count, 18) // 6 * 3
    }

    func testTriSurfelBufferCounts() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .triSurfel)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 4)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 2)
        XCTAssertEqual(readCanonicalIndices(from: geo.primitiveGeometry).count, 6) // 2 * 3
        XCTAssertNotNil(geo.instances.surfelNormalDensityBuffer)
    }

    func testMinAxisTriSurfelBufferCounts() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .minAxisTriSurfel)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 4)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 2)
        XCTAssertNotNil(geo.instances.surfelNormalDensityBuffer)
    }

    func testTetrahedronBufferCounts() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .tetrahedron)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 4)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 4)
        XCTAssertEqual(readCanonicalIndices(from: geo.primitiveGeometry).count, 12) // 4 * 3
    }

    func testDiamondBufferCounts() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .diamond)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 5)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 6)
        XCTAssertEqual(readCanonicalIndices(from: geo.primitiveGeometry).count, 18) // 6 * 3
    }

    // MARK: - Vertex Symmetry (Octahedron at origin)

    func testOctahedronVerticesCenteredAtOrigin() throws {
        let geo = try buildGeometry(
            gaussians: [makeIdentityGaussian(position: .zero, scale: SIMD3<Float>(repeating: 1.0))],
            primitive: .octahedron
        )
        let vertices = transformedVertices(from: geo)
        // Centroid of an octahedron at the origin should be approximately zero.
        let centroid = vertices.reduce(SIMD3<Float>.zero, +) / Float(vertices.count)
        XCTAssertEqual(centroid.x, 0, accuracy: 1e-3)
        XCTAssertEqual(centroid.y, 0, accuracy: 1e-3)
        XCTAssertEqual(centroid.z, 0, accuracy: 1e-3)
    }

    // MARK: - Position Offset

    func testOctahedronVerticesOffsetByPosition() throws {
        let offset = SIMD3<Float>(5, -3, 7)
        let geo = try buildGeometry(
            gaussians: [makeIdentityGaussian(position: offset)],
            primitive: .octahedron
        )
        let vertices = transformedVertices(from: geo)
        let centroid = vertices.reduce(SIMD3<Float>.zero, +) / Float(vertices.count)
        XCTAssertEqual(centroid.x, offset.x, accuracy: 1e-3)
        XCTAssertEqual(centroid.y, offset.y, accuracy: 1e-3)
        XCTAssertEqual(centroid.z, offset.z, accuracy: 1e-3)
    }

    // MARK: - Index Validity

    func testAllIndicesWithinVertexRange() throws {
        let primitives: [GaussianEnclosingPrimitive] = [
            .icosahedron, .octahedron, .triHexahedron,
            .triSurfel, .minAxisTriSurfel, .tetrahedron, .diamond
        ]
        for primitive in primitives {
            let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: primitive)
            let indices = readCanonicalIndices(from: geo.primitiveGeometry)
            let maxIndex = indices.max() ?? 0
            XCTAssertLessThan(
                Int(maxIndex), geo.primitiveGeometry.vertexCount,
                "Index out of range for \(primitive)"
            )
        }
    }

    // MARK: - Non-degenerate Triangles

    func testTrianglesAreNonDegenerate() throws {
        let primitives: [GaussianEnclosingPrimitive] = [
            .icosahedron, .octahedron, .triHexahedron,
            .triSurfel, .minAxisTriSurfel, .tetrahedron, .diamond
        ]
        for primitive in primitives {
            let geo = try buildGeometry(
                gaussians: [makeIdentityGaussian(scale: SIMD3<Float>(repeating: 1.0))],
                primitive: primitive
            )
            let vertices = transformedVertices(from: geo)
            let indices = readCanonicalIndices(from: geo.primitiveGeometry)

            for tri in stride(from: 0, to: indices.count, by: 3) {
                let v0 = vertices[Int(indices[tri])]
                let v1 = vertices[Int(indices[tri + 1])]
                let v2 = vertices[Int(indices[tri + 2])]
                let edge1 = v1 - v0
                let edge2 = v2 - v0
                let crossProduct = simd_cross(edge1, edge2)
                let area = simd_length(crossProduct) * 0.5
                XCTAssertGreaterThan(
                    area, 1e-6,
                    "Degenerate triangle at index \(tri / 3) for \(primitive)"
                )
            }
        }
    }

    // MARK: - Scale Affects Vertex Extent

    func testLargerScaleProducesLargerExtent() throws {
        let smallGaussian = makeIdentityGaussian(scale: SIMD3<Float>(repeating: 0.5))
        let largeGaussian = makeIdentityGaussian(scale: SIMD3<Float>(repeating: 2.0))

        let geoSmall = try buildGeometry(gaussians: [smallGaussian], primitive: .octahedron)
        let geoLarge = try buildGeometry(gaussians: [largeGaussian], primitive: .octahedron)

        let extentSmall = maxExtent(transformedVertices(from: geoSmall))
        let extentLarge = maxExtent(transformedVertices(from: geoLarge))
        XCTAssertGreaterThan(extentLarge, extentSmall)
    }

    private func maxExtent(_ vertices: [SIMD3<Float>]) -> Float {
        vertices.map { simd_length($0) }.max() ?? 0
    }

    // MARK: - Multiple Gaussians

    func testMultipleGaussiansProduceCorrectInstanceCounts() throws {
        let gaussians = (0..<10).map { i in
            makeIdentityGaussian(position: SIMD3<Float>(Float(i), 0, 0))
        }
        let geo = try buildGeometry(gaussians: gaussians, primitive: .octahedron)
        XCTAssertEqual(geo.gaussianCount, 10)
        XCTAssertEqual(geo.primitiveGeometry.vertexCount, 6)
        XCTAssertEqual(geo.primitiveGeometry.triangleCount, 8)
        XCTAssertEqual(readInstanceDescriptors(from: geo.instances).count, 10)
    }

    func testMultipleGaussiansIndicesAreValid() throws {
        let gaussians = (0..<5).map { i in
            makeIdentityGaussian(position: SIMD3<Float>(Float(i) * 10, 0, 0))
        }
        let geo = try buildGeometry(gaussians: gaussians, primitive: .icosahedron)
        let indices = readCanonicalIndices(from: geo.primitiveGeometry)
        let vertices = readCanonicalVertices(from: geo.primitiveGeometry)
        for idx in indices {
            XCTAssertLessThan(Int(idx), vertices.count)
        }
    }

    func testInstancedGeometryUsesSingleCanonicalPrimitive() throws {
        let gaussians = (0..<10).map { i in
            makeIdentityGaussian(position: SIMD3<Float>(Float(i), 0, 0))
        }

        let geometry = try buildGeometry(gaussians: gaussians, primitive: .octahedron)
        XCTAssertEqual(geometry.primitiveGeometry.vertexCount, 6)
        XCTAssertEqual(geometry.primitiveGeometry.triangleCount, 8)
        XCTAssertEqual(geometry.instances.gaussianCount, 10)
        XCTAssertEqual(readCanonicalVertices(from: geometry.primitiveGeometry).count, 6)
        XCTAssertEqual(readCanonicalIndices(from: geometry.primitiveGeometry).count, 24)
    }

    func testInstancedGeometryBuildsPerGaussianInstanceDescriptors() throws {
        let gaussians = [
            makeIdentityGaussian(position: SIMD3<Float>(1, 2, 3)),
            makeIdentityGaussian(position: SIMD3<Float>(4, 5, 6)),
            makeIdentityGaussian(position: SIMD3<Float>(7, 8, 9))
        ]

        let geometry = try buildGeometry(gaussians: gaussians, primitive: .octahedron)
        let descriptors = readInstanceDescriptors(from: geometry.instances)
        XCTAssertEqual(descriptors.count, gaussians.count)

        for (index, descriptor) in descriptors.enumerated() {
            XCTAssertEqual(descriptor.accelerationStructureIndex, 0)
            XCTAssertEqual(descriptor.userID, UInt32(index))
            XCTAssertEqual(descriptor.transformationMatrix.columns.3.x, gaussians[index].position.x, accuracy: 1e-5)
            XCTAssertEqual(descriptor.transformationMatrix.columns.3.y, gaussians[index].position.y, accuracy: 1e-5)
            XCTAssertEqual(descriptor.transformationMatrix.columns.3.z, gaussians[index].position.z, accuracy: 1e-5)
        }
    }

    func testInstancedTriSurfelEmitsPerGaussianMetadata() throws {
        let gaussians = [
            makeIdentityGaussian(density: 0.25),
            makeIdentityGaussian(density: 0.75)
        ]

        let geometry = try buildGeometry(gaussians: gaussians, primitive: .triSurfel)
        guard let metadata = readInstancedSurfelNormalDensity(from: geometry.instances) else {
            XCTFail("surfelNormalDensityBuffer should not be nil for instanced triSurfel")
            return
        }

        XCTAssertEqual(metadata.count, gaussians.count)
        XCTAssertEqual(metadata[0].w, gaussians[0].density, accuracy: 1e-5)
        XCTAssertEqual(metadata[1].w, gaussians[1].density, accuracy: 1e-5)
    }

    // MARK: - AccelerationStructure Descriptor

    func testGeometryDescriptorProperties() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .octahedron)
        let descriptor = geo.primitiveGeometry.makeGeometryDescriptor(opaque: false)
        XCTAssertEqual(descriptor.triangleCount, 8)
        XCTAssertFalse(descriptor.opaque)

        let opaqueDescriptor = geo.primitiveGeometry.makeGeometryDescriptor(opaque: true)
        XCTAssertTrue(opaqueDescriptor.opaque)
    }

    func testPrimitiveAccelerationStructureDescriptor() throws {
        let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: .octahedron)
        let descriptor = geo.primitiveGeometry.makePrimitiveAccelerationStructureDescriptor()
        XCTAssertEqual(descriptor.geometryDescriptors?.count, 1)
    }

    // MARK: - Surfel Metadata

    func testTriSurfelEmitsSurfelNormalDensity() throws {
        let density: Float = 0.75
        let geo = try buildGeometry(
            gaussians: [makeIdentityGaussian(density: density)],
            primitive: .triSurfel
        )
        guard let surfelData = readInstancedSurfelNormalDensity(from: geo.instances) else {
            XCTFail("surfelNormalDensityBuffer should not be nil for triSurfel")
            return
        }
        XCTAssertEqual(surfelData.count, 1)
        // Density stored in the w component
        XCTAssertEqual(surfelData[0].w, density, accuracy: 1e-5)
        // Normal should be unit-length
        let normal = SIMD3<Float>(surfelData[0].x, surfelData[0].y, surfelData[0].z)
        let normalLength = simd_length(normal)
        XCTAssertEqual(normalLength, 1.0, accuracy: 1e-3, "Surfel normal should be unit-length")
    }

    func testNonSurfelPrimitivesHaveNoSurfelBuffer() throws {
        let nonSurfel: [GaussianEnclosingPrimitive] = [
            .icosahedron, .octahedron, .triHexahedron, .tetrahedron, .diamond
        ]
        for primitive in nonSurfel {
            let geo = try buildGeometry(gaussians: [makeIdentityGaussian()], primitive: primitive)
            XCTAssertNil(geo.instances.surfelNormalDensityBuffer, "\(primitive) should not emit instance surfel metadata")
            XCTAssertNil(geo.primitiveGeometry.canonicalSurfelNormalDensityBuffer, "\(primitive) should not emit canonical surfel metadata")
        }
    }

    // MARK: - Anisotropic Scale

    func testAnisotropicScaleProducesElongatedShape() throws {
        let gaussian = makeIdentityGaussian(
            scale: SIMD3<Float>(0.1, 0.1, 5.0)
        )
        let geo = try buildGeometry(gaussians: [gaussian], primitive: .octahedron)
        let vertices = transformedVertices(from: geo)

        let xExtent = vertices.map { abs($0.x) }.max() ?? 0
        let zExtent = vertices.map { abs($0.z) }.max() ?? 0
        // z scale is 50× the x scale, so z extent should be much larger
        XCTAssertGreaterThan(zExtent, xExtent * 5)
    }

    // MARK: - Rotation

    func testRotationAffectsVertices() throws {
        let identityGaussian = makeIdentityGaussian(scale: SIMD3<Float>(1, 1, 3))
        // 90° rotation around Y axis: quaternion = (cos(45°), 0, sin(45°), 0)
        let angle = Float.pi / 2
        let rotatedGaussian = GaussianSplat(
            position: .zero,
            rotationWXYZ: SIMD4<Float>(cos(angle / 2), 0, sin(angle / 2), 0),
            scale: SIMD3<Float>(1, 1, 3),
            density: 1.0,
            sh: identityGaussian.sh
        )

        let geoIdentity = try buildGeometry(gaussians: [identityGaussian], primitive: .octahedron)
        let geoRotated = try buildGeometry(gaussians: [rotatedGaussian], primitive: .octahedron)

        let vertsIdentity = transformedVertices(from: geoIdentity)
        let vertsRotated = transformedVertices(from: geoRotated)

        // The identity gaussian has z-extent > x-extent.
        // After 90° Y rotation, the x-extent should become larger.
        let identityZExtent = vertsIdentity.map { abs($0.z) }.max() ?? 0
        let rotatedXExtent = vertsRotated.map { abs($0.x) }.max() ?? 0
        XCTAssertEqual(identityZExtent, rotatedXExtent, accuracy: 1e-2,
                       "90° Y rotation should swap Z and X extents")
    }

    // MARK: - Kernel Configuration

    func testDifferentKernelDegreeChangesExtent() throws {
        let gaussian = [makeIdentityGaussian()]
        let kernel1 = GaussianKernelConfiguration(minResponse: 0.0113, degree: 2.0)
        let kernel2 = GaussianKernelConfiguration(minResponse: 0.0113, degree: 6.0)

        let geo1 = try buildGeometry(gaussians: gaussian, primitive: .octahedron, kernel: kernel1)
        let geo2 = try buildGeometry(gaussians: gaussian, primitive: .octahedron, kernel: kernel2)

        let extent1 = maxExtent(transformedVertices(from: geo1))
        let extent2 = maxExtent(transformedVertices(from: geo2))

        // Different degrees should produce different extents
        XCTAssertNotEqual(extent1, extent2, accuracy: 1e-4,
                          "Different kernel degrees should produce different vertex extents")
    }

    // MARK: - MinAxisTriSurfel Axis Selection

    func testMinAxisTriSurfelSelectsMinAxis() throws {
        // Scale with smallest component on Y axis
        let gaussian = makeIdentityGaussian(scale: SIMD3<Float>(2.0, 0.1, 2.0))
        let geo = try buildGeometry(gaussians: [gaussian], primitive: .minAxisTriSurfel)
        let vertices = transformedVertices(from: geo)

        // For minAxisTriSurfel with Y as the smallest axis, the surfel
        // should lie in the XZ plane; Y-extent of vertices should be small.
        let yExtent = vertices.map { abs($0.y) }.max() ?? 0
        let xExtent = vertices.map { abs($0.x) }.max() ?? 0
        let zExtent = vertices.map { abs($0.z) }.max() ?? 0

        XCTAssertLessThan(yExtent, xExtent,
                          "minAxisTriSurfel should have minimal extent along the smallest scale axis")
        XCTAssertLessThan(yExtent, zExtent,
                          "minAxisTriSurfel should have minimal extent along the smallest scale axis")
    }
}
