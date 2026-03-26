//
//  GaussianBLASBuffers.swift
//  Metal3DGRT
//
//  Standalone compute-driven helpers for building triangle BLAS input buffers
//  from Gaussian splats. Primitive generation itself lives in
//  GaussianBLASGenerator.metal so this path stays independent from the mesh
//  renderer.
//

import Metal
import simd

private enum GaussianBLASStorageMode {
    static var sharedCPUAccessible: MTLResourceOptions {
        #if os(macOS)
        return .storageModeManaged
        #else
        return .storageModeShared
        #endif
    }
}

struct GaussianSplat {
    var position: SIMD3<Float>
    var rotationWXYZ: SIMD4<Float>
    var scale: SIMD3<Float>
    var density: Float
    var sh: [Float] // Max 48 coefficients
}

struct GaussianKernelOptions: OptionSet {
    let rawValue: UInt32

    static let adaptiveDensityClamping = GaussianKernelOptions(rawValue: 1 << 0)
}

struct GaussianKernelConfiguration {
    var minResponse: Float = 0.0113
    var degree: Float = 4.0
    var options: GaussianKernelOptions = []
}

enum GaussianEnclosingPrimitive: CaseIterable {
    case icosahedron
    case octahedron
    case triHexahedron
    case triSurfel
    case minAxisTriSurfel
    case tetrahedron
    case diamond

    var verticesPerGaussian: Int {
        switch self {
        case .icosahedron:
            return 12
        case .octahedron, .triHexahedron:
            return 6
        case .triSurfel, .minAxisTriSurfel:
            return 4
        case .tetrahedron:
            return 4
        case .diamond:
            return 5
        }
    }

    var trianglesPerGaussian: Int {
        switch self {
        case .icosahedron:
            return 20
        case .octahedron:
            return 8
        case .triHexahedron, .diamond:
            return 6
        case .triSurfel, .minAxisTriSurfel:
            return 2
        case .tetrahedron:
            return 4
        }
    }

    var shaderValue: UInt32 {
        switch self {
        case .icosahedron:
            return 0
        case .octahedron:
            return 1
        case .triHexahedron:
            return 2
        case .triSurfel:
            return 3
        case .minAxisTriSurfel:
            return 4
        case .tetrahedron:
            return 5
        case .diamond:
            return 6
        }
    }

    /// Maps the enclosing BLAS shape to the intersection evaluation primitive type
    /// expected by `evaluateGaussianIntersection`.
    var intersectionPrimitiveType: UInt32 {
        switch self {
        case .triSurfel:
            return 3  // surfel
        case .minAxisTriSurfel:
            return 4  // min-axis surfel
        default:
            return 0  // volumetric
        }
    }

    fileprivate var emitsSurfelMetadata: Bool {
        self == .triSurfel || self == .minAxisTriSurfel
    }

    var displayName: String {
        switch self {
        case .icosahedron:
            return "Icosahedron"
        case .octahedron:
            return "Octahedron"
        case .triHexahedron:
            return "TriHexahedron"
        case .triSurfel:
            return "TriSurfel"
        case .minAxisTriSurfel:
            return "MinAxisTriSurfel"
        case .tetrahedron:
            return "Tetrahedron"
        case .diamond:
            return "Diamond"
        }
    }
}

struct GaussianBLASPrimitiveGeometry {
    enum Storage {
        case triangles(vertexBuffer: MTLBuffer,
                       indexBuffer: MTLBuffer,
                       canonicalSurfelNormalDensityBuffer: MTLBuffer?)
        case boundingBoxes(buffer: MTLBuffer,
                           count: Int)
    }

    let primitive: GaussianEnclosingPrimitive
    let canonicalKernelScale: Float
    let storage: Storage

    var vertexCount: Int { primitive.verticesPerGaussian }
    var triangleCount: Int { primitive.trianglesPerGaussian }

    func makeGeometryDescriptor(opaque: Bool = false) -> MTLAccelerationStructureGeometryDescriptor {
        switch storage {
        case let .triangles(vertexBuffer, indexBuffer, _):
            let descriptor = MTLAccelerationStructureTriangleGeometryDescriptor()
            descriptor.vertexBuffer = vertexBuffer
            descriptor.vertexStride = MemoryLayout<SIMD3<Float>>.stride
            descriptor.indexBuffer = indexBuffer
            descriptor.indexType = .uint32
            descriptor.triangleCount = triangleCount
            descriptor.opaque = opaque
            return descriptor
        case let .boundingBoxes(buffer, count):
            let descriptor = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
            descriptor.boundingBoxBuffer = buffer
            descriptor.boundingBoxStride = MemoryLayout<MTLAxisAlignedBoundingBox>.stride
            descriptor.boundingBoxCount = count
            descriptor.opaque = opaque
            return descriptor
        }
    }

    func makePrimitiveAccelerationStructureDescriptor(opaque: Bool = false) -> MTLPrimitiveAccelerationStructureDescriptor {
        let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
        descriptor.geometryDescriptors = [makeGeometryDescriptor(opaque: opaque)]
        return descriptor
    }
}

struct GaussianBLASInstanceGeometry {
    let primitive: GaussianEnclosingPrimitive
    let gaussianCount: Int
    let gaussianBuffer: MTLBuffer
    let instanceDescriptorBuffer: MTLBuffer
    let transformBuffer: MTLBuffer
    let surfelNormalDensityBuffer: MTLBuffer?

    func makeInstanceAccelerationStructureDescriptor(
        primitiveAccelerationStructure: MTLAccelerationStructure
    ) -> MTLInstanceAccelerationStructureDescriptor {
        let descriptor = MTLInstanceAccelerationStructureDescriptor()
        descriptor.instanceCount = gaussianCount
        descriptor.instancedAccelerationStructures = [primitiveAccelerationStructure]
        descriptor.instanceDescriptorBuffer = instanceDescriptorBuffer
        descriptor.instanceDescriptorType = .userID
        return descriptor
    }
}

struct GaussianBLASTriangleGeometry {
    let primitiveGeometry: GaussianBLASPrimitiveGeometry
    let instances: GaussianBLASInstanceGeometry

    var primitive: GaussianEnclosingPrimitive { primitiveGeometry.primitive }
    var gaussianCount: Int { instances.gaussianCount }

    func makeInstanceAccelerationStructureDescriptor(
        primitiveAccelerationStructure: MTLAccelerationStructure
    ) -> MTLInstanceAccelerationStructureDescriptor {
        instances.makeInstanceAccelerationStructureDescriptor(
            primitiveAccelerationStructure: primitiveAccelerationStructure
        )
    }
}

final class GaussianBLASBufferBuilder {
    private struct GeneratedPrimitiveBuffers {
        let vertexBuffer: MTLBuffer
        let indexBuffer: MTLBuffer
        let surfelNormalDensityBuffer: MTLBuffer?
    }

    private struct GeneratedBoundingBoxBuffers {
        let boundingBoxBuffer: MTLBuffer
    }

    enum Error: Swift.Error {
        case defaultLibraryUnavailable
        case computeFunctionNotFound
        case pipelineCreationFailed
        case emptyInput
        case gaussianBufferAllocationFailed
        case uniformBufferAllocationFailed
        case vertexBufferAllocationFailed
        case indexBufferAllocationFailed
        case surfelMetadataBufferAllocationFailed
        case instanceDescriptorBufferAllocationFailed
        case instanceTransformBufferAllocationFailed
        case commandBufferCreationFailed
        case computeEncoderCreationFailed
        case gpuExecutionFailed(String?)
    }

    private let device: MTLDevice
    private let pipelineState: MTLComputePipelineState

    convenience init(device: MTLDevice) throws {
        guard let library = device.makeDefaultLibrary() else {
            throw Error.defaultLibraryUnavailable
        }
        try self.init(device: device, library: library)
    }

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device

        guard let function = library.makeFunction(name: "gaussianBLASBuildKernel") else {
            throw Error.computeFunctionNotFound
        }

        do {
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } catch {
            throw Error.pipelineCreationFailed
        }
    }

    private func buildPrimitiveBuffers(
        gaussians: [GaussianSplat],
        primitive: GaussianEnclosingPrimitive,
        kernel: GaussianKernelConfiguration = GaussianKernelConfiguration(),
        commandQueue: MTLCommandQueue,
        inputStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible,
        outputStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible
    ) throws -> GeneratedPrimitiveBuffers {
        guard !gaussians.isEmpty else {
            throw Error.emptyInput
        }

        let gaussianBuffer = try makeGaussianBuffer(from: gaussians, options: inputStorageMode)
        gaussianBuffer.label = "Gaussian BLAS Source Gaussians"

        let uniformBuffer = try makeUniformBuffer(
            primitive: primitive,
            gaussianCount: gaussians.count,
            kernel: kernel,
            options: inputStorageMode
        )
        uniformBuffer.label = "Gaussian BLAS Build Uniforms"

        let vertexBufferLength = gaussians.count
            * primitive.verticesPerGaussian
            * MemoryLayout<SIMD3<Float>>.stride
        guard let vertexBuffer = device.makeBuffer(length: vertexBufferLength, options: outputStorageMode) else {
            throw Error.vertexBufferAllocationFailed
        }
        vertexBuffer.label = "Gaussian BLAS Vertices"

        let indexBufferLength = gaussians.count
            * primitive.trianglesPerGaussian
            * 3
            * MemoryLayout<UInt32>.stride
        guard let indexBuffer = device.makeBuffer(length: indexBufferLength, options: outputStorageMode) else {
            throw Error.indexBufferAllocationFailed
        }
        indexBuffer.label = "Gaussian BLAS Indices"

        let surfelNormalDensityBuffer: MTLBuffer?
        if primitive.emitsSurfelMetadata {
            let length = gaussians.count * MemoryLayout<SIMD4<Float>>.stride
            guard let buffer = device.makeBuffer(length: length, options: outputStorageMode) else {
                throw Error.surfelMetadataBufferAllocationFailed
            }
            buffer.label = "Gaussian BLAS Surfel NormalDensity"
            surfelNormalDensityBuffer = buffer
        } else {
            surfelNormalDensityBuffer = nil
        }

        // Metal validation requires a valid buffer at every declared binding index.
        // The shader declares outSurfelNormalDensity at [[buffer(4)]] unconditionally,
        // so bind a small placeholder when no real surfel buffer is needed.
        // The shader guards writes behind uniforms.emitsSurfelMetadata.
        let surfelBindBuffer: MTLBuffer
        if let buf = surfelNormalDensityBuffer {
            surfelBindBuffer = buf
        } else {
            guard let placeholder = device.makeBuffer(length: MemoryLayout<SIMD4<Float>>.stride, options: outputStorageMode) else {
                throw Error.surfelMetadataBufferAllocationFailed
            }
            placeholder.label = "Gaussian BLAS Surfel NormalDensity (placeholder)"
            surfelBindBuffer = placeholder
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw Error.commandBufferCreationFailed
        }
        commandBuffer.label = "Gaussian BLAS Build CommandBuffer"

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Error.computeEncoderCreationFailed
        }
        computeEncoder.label = "Gaussian BLAS Build Encoder"
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setBuffer(gaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(uniformBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(indexBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(surfelBindBuffer, offset: 0, index: 4)

        let threadsPerThreadgroupWidth = max(1, min(pipelineState.threadExecutionWidth,
                                                    pipelineState.maxTotalThreadsPerThreadgroup))
        let threadsPerThreadgroup = MTLSize(width: threadsPerThreadgroupWidth, height: 1, depth: 1)
        let threadsPerGrid = MTLSize(width: gaussians.count, height: 1, depth: 1)
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            throw Error.gpuExecutionFailed(commandBuffer.error?.localizedDescription)
        }

        return GeneratedPrimitiveBuffers(
            vertexBuffer: vertexBuffer,
            indexBuffer: indexBuffer,
            surfelNormalDensityBuffer: surfelNormalDensityBuffer
        )
    }

    private func buildBoundingBoxBuffers(
        primitive: GaussianEnclosingPrimitive,
        kernel: GaussianKernelConfiguration,
        outputStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible
    ) throws -> GeneratedBoundingBoxBuffers {
        let canonicalGaussian = Self.canonicalGaussian(for: primitive)
        let canonicalKernelScale = Self.kernelScale(density: canonicalGaussian.density, kernel: kernel)
        let extent = SIMD3<Float>(repeating: canonicalKernelScale)
        var boundingBox = MTLAxisAlignedBoundingBox()
        boundingBox.min.x = -extent.x
        boundingBox.min.y = -extent.y
        boundingBox.min.z = -extent.z
        boundingBox.max.x = extent.x
        boundingBox.max.y = extent.y
        boundingBox.max.z = extent.z
        guard let boundingBoxBuffer = makeBuffer(from: [boundingBox], options: outputStorageMode) else {
            throw Error.vertexBufferAllocationFailed
        }
        boundingBoxBuffer.label = "Gaussian BLAS Bounding Boxes"
        return GeneratedBoundingBoxBuffers(boundingBoxBuffer: boundingBoxBuffer)
    }

    func makeTriangleGeometry(
        gaussians: [GaussianSplat],
        primitive: GaussianEnclosingPrimitive,
        kernel: GaussianKernelConfiguration = GaussianKernelConfiguration(),
        commandQueue: MTLCommandQueue,
        inputStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible,
        outputStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible,
        instanceStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible
    ) throws -> GaussianBLASTriangleGeometry {
        guard !gaussians.isEmpty else {
            throw Error.emptyInput
        }

        let gaussianBuffer = try makeGaussianBuffer(from: gaussians, options: inputStorageMode)
        gaussianBuffer.label = "Gaussian BLAS Instance Gaussians"

        let canonicalGaussian = Self.canonicalGaussian(for: primitive)
        let canonicalKernelScale = Self.kernelScale(density: canonicalGaussian.density, kernel: kernel)
        let canonicalGeometry = try buildPrimitiveBuffers(
            gaussians: [canonicalGaussian],
            primitive: primitive,
            kernel: kernel,
            commandQueue: commandQueue,
            inputStorageMode: inputStorageMode,
            outputStorageMode: outputStorageMode
        )

        let instanceTransforms = gaussians.map { gaussian in
            Self.instanceTransform(for: gaussian,
                                   primitive: primitive,
                                   kernel: kernel,
                                   canonicalKernelScale: canonicalKernelScale)
        }
        let instanceDescriptors = instanceTransforms.enumerated().map { index, transform -> MTLAccelerationStructureUserIDInstanceDescriptor in
            var descriptor = MTLAccelerationStructureUserIDInstanceDescriptor()
            descriptor.accelerationStructureIndex = 0
            descriptor.userID = UInt32(index)
            descriptor.mask = UInt32(GEOMETRY_MASK_TRIANGLE)
            descriptor.options = []
            descriptor.transformationMatrix = Self.packedTransform(transform)
            return descriptor
        }

        guard let instanceDescriptorBuffer = makeBuffer(from: instanceDescriptors, options: instanceStorageMode) else {
            throw Error.instanceDescriptorBufferAllocationFailed
        }
        instanceDescriptorBuffer.label = "Gaussian BLAS Instance Descriptors"

        guard let transformBuffer = makeBuffer(from: instanceTransforms, options: instanceStorageMode) else {
            throw Error.instanceTransformBufferAllocationFailed
        }
        transformBuffer.label = "Gaussian BLAS Instance Transforms"

        let surfelNormalDensityBuffer: MTLBuffer?
        if primitive.emitsSurfelMetadata {
            let values = gaussians.map { gaussian in
                Self.surfelNormalDensity(for: gaussian, primitive: primitive)
            }
            guard let buffer = makeBuffer(from: values, options: instanceStorageMode) else {
                throw Error.surfelMetadataBufferAllocationFailed
            }
            buffer.label = "Gaussian BLAS Instance Surfel NormalDensity"
            surfelNormalDensityBuffer = buffer
        } else {
            surfelNormalDensityBuffer = nil
        }

        let primitiveGeometry = GaussianBLASPrimitiveGeometry(
            primitive: primitive,
            canonicalKernelScale: canonicalKernelScale,
            storage: .triangles(vertexBuffer: canonicalGeometry.vertexBuffer,
                                indexBuffer: canonicalGeometry.indexBuffer,
                                canonicalSurfelNormalDensityBuffer: canonicalGeometry.surfelNormalDensityBuffer)
        )
        let instances = GaussianBLASInstanceGeometry(
            primitive: primitive,
            gaussianCount: gaussians.count,
            gaussianBuffer: gaussianBuffer,
            instanceDescriptorBuffer: instanceDescriptorBuffer,
            transformBuffer: transformBuffer,
            surfelNormalDensityBuffer: surfelNormalDensityBuffer
        )
        return GaussianBLASTriangleGeometry(primitiveGeometry: primitiveGeometry, instances: instances)
    }

    func makeBoundingBoxGeometry(
        gaussians: [GaussianSplat],
        primitive: GaussianEnclosingPrimitive,
        kernel: GaussianKernelConfiguration = GaussianKernelConfiguration(),
        instanceStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible,
        outputStorageMode: MTLResourceOptions = GaussianBLASStorageMode.sharedCPUAccessible
    ) throws -> GaussianBLASTriangleGeometry {
        guard !gaussians.isEmpty else {
            throw Error.emptyInput
        }

        let gaussianBuffer = try makeGaussianBuffer(from: gaussians, options: instanceStorageMode)
        gaussianBuffer.label = "Gaussian BLAS Instance Gaussians"

        let canonicalGaussian = Self.canonicalGaussian(for: primitive)
        let canonicalKernelScale = Self.kernelScale(density: canonicalGaussian.density, kernel: kernel)
        let canonicalGeometry = try buildBoundingBoxBuffers(primitive: primitive,
                                                            kernel: kernel,
                                                            outputStorageMode: outputStorageMode)

        let instanceTransforms = gaussians.map { gaussian in
            Self.instanceTransform(for: gaussian,
                                   primitive: primitive,
                                   kernel: kernel,
                                   canonicalKernelScale: canonicalKernelScale)
        }
        let instanceDescriptors = instanceTransforms.enumerated().map { index, transform -> MTLAccelerationStructureUserIDInstanceDescriptor in
            var descriptor = MTLAccelerationStructureUserIDInstanceDescriptor()
            descriptor.accelerationStructureIndex = 0
            descriptor.userID = UInt32(index)
            descriptor.mask = UInt32(GEOMETRY_MASK_TRIANGLE)
            descriptor.options = []
            descriptor.intersectionFunctionTableOffset = 0
            descriptor.transformationMatrix = Self.packedTransform(transform)
            return descriptor
        }

        guard let instanceDescriptorBuffer = makeBuffer(from: instanceDescriptors, options: instanceStorageMode) else {
            throw Error.instanceDescriptorBufferAllocationFailed
        }
        instanceDescriptorBuffer.label = "Gaussian BLAS Bounding Box Instance Descriptors"

        guard let transformBuffer = makeBuffer(from: instanceTransforms, options: instanceStorageMode) else {
            throw Error.instanceTransformBufferAllocationFailed
        }
        transformBuffer.label = "Gaussian BLAS Bounding Box Instance Transforms"

        let surfelNormalDensityBuffer: MTLBuffer?
        if primitive.emitsSurfelMetadata {
            let values = gaussians.map { gaussian in
                Self.surfelNormalDensity(for: gaussian, primitive: primitive)
            }
            guard let buffer = makeBuffer(from: values, options: instanceStorageMode) else {
                throw Error.surfelMetadataBufferAllocationFailed
            }
            buffer.label = "Gaussian BLAS Bounding Box Instance Surfel NormalDensity"
            surfelNormalDensityBuffer = buffer
        } else {
            surfelNormalDensityBuffer = nil
        }

        let primitiveGeometry = GaussianBLASPrimitiveGeometry(
            primitive: primitive,
            canonicalKernelScale: canonicalKernelScale,
            storage: .boundingBoxes(buffer: canonicalGeometry.boundingBoxBuffer, count: 1)
        )
        let instances = GaussianBLASInstanceGeometry(
            primitive: primitive,
            gaussianCount: gaussians.count,
            gaussianBuffer: gaussianBuffer,
            instanceDescriptorBuffer: instanceDescriptorBuffer,
            transformBuffer: transformBuffer,
            surfelNormalDensityBuffer: surfelNormalDensityBuffer
        )
        return GaussianBLASTriangleGeometry(primitiveGeometry: primitiveGeometry, instances: instances)
    }

    private func makeGaussianBuffer(from gaussians: [GaussianSplat], options: MTLResourceOptions) throws -> MTLBuffer {
        let gpuGaussians: [GaussianBLASGaussian] = gaussians.map { gaussian in
            var gpu = GaussianBLASGaussian()
            gpu.positionDensity = SIMD4<Float>(gaussian.position.x, gaussian.position.y, gaussian.position.z, gaussian.density)
            gpu.rotationWXYZ = gaussian.rotationWXYZ
            gpu.scaleReserved = SIMD4<Float>(gaussian.scale.x, gaussian.scale.y, gaussian.scale.z, 0)
            
            withUnsafeMutablePointer(to: &gpu.sh) { ptr in
                let dst = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: Float.self)
                for i in 0 ..< min(48, gaussian.sh.count) {
                    dst[i] = gaussian.sh[i]
                }
            }
            return gpu
        }

        guard let buffer = makeBuffer(from: gpuGaussians, options: options) else {
            throw Error.gaussianBufferAllocationFailed
        }
        return buffer
    }

    private func makeUniformBuffer(
        primitive: GaussianEnclosingPrimitive,
        gaussianCount: Int,
        kernel: GaussianKernelConfiguration,
        options: MTLResourceOptions
    ) throws -> MTLBuffer {
        var uniforms = GaussianBLASBuildUniforms()
        uniforms.primitiveType = primitive.shaderValue
        uniforms.gaussianCount = UInt32(gaussianCount)
        uniforms.verticesPerGaussian = UInt32(primitive.verticesPerGaussian)
        uniforms.trianglesPerGaussian = UInt32(primitive.trianglesPerGaussian)
        uniforms.kernelOptions = kernel.options.rawValue
        uniforms.emitsSurfelMetadata = primitive.emitsSurfelMetadata ? 1 : 0
        uniforms.kernelMinResponse = kernel.minResponse
        uniforms.kernelDegree = kernel.degree

        guard let buffer = makeBuffer(from: [uniforms], options: options) else {
            throw Error.uniformBufferAllocationFailed
        }
        return buffer
    }

    private func makeBuffer<T>(from values: [T], options: MTLResourceOptions) -> MTLBuffer? {
        values.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                return nil
            }

            let buffer = device.makeBuffer(bytes: baseAddress, length: bytes.count, options: options)
            #if os(macOS)
            if options.contains(.storageModeManaged) {
                buffer?.didModifyRange(0..<bytes.count)
            }
            #endif
            return buffer
        }
    }

    private static func canonicalGaussian(for primitive: GaussianEnclosingPrimitive) -> GaussianSplat {
        let canonicalScale: SIMD3<Float>
        switch primitive {
        case .minAxisTriSurfel:
            canonicalScale = SIMD3<Float>(1.0, 1.0, 0.5)
        default:
            canonicalScale = SIMD3<Float>(repeating: 1.0)
        }

        var sh = [Float](repeating: 0, count: 48)
        let C0: Float = 0.28209479177387814
        sh[0] = (1.0 - 0.5) / C0
        sh[1] = (1.0 - 0.5) / C0
        sh[2] = (1.0 - 0.5) / C0

        return GaussianSplat(position: .zero,
                             rotationWXYZ: SIMD4<Float>(1, 0, 0, 0),
                             scale: canonicalScale,
                             density: 1.0,
                             sh: sh)
    }

    private static func kernelScale(density: Float, kernel: GaussianKernelConfiguration) -> Float {
        let responseModulation = kernel.options.contains(.adaptiveDensityClamping) ? density : 1.0
        let modulatedMinResponse = min(kernel.minResponse / max(responseModulation, 1.0e-6), 0.97)

        if kernel.degree < 0.0 {
            let k = abs(kernel.degree)
            let s = 1.0 / pow(3.0, k)
            return pow((1.0 / (log(modulatedMinResponse) - 1.0) + 1.0) / s, 1.0 / k)
        }

        if kernel.degree == 0.0 {
            return ((1.0 - modulatedMinResponse) / 3.0) / -0.329630334487
        }

        let a = -4.5 / pow(3.0, kernel.degree)
        return pow(log(modulatedMinResponse) / a, 1.0 / kernel.degree)
    }

    private static func surfelAxis(for primitive: GaussianEnclosingPrimitive, scale: SIMD3<Float>) -> Int {
        switch primitive {
        case .minAxisTriSurfel:
            if scale.y < scale.x {
                return scale.z < scale.y ? 2 : 1
            }
            return scale.z < scale.x ? 2 : 0
        case .triSurfel:
            return 2
        default:
            return 2
        }
    }

    private static func surfelNormalDensity(for gaussian: GaussianSplat,
                                            primitive: GaussianEnclosingPrimitive) -> SIMD4<Float> {
        let axis = surfelAxis(for: primitive, scale: gaussian.scale)
        let localNormal: SIMD3<Float>
        switch axis {
        case 0:
            localNormal = SIMD3<Float>(1, 0, 0)
        case 1:
            localNormal = SIMD3<Float>(0, 1, 0)
        default:
            localNormal = SIMD3<Float>(0, 0, 1)
        }

        let rotation = quaternionMatrix(fromWXYZ: gaussian.rotationWXYZ)
        let rotated = rotation * SIMD4<Float>(localNormal.x, localNormal.y, localNormal.z, 0)
        let worldNormal = simd_normalize(SIMD3<Float>(rotated.x, rotated.y, rotated.z))
        return SIMD4<Float>(worldNormal.x, worldNormal.y, worldNormal.z, gaussian.density)
    }

    private static func instanceTransform(for gaussian: GaussianSplat,
                                          primitive: GaussianEnclosingPrimitive,
                                          kernel: GaussianKernelConfiguration,
                                          canonicalKernelScale: Float) -> matrix_float4x4 {
        let actualKernelScale = kernelScale(density: gaussian.density, kernel: kernel)
        let relativeScale = gaussian.scale * (actualKernelScale / max(canonicalKernelScale, 1.0e-6))
        let translation = matrix_float4x4(translation: gaussian.position)
        let rotation = quaternionMatrix(fromWXYZ: gaussian.rotationWXYZ)
        let scale = matrix_float4x4(scale: relativeScale)
        let localAlignment = surfelAlignmentMatrix(for: primitive, scale: gaussian.scale)
        return translation * rotation * scale * localAlignment
    }

    private static func surfelAlignmentMatrix(for primitive: GaussianEnclosingPrimitive,
                                              scale: SIMD3<Float>) -> matrix_float4x4 {
        switch primitive {
        case .minAxisTriSurfel:
            switch surfelAxis(for: primitive, scale: scale) {
            case 0:
                return matrix_float4x4(columns: (
                    SIMD4<Float>(0, 1, 0, 0),
                    SIMD4<Float>(0, 0, 1, 0),
                    SIMD4<Float>(1, 0, 0, 0),
                    SIMD4<Float>(0, 0, 0, 1)
                ))
            case 1:
                return matrix_float4x4(columns: (
                    SIMD4<Float>(0, 0, 1, 0),
                    SIMD4<Float>(1, 0, 0, 0),
                    SIMD4<Float>(0, 1, 0, 0),
                    SIMD4<Float>(0, 0, 0, 1)
                ))
            default:
                return matrix_identity_float4x4
            }
        default:
            return matrix_identity_float4x4
        }
    }

    private static func quaternionMatrix(fromWXYZ quaternion: SIMD4<Float>) -> matrix_float4x4 {
        let r = quaternion.x
        let x = quaternion.y
        let y = quaternion.z
        let z = quaternion.w

        return matrix_float4x4(columns: (
            SIMD4<Float>(1.0 - 2.0 * (y * y + z * z),
                         2.0 * (x * y + r * z),
                         2.0 * (x * z - r * y),
                         0.0),
            SIMD4<Float>(2.0 * (x * y - r * z),
                         1.0 - 2.0 * (x * x + z * z),
                         2.0 * (y * z + r * x),
                         0.0),
            SIMD4<Float>(2.0 * (x * z + r * y),
                         2.0 * (y * z - r * x),
                         1.0 - 2.0 * (x * x + y * y),
                         0.0),
            SIMD4<Float>(0.0, 0.0, 0.0, 1.0)
        ))
    }

    private static func packedTransform(_ matrix: matrix_float4x4) -> MTLPackedFloat4x3 {
        MTLPackedFloat4x3(columns: (
            packedFloat3(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z),
            packedFloat3(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z),
            packedFloat3(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z),
            packedFloat3(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z)
        ))
    }

    private static func packedFloat3(_ x: Float, _ y: Float, _ z: Float) -> MTLPackedFloat3 {
        var value = MTLPackedFloat3()
        value.x = x
        value.y = y
        value.z = z
        return value
    }
}

private extension matrix_float4x4 {
    init(translation: SIMD3<Float>) {
        self = matrix_float4x4(columns: (
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0, 1, 0, 0),
            SIMD4<Float>(0, 0, 1, 0),
            SIMD4<Float>(translation.x, translation.y, translation.z, 1)
        ))
    }

    init(scale: SIMD3<Float>) {
        self = matrix_float4x4(columns: (
            SIMD4<Float>(scale.x, 0, 0, 0),
            SIMD4<Float>(0, scale.y, 0, 0),
            SIMD4<Float>(0, 0, scale.z, 0),
            SIMD4<Float>(0, 0, 0, 1)
        ))
    }
}
