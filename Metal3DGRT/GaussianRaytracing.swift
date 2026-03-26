//
//  GaussianRaytracing.swift
//  Metal3DGRT
//

import Metal
import simd

private enum GaussianRaytracingDefaults {
    static let aspectRatio: Float = 1.0
    static let halfFieldOfViewScale: Float = 0.5
    static let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
    static let previewHitColor = SIMD3<Float>(1.0, 0.82, 0.42)
    static let previewMissColor = SIMD3<Float>(0.03, 0.04, 0.06)
    static let testHitColor = SIMD3<Float>(1.0, 0.8, 0.35)
    static let testMissColor = SIMD3<Float>(0.02, 0.03, 0.05)
    static let previewBodyCenter = SIMD3<Float>(0.0, 0.95, 0.0)
    static let previewBodyScale = SIMD3<Float>(0.60, 0.42, 0.60)
    static let previewBodyDensity: Float = 1.0
    static let previewLowerLobeCenter = SIMD3<Float>(0.0, 0.38, 0.0)
    static let previewLowerLobeScale = SIMD3<Float>(0.42, 0.62, 0.42)
    static let previewLowerLobeDensity: Float = 0.9
    static let previewLeftShoulderCenter = SIMD3<Float>(-0.48, 1.02, 0.08)
    static let previewShoulderScale = SIMD3<Float>(0.24, 0.24, 0.24)
    static let previewShoulderDensity: Float = 0.8
    static let previewRightShoulderCenter = SIMD3<Float>(0.48, 1.04, 0.12)
    static let previewRightShoulderScale = SIMD3<Float>(0.28, 0.20, 0.28)
    static let previewHeadCenter = SIMD3<Float>(0.0, 1.55, -0.05)
    static let previewHeadScale = SIMD3<Float>(0.30, 0.30, 0.30)
    static let previewHeadDensity: Float = 0.75
}

struct GaussianRaytracingCamera {
    var position: SIMD3<Float>
    var forward: SIMD3<Float>
    var up: SIMD3<Float>
    var right: SIMD3<Float>

    init(position: SIMD3<Float>,
         target: SIMD3<Float>,
         upHint: SIMD3<Float> = SIMD3<Float>(0, 1, 0),
         verticalFieldOfView: Float = .pi / 3,
         aspectRatio: Float = GaussianRaytracingDefaults.aspectRatio) {
        let forward = simd_normalize(target - position)
        let right = simd_normalize(simd_cross(forward, upHint))
        let up = simd_normalize(simd_cross(right, forward))
        let tanHalfFov = tan(verticalFieldOfView * GaussianRaytracingDefaults.halfFieldOfViewScale)

        self.position = position
        self.forward = forward
        self.up = up * tanHalfFov
        self.right = right * tanHalfFov * aspectRatio
    }

    fileprivate var shaderCamera: Camera {
        var camera = Camera()
        camera.position = position
        camera.forward = forward
        camera.up = up
        camera.right = right
        return camera
    }
}

enum GaussianRayIntersectionMode: CaseIterable {
    case triangle
    case boundingBox

    var functionConstantValue: UInt32 {
        switch self {
        case .triangle:
            return 0
        case .boundingBox:
            return 1
        }
    }

    var displayName: String {
        switch self {
        case .triangle:
            return "Triangle"
        case .boundingBox:
            return "BoundingBox"
        }
    }
}

final class GaussianRaytracingRenderer {
    enum Error: Swift.Error {
        case accelerationStructureBuildFailed
        case bufferAllocationFailed
        case commandEncoderCreationFailed
        case defaultLibraryUnavailable
        case kernelFunctionNotFound
        case pipelineCreationFailed(Swift.Error)
        case sceneCreationFailed(Swift.Error)
    }

    private let device: MTLDevice
    private var pipelineState: MTLComputePipelineState
    private let geometry: GaussianBLASTriangleGeometry
    private let primitiveAccelerationStructure: MTLAccelerationStructure
    private let accelerationStructure: MTLAccelerationStructure
    private let uniformBuffer: MTLBuffer
    private let primitiveType: GaussianEnclosingPrimitive
    private let kernelConfiguration: GaussianKernelConfiguration
    private let intersectionMode: GaussianRayIntersectionMode
    private let sceneAabbMin: SIMD4<Float>
    private let sceneAabbMax: SIMD4<Float>
    private var shDegree: Int

    var gaussianCount: Int {
        geometry.gaussianCount
    }

    var primitiveDisplayName: String {
        "\(primitiveType.displayName) / \(intersectionMode.displayName)"
    }

    convenience init(device: MTLDevice,
                     commandQueue: MTLCommandQueue,
                     gaussians: [GaussianSplat],
                     primitive: GaussianEnclosingPrimitive = .octahedron,
                     kernel: GaussianKernelConfiguration = GaussianKernelConfiguration(),
                     intersectionMode: GaussianRayIntersectionMode = .boundingBox) throws {
        guard let library = device.makeDefaultLibrary() else {
            throw Error.defaultLibraryUnavailable
        }
        try self.init(device: device,
                      library: library,
                      commandQueue: commandQueue,
                      gaussians: gaussians,
                      primitive: primitive,
                      kernel: kernel,
                      intersectionMode: intersectionMode)
    }

    init(device: MTLDevice,
         library: MTLLibrary,
         commandQueue: MTLCommandQueue,
         gaussians: [GaussianSplat],
         primitive: GaussianEnclosingPrimitive = .octahedron,
         kernel: GaussianKernelConfiguration = GaussianKernelConfiguration(),
         shDegree: Int = 3,
         intersectionMode: GaussianRayIntersectionMode = .boundingBox) throws {
        self.device = device
        self.primitiveType = primitive
        self.kernelConfiguration = kernel
        self.intersectionMode = intersectionMode
        self.shDegree = shDegree

        var aabbMin = SIMD3<Float>(repeating: .infinity)
        var aabbMax = SIMD3<Float>(repeating: -.infinity)
        for g in gaussians {
            let bounds = Self.sceneBounds(for: g, kernel: kernel)
            aabbMin = simd_min(aabbMin, bounds.min)
            aabbMax = simd_max(aabbMax, bounds.max)
        }
        self.sceneAabbMin = SIMD4<Float>(aabbMin.x, aabbMin.y, aabbMin.z, 1.0)
        self.sceneAabbMax = SIMD4<Float>(aabbMax.x, aabbMax.y, aabbMax.z, 1.0)

        let blasBuilder = try GaussianBLASBufferBuilder(device: device, library: library)
        do {
            switch intersectionMode {
            case .triangle:
                self.geometry = try blasBuilder.makeTriangleGeometry(gaussians: gaussians,
                                                                     primitive: primitive,
                                                                     kernel: kernel,
                                                                     commandQueue: commandQueue)
            case .boundingBox:
                self.geometry = try blasBuilder.makeBoundingBoxGeometry(gaussians: gaussians,
                                                                        primitive: primitive,
                                                                        kernel: kernel)
            }
        } catch {
            throw Error.sceneCreationFailed(error)
        }

        do {
            self.pipelineState = try Self.makePipelineState(device: device,
                                                            library: library,
                                                            kernelConfiguration: kernel,
                                                            primitiveType: primitive,
                                                            shDegree: shDegree,
                                                            intersectionMode: intersectionMode)
        } catch {
            throw Error.pipelineCreationFailed(error)
        }

        guard let primitiveAccelerationStructure = Self.buildCompactedAccelerationStructure(
            device: device,
            descriptor: geometry.primitiveGeometry.makePrimitiveAccelerationStructureDescriptor(),
            commandQueue: commandQueue
        ) else {
            throw Error.accelerationStructureBuildFailed
        }
        self.primitiveAccelerationStructure = primitiveAccelerationStructure

        let instanceDescriptor = geometry.makeInstanceAccelerationStructureDescriptor(
            primitiveAccelerationStructure: primitiveAccelerationStructure
        )
        guard let accelerationStructure = Self.buildCompactedAccelerationStructure(
            device: device,
            descriptor: instanceDescriptor,
            commandQueue: commandQueue
        ) else {
            throw Error.accelerationStructureBuildFailed
        }
        self.accelerationStructure = accelerationStructure

        guard let uniformBuffer = device.makeBuffer(length: MemoryLayout<GaussianRaytracingUniforms>.stride,
                                                    options: .storageModeShared) else {
            throw Error.bufferAllocationFailed
        }
        uniformBuffer.label = "Gaussian Raytracing Uniforms"
        self.uniformBuffer = uniformBuffer
    }

    func updateSHDegree(_ newDegree: Int) throws {
        if shDegree == newDegree { return }
        shDegree = newDegree

        guard let library = device.makeDefaultLibrary() else {
            throw Error.defaultLibraryUnavailable
        }

        let newPipelineState = try Self.makePipelineState(device: device,
                                                          library: library,
                                                          kernelConfiguration: kernelConfiguration,
                                                          primitiveType: primitiveType,
                                                          shDegree: shDegree,
                                                          intersectionMode: intersectionMode)
        self.pipelineState = newPipelineState
    }

    func encode(commandBuffer: MTLCommandBuffer,
                destinationTexture: MTLTexture,
                size: CGSize,
                camera: GaussianRaytracingCamera,
                shDegree: Int = 0,
                maxIntersectionCount: Int = 3,
                visualizeRawPolygonHit: Bool = false,
                strideSize: Int = 1,
                stridePhaseX: Int = 0,
                stridePhaseY: Int = 0,
                hitColor: SIMD3<Float> = GaussianRaytracingDefaults.previewHitColor,
                missColor: SIMD3<Float> = GaussianRaytracingDefaults.previewMissColor) throws {
        var uniforms = GaussianRaytracingUniforms()
        uniforms.width = UInt32(max(Int(size.width), 1))
        uniforms.height = UInt32(max(Int(size.height), 1))
        uniforms.gaussianCount = UInt32(geometry.gaussianCount)
        uniforms.visualizeRawPolygonHit = visualizeRawPolygonHit ? 1 : 0
        uniforms.camera = camera.shaderCamera
        uniforms.hitColor = SIMD4<Float>(hitColor.x, hitColor.y, hitColor.z, 1.0)
        uniforms.missColor = SIMD4<Float>(missColor.x, missColor.y, missColor.z, 1.0)
        uniforms.kernelMinResponse = kernelConfiguration.minResponse
        uniforms.maxIntersectionCount = UInt32(max(1, maxIntersectionCount))
        uniforms.sceneAabbMin = sceneAabbMin
        uniforms.sceneAabbMax = sceneAabbMax
        let clampedStride = max(1, strideSize)
        uniforms.strideSize = UInt32(clampedStride)
        uniforms.stridePhaseX = UInt32(stridePhaseX % clampedStride)
        uniforms.stridePhaseY = UInt32(stridePhaseY % clampedStride)
        withUnsafeBytes(of: uniforms) { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            uniformBuffer.contents().copyMemory(from: baseAddress, byteCount: bytes.count)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Error.commandEncoderCreationFailed
        }
        encoder.label = "Gaussian Raytracing Pass"
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(uniformBuffer, offset: 0, index: 0)
        encoder.setBuffer(geometry.instances.gaussianBuffer, offset: 0, index: 1)
        encoder.setAccelerationStructure(accelerationStructure, bufferIndex: 2)
        encoder.setTexture(destinationTexture, index: 0)
        encoder.useResource(accelerationStructure, usage: .read)
        encoder.useResource(primitiveAccelerationStructure, usage: .read)
        encoder.useResource(geometry.instances.gaussianBuffer, usage: .read)
        encoder.useResource(geometry.instances.instanceDescriptorBuffer, usage: .read)
        encoder.useResource(uniformBuffer, usage: .read)
        encoder.useResource(destinationTexture, usage: .write)

        let threadsPerThreadgroup = GaussianRaytracingDefaults.threadgroupSize
        let stridedWidth = (max(Int(size.width), 1) + clampedStride - 1) / clampedStride
        let stridedHeight = (max(Int(size.height), 1) + clampedStride - 1) / clampedStride
        let threadsPerGrid = MTLSize(width: stridedWidth,
                                     height: stridedHeight,
                                     depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    private static func makePipelineState(device: MTLDevice,
                                          library: MTLLibrary,
                                          kernelConfiguration: GaussianKernelConfiguration,
                                          primitiveType: GaussianEnclosingPrimitive,
                                          shDegree: Int,
                                          intersectionMode: GaussianRayIntersectionMode) throws -> MTLComputePipelineState {
        let constantValues = MTLFunctionConstantValues()
        var degree = kernelConfiguration.degree
        constantValues.setConstantValue(&degree, type: .float, index: 0)

        var primitiveTypeValue = primitiveType.intersectionPrimitiveType
        constantValues.setConstantValue(&primitiveTypeValue, type: .uint, index: 1)

        var shDegreeValue = UInt32(shDegree)
        constantValues.setConstantValue(&shDegreeValue, type: .uint, index: 2)

        var intersectionModeValue = intersectionMode.functionConstantValue
        constantValues.setConstantValue(&intersectionModeValue, type: .uint, index: 3)

        guard let kernelFunction = try? library.makeFunction(name: "gaussianTriangleRaytracingKernel",
                                                             constantValues: constantValues) else {
            throw Error.kernelFunctionNotFound
        }

        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = kernelFunction
        return try device.makeComputePipelineState(descriptor: descriptor,
                                                   options: [],
                                                   reflection: nil)
    }

    private static func kernelScale(density: Float,
                                    kernel: GaussianKernelConfiguration) -> Float {
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

    private static func sceneBounds(for gaussian: GaussianSplat,
                                    kernel: GaussianKernelConfiguration) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        let extent = simd_length(gaussian.scale * kernelScale(density: gaussian.density, kernel: kernel))
        let radius = SIMD3<Float>(repeating: max(extent, 1.0e-3))
        return (gaussian.position - radius, gaussian.position + radius)
    }


    private static func buildCompactedAccelerationStructure(
        device: MTLDevice,
        descriptor: MTLAccelerationStructureDescriptor,
        commandQueue: MTLCommandQueue
    ) -> MTLAccelerationStructure? {
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard
            let scratchBuffer = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate),
            let compactedSizeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
            let buildCommandBuffer = commandQueue.makeCommandBuffer(),
            let buildEncoder = buildCommandBuffer.makeAccelerationStructureCommandEncoder(),
            let accelerationStructure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize)
        else {
            return nil
        }

        buildEncoder.build(accelerationStructure: accelerationStructure,
                           descriptor: descriptor,
                           scratchBuffer: scratchBuffer,
                           scratchBufferOffset: 0)
        buildEncoder.writeCompactedSize(accelerationStructure: accelerationStructure,
                                        buffer: compactedSizeBuffer,
                                        offset: 0)
        buildEncoder.endEncoding()
        buildCommandBuffer.commit()
        buildCommandBuffer.waitUntilCompleted()
        if buildCommandBuffer.status == .error {
            return nil
        }

        let compactedSize = compactedSizeBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
        guard
            let compactedAccelerationStructure = device.makeAccelerationStructure(size: Int(compactedSize)),
            let compactCommandBuffer = commandQueue.makeCommandBuffer(),
            let compactEncoder = compactCommandBuffer.makeAccelerationStructureCommandEncoder()
        else {
            return nil
        }

        compactEncoder.copyAndCompact(sourceAccelerationStructure: accelerationStructure,
                                      destinationAccelerationStructure: compactedAccelerationStructure)
        compactEncoder.endEncoding()
        compactCommandBuffer.commit()
        compactCommandBuffer.waitUntilCompleted()
        return compactCommandBuffer.status == .error ? nil : compactedAccelerationStructure
    }
}

private extension MTLResourceOptions {
    var storageMode: MTLStorageMode {
        if contains(.storageModePrivate) {
            return .private
        }
        if contains(.storageModeMemoryless) {
            return .memoryless
        }
        #if os(macOS)
        if contains(.storageModeManaged) {
            return .managed
        }
        #endif
        return .shared
    }
}
