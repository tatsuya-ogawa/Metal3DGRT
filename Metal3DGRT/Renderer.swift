//
//  Renderer.swift
//  Metal3DGRT
//

import Metal
import MetalKit
import QuartzCore
import simd

struct RendererStatistics {
    let fps: Double
    let frameTimeMilliseconds: Double
    let raytracingTimeMilliseconds: Double
    let drawableSize: CGSize
    let gaussianCount: Int
    let primitiveName: String
    let cameraDistance: Float
}

final class Renderer: NSObject {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary

    var statisticsHandler: ((RendererStatistics) -> Void)?
    weak var progressDelegate: RendererProgressDelegate?

    private let mtkView: MTKView
    private var gaussianRenderer: GaussianRaytracingRenderer?
    private var currentGaussians: [GaussianSplat] = []
    private var isLoading: Bool = false {
        didSet {
            loadingStatusChanged?(isLoading)
        }
    }
    var loadingStatusChanged: ((Bool) -> Void)?
    private let presentPipelineState: MTLRenderPipelineState
    private var renderTexture: MTLTexture?

    private var orbitYaw: Float = .pi
    private var orbitPitch: Float = 0.08
    private var orbitDistance: Float = 4.6
    private let orbitTarget = SIMD3<Float>(0.0, 0.95, 0.0)
    private let verticalFieldOfView: Float = .pi / 3.0
    private var lastDrawableSize: CGSize = .zero
    var visualizeRawPolygonHit: Bool = false {
        didSet {
            guard oldValue != visualizeRawPolygonHit else { return }
            resetStrideProgress()
        }
    }
    var selectedPrimitive: GaussianEnclosingPrimitive = .octahedron {
        didSet {
            guard oldValue != selectedPrimitive else { return }
            resetStrideProgress()
            rebuildGaussianRenderer()
        }
    }
    var selectedIntersectionMode: GaussianRayIntersectionMode = .boundingBox {
        didSet {
            guard oldValue != selectedIntersectionMode else { return }
            resetStrideProgress()
            rebuildGaussianRenderer()
        }
    }
    var shDegree: Int = 3 {
        didSet {
            guard oldValue != shDegree else { return }
            resetStrideProgress()
            try? gaussianRenderer?.updateSHDegree(shDegree)
        }
    }
    var maxIntersectionCount: Int = 3 {
        didSet {
            guard oldValue != maxIntersectionCount else { return }
            resetStrideProgress()
        }
    }
    var strideSize: Int = 3 {
        didSet {
            guard oldValue != strideSize else { return }
            resetStrideProgress()
        }
    }
    private var stridePhaseIndex: Int = 0
    private var progressGeneration: Int = 0
    private var totalStridePhases: Int {
        let stride = max(1, strideSize)
        return stride * stride
    }

    private var totalPhasesForVisualize: Int {
        totalStridePhases * 3
    }

    private var lastCameraYaw: Float = .pi
    private var lastCameraPitch: Float = 0.08
    private var lastCameraDistance: Float = 4.6

    private let timingLock = NSLock()
    private var lastFrameTimestamp = CACurrentMediaTime()
    private var statisticsWindowStart = CACurrentMediaTime()
    private var accumulatedFrameTime = 0.0
    private var framesInStatisticsWindow = 0
    private let statisticsUpdateInterval = 0.25
    private var accumulatedRaytracingGPUTime = 0.0
    private var timingSampleCount = 0

    init?(metalView: MTKView, gaussians: [GaussianSplat], progressDelegate: RendererProgressDelegate? = nil) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary()
        else {
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue
        self.library = library
        self.mtkView = metalView

        metalView.device = device
        metalView.sampleCount = 1
        metalView.colorPixelFormat = .bgra8Unorm
        metalView.depthStencilPixelFormat = .invalid

        do {
            currentGaussians = gaussians
            gaussianRenderer = try Self.buildGaussianRenderer(device: device,
                                                              library: library,
                                                              commandQueue: commandQueue,
                                                              gaussians: gaussians,
                                                              primitive: selectedPrimitive,
                                                              shDegree: shDegree,
                                                              intersectionMode: selectedIntersectionMode)
        } catch {
            print("Failed to create Gaussian renderer: \(error)")
            return nil
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "Gaussian Presentation Pipeline"
        pipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexShader")
        pipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentShader")
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat

        do {
            presentPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Failed to create presentation pipeline: \(error)")
            return nil
        }

        super.init()

        self.progressDelegate = progressDelegate
        resetStrideProgress()
        metalView.delegate = self
        mtkView(metalView, drawableSizeWillChange: metalView.drawableSize)
    }

    var gaussianCount: Int {
        gaussianRenderer?.gaussianCount ?? 0
    }

    var primitiveName: String {
        gaussianRenderer?.primitiveDisplayName ?? "None"
    }

    func replaceGaussians(_ gaussians: [GaussianSplat]) {
        isLoading = true
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let newRenderer = try self.makeGaussianRenderer(gaussians: gaussians)
                DispatchQueue.main.async {
                    self.currentGaussians = gaussians
                    self.gaussianRenderer = newRenderer
                    self.resetStrideProgress()
                    self.isLoading = false
                    print("Successfully loaded gaussians count: \(gaussians.count).")
                }
            } catch {
                DispatchQueue.main.async {
                    self.isLoading = false
                    print("Failed to load gaussians: \(error)")
                }
            }
        }
    }

    func orbitCamera(deltaX: Float, deltaY: Float) {
        orbitYaw -= deltaX * 0.01
        orbitPitch = simd_clamp(orbitPitch - deltaY * 0.01, -1.35, 1.35)
        resetStrideProgressIfCameraChanged()
    }

    func zoomCamera(magnification: Float) {
        orbitDistance = simd_clamp(orbitDistance * expf(-magnification), 1.2, 20.0)
        resetStrideProgressIfCameraChanged()
    }

    func resetCamera() {
        orbitYaw = .pi
        orbitPitch = 0.08
        orbitDistance = 4.6
        resetStrideProgressIfCameraChanged()
    }

    private func resetStrideProgress() {
        stridePhaseIndex = 0
        progressGeneration += 1
        let progress = RendererProgressSnapshot(
            strideSize: max(1, strideSize),
            completedPhases: 0,
            totalPhases: totalPhasesForVisualize
        )
        if Thread.isMainThread {
            progressDelegate?.renderer(self, didResetProgress: progress)
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.progressDelegate?.renderer(self, didResetProgress: progress)
            }
        }
    }

    private func resetStrideProgressIfCameraChanged() {
        if orbitYaw != lastCameraYaw || orbitPitch != lastCameraPitch || orbitDistance != lastCameraDistance {
            resetStrideProgress()
            lastCameraYaw = orbitYaw
            lastCameraPitch = orbitPitch
            lastCameraDistance = orbitDistance
        }
    }

    private func rebuildGaussianRenderer() {
        guard !currentGaussians.isEmpty, !isLoading else { return }
        isLoading = true
        let gaussians = currentGaussians
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let newRenderer = try self.makeGaussianRenderer(gaussians: gaussians)
                DispatchQueue.main.async {
                    self.gaussianRenderer = newRenderer
                    self.resetStrideProgress()
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.isLoading = false
                    print("Failed to rebuild Gaussian renderer: \(error)")
                }
            }
        }
    }

    private func makeGaussianRenderer(gaussians: [GaussianSplat]) throws -> GaussianRaytracingRenderer {
        try Self.buildGaussianRenderer(device: device,
                                       library: library,
                                       commandQueue: commandQueue,
                                       gaussians: gaussians,
                                       primitive: selectedPrimitive,
                                       shDegree: shDegree,
                                       intersectionMode: selectedIntersectionMode)
    }

    private static func buildGaussianRenderer(device: MTLDevice,
                                              library: MTLLibrary,
                                              commandQueue: MTLCommandQueue,
                                              gaussians: [GaussianSplat],
                                              primitive: GaussianEnclosingPrimitive,
                                              shDegree: Int,
                                              intersectionMode: GaussianRayIntersectionMode) throws -> GaussianRaytracingRenderer {
        try GaussianRaytracingRenderer(device: device,
                                       library: library,
                                       commandQueue: commandQueue,
                                       gaussians: gaussians,
                                       primitive: primitive,
                                       shDegree: shDegree,
                                       intersectionMode: intersectionMode)
    }

    private func makeCamera(for size: CGSize) -> GaussianRaytracingCamera {
        let cosPitch = cos(orbitPitch)
        let position = SIMD3<Float>(
            orbitTarget.x + orbitDistance * cosPitch * sin(orbitYaw),
            orbitTarget.y + orbitDistance * sin(orbitPitch),
            orbitTarget.z + orbitDistance * cosPitch * cos(orbitYaw)
        )
        let aspectRatio = Float(size.width) / max(Float(size.height), 1.0)
        return GaussianRaytracingCamera(position: position,
                                        target: orbitTarget,
                                        verticalFieldOfView: verticalFieldOfView,
                                        aspectRatio: aspectRatio)
    }

    private func recreateRenderTexture(size: CGSize) {
        let width = max(Int(size.width), 1)
        let height = max(Int(size.height), 1)

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                  width: width,
                                                                  height: height,
                                                                  mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        renderTexture = device.makeTexture(descriptor: descriptor)
        renderTexture?.label = "Gaussian Render Texture"
    }

    private func measuredGPUTimeMilliseconds(for commandBuffer: MTLCommandBuffer) -> Double {
        let kernelDuration = commandBuffer.kernelEndTime - commandBuffer.kernelStartTime
        if kernelDuration.isFinite, kernelDuration > 0 {
            return kernelDuration * 1000.0
        }

        let gpuDuration = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        if gpuDuration.isFinite, gpuDuration > 0 {
            return gpuDuration * 1000.0
        }

        return 0.0
    }

    private func recordCompletion(raytracingTime: Double, generation: Int) {
        timingLock.lock()
        accumulatedRaytracingGPUTime += raytracingTime
        timingSampleCount += 1
        timingLock.unlock()
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard generation == self.progressGeneration else { return }
            self.progressDelegate?.rendererDidAdvanceProgress(self)
        }
    }

    private func publishStatisticsIfNeeded(for size: CGSize) {
        let now = CACurrentMediaTime()
        let frameTime = now - lastFrameTimestamp
        lastFrameTimestamp = now

        accumulatedFrameTime += frameTime
        framesInStatisticsWindow += 1

        let elapsed = now - statisticsWindowStart
        guard elapsed >= statisticsUpdateInterval else { return }

        let frameCount = max(framesInStatisticsWindow, 1)
        timingLock.lock()
        let timingCount = max(timingSampleCount, 1)
        let averageRaytracingTime = accumulatedRaytracingGPUTime / Double(timingCount)
        accumulatedRaytracingGPUTime = 0.0
        timingSampleCount = 0
        timingLock.unlock()

        let statistics = RendererStatistics(
            fps: Double(frameCount) / elapsed,
            frameTimeMilliseconds: (accumulatedFrameTime / Double(frameCount)) * 1000.0,
            raytracingTimeMilliseconds: averageRaytracingTime,
            drawableSize: size,
            gaussianCount: gaussianCount,
            primitiveName: primitiveName,
            cameraDistance: orbitDistance
        )

        statisticsWindowStart = now
        accumulatedFrameTime = 0.0
        framesInStatisticsWindow = 0

        DispatchQueue.main.async { [statisticsHandler] in
            statisticsHandler?(statistics)
        }
    }
}

extension Renderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        lastDrawableSize = size
        resetStrideProgress()
        recreateRenderTexture(size: size)
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor
        else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        if let gaussianRenderer, let renderTexture {
            let progressGeneration = self.progressGeneration
            commandBuffer.addCompletedHandler { [weak self] completedBuffer in
                guard let self else { return }
                self.recordCompletion(raytracingTime: self.measuredGPUTimeMilliseconds(for: completedBuffer),
                                      generation: progressGeneration)
            }

            let camera = makeCamera(for: lastDrawableSize)
            let effectiveStride = max(1, strideSize)
            let phaseX = stridePhaseIndex % effectiveStride
            let phaseY = stridePhaseIndex / effectiveStride
            
            do {
                try gaussianRenderer.encode(commandBuffer: commandBuffer,
                                            destinationTexture: renderTexture,
                                            size: lastDrawableSize,
                                            camera: camera,
                                            shDegree: shDegree,
                                            maxIntersectionCount: maxIntersectionCount,
                                            visualizeRawPolygonHit: visualizeRawPolygonHit,
                                            strideSize: effectiveStride,
                                            stridePhaseX: phaseX,
                                            stridePhaseY: phaseY)
            } catch {
                print("Gaussian renderer encode failed: \(error)")
                return
            }
            
            stridePhaseIndex = (stridePhaseIndex + 1) % totalStridePhases
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        renderEncoder.label = "Gaussian Presentation Pass"
        renderEncoder.setRenderPipelineState(presentPipelineState)
        if let renderTexture {
            renderEncoder.setFragmentTexture(renderTexture, index: 0)
        }
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderEncoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()

        publishStatisticsIfNeeded(for: lastDrawableSize)
    }
}
