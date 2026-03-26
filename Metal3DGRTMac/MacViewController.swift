//
//  MacViewController.swift
//  Metal3DGRT
//

import Cocoa
import MetalKit

final class MacViewController: NSViewController {
    var renderer: Renderer?
    var mtkView: MTKView!
    private var currentScene: GaussianScene?

    private let controlsContainer = NSVisualEffectView()
    private let statisticsContainer = NSView()
    private let statisticsLabel = NSTextField(labelWithString: "Collecting statistics...")
    private let loadingOverlay = NSView()
    private let loadingPanel = NSVisualEffectView()
    private let loadingIndicator = NSProgressIndicator()
    private let loadingLabel = NSTextField(labelWithString: "Loading...")
    private var pendingLoadingMessage: String?

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let mtkView = view as? MTKView else {
            print("View attached to MacViewController is not an MTKView")
            return
        }

        self.mtkView = mtkView

        configureStatisticsOverlay(in: mtkView)
        configureControlsPanel(in: mtkView)
        configureLoadingOverlay(in: mtkView)
        configureGestures(for: mtkView)
        loadDefaultScene()
    }

    private func configureControlsPanel(in view: MTKView) {
        controlsContainer.removeFromSuperview()
        controlsContainer.subviews.forEach { $0.removeFromSuperview() }
        
        controlsContainer.translatesAutoresizingMaskIntoConstraints = false
        controlsContainer.material = .hudWindow
        controlsContainer.blendingMode = .withinWindow
        controlsContainer.state = .active
        controlsContainer.wantsLayer = true
        controlsContainer.layer?.cornerRadius = 10

        let titleLabel = NSTextField(labelWithString: "Gaussian Preview")
        titleLabel.font = .systemFont(ofSize: 13, weight: .semibold)

        let primitiveLabel = NSTextField(labelWithString: "Renderer state is shown in the stats panel")
        primitiveLabel.font = .systemFont(ofSize: 12)

        let gaussianCountLabel = NSTextField(labelWithString: "Gaussians: \(renderer?.gaussianCount ?? 0)")
        gaussianCountLabel.font = .systemFont(ofSize: 12)

        let cameraHintLabel = NSTextField(labelWithString: "Drag: orbit  Pinch: zoom")
        cameraHintLabel.font = .systemFont(ofSize: 11)
        cameraHintLabel.textColor = .secondaryLabelColor

        let modeLabel = NSTextField(labelWithString: "Mode:")
        modeLabel.font = .systemFont(ofSize: 12)

        let modeControl = NSSegmentedControl(labels: GaussianRayIntersectionMode.allCases.map(\.displayName),
                                             trackingMode: .selectOne,
                                             target: self,
                                             action: #selector(intersectionModeChanged(_:)))
        modeControl.selectedSegment = renderer.flatMap { GaussianRayIntersectionMode.allCases.firstIndex(of: $0.selectedIntersectionMode) } ?? 0

        let modeStack = NSStackView(views: [modeLabel, modeControl])
        modeStack.orientation = .horizontal
        modeStack.spacing = 6

        let primitiveControl = NSPopUpButton(frame: .zero, pullsDown: false)
        GaussianEnclosingPrimitive.allCases.forEach { primitive in
            primitiveControl.addItem(withTitle: primitive.displayName)
        }
        if let renderer,
           let selectedIndex = GaussianEnclosingPrimitive.allCases.firstIndex(of: renderer.selectedPrimitive) {
            primitiveControl.selectItem(at: selectedIndex)
        }
        primitiveControl.target = self
        primitiveControl.action = #selector(primitiveChanged(_:))

        let primitiveControlLabel = NSTextField(labelWithString: "Primitive:")
        primitiveControlLabel.font = .systemFont(ofSize: 12)

        let primitiveControlStack = NSStackView(views: [primitiveControlLabel, primitiveControl])
        primitiveControlStack.orientation = .horizontal
        primitiveControlStack.spacing = 6

        let resetCameraButton = NSButton(title: "Reset Camera", target: self, action: #selector(resetCamera(_:)))
        resetCameraButton.bezelStyle = .rounded

        let rawHitCheckbox = NSButton(checkboxWithTitle: "Raw Polygon Hits", target: self, action: #selector(toggleRawHits(_:)))
        rawHitCheckbox.state = renderer?.visualizeRawPolygonHit == true ? .on : .off
        
        let shDegreeLabel = NSTextField(labelWithString: "SH Degree:")
        shDegreeLabel.font = .systemFont(ofSize: 12)
        
        let shDegreeControl = NSSegmentedControl(labels: ["0", "1", "2", "3"], trackingMode: .selectOne, target: self, action: #selector(shDegreeChanged(_:)))
        shDegreeControl.selectedSegment = renderer?.shDegree ?? 3
        
        let shDegreeStack = NSStackView(views: [shDegreeLabel, shDegreeControl])
        shDegreeStack.orientation = .horizontal
        shDegreeStack.spacing = 6

        let maxIsectLabel = NSTextField(labelWithString: "Max Intersections: \(renderer?.maxIntersectionCount ?? 3)")
        maxIsectLabel.font = .systemFont(ofSize: 12)
        maxIsectLabel.tag = 9001

        let maxIsectSlider = NSSlider(value: Double(renderer?.maxIntersectionCount ?? 3), minValue: 1, maxValue: 64, target: self, action: #selector(maxIntersectionCountChanged(_:)))
        maxIsectSlider.numberOfTickMarks = 64
        maxIsectSlider.allowsTickMarkValuesOnly = true
        maxIsectSlider.tag = 9002

        let maxIsectStack = NSStackView(views: [maxIsectLabel, maxIsectSlider])
        maxIsectStack.orientation = .horizontal
        maxIsectStack.spacing = 6

        let strideLabel = NSTextField(labelWithString: "Stride: \(renderer?.strideSize ?? 3)")
        strideLabel.font = .systemFont(ofSize: 12)
        strideLabel.tag = 9003

        let strideSlider = NSSlider(value: Double(renderer?.strideSize ?? 3), minValue: 1, maxValue: 16, target: self, action: #selector(strideSizeChanged(_:)))
        strideSlider.numberOfTickMarks = 16
        strideSlider.allowsTickMarkValuesOnly = true
        strideSlider.tag = 9004

        let strideStack = NSStackView(views: [strideLabel, strideSlider])
        strideStack.orientation = .horizontal
        strideStack.spacing = 6

        let sceneLabel = NSTextField(labelWithString: "Scene:")
        sceneLabel.font = .systemFont(ofSize: 12)
        
        let sceneControl = NSPopUpButton(frame: .zero, pullsDown: false)
        let availableScenes = GaussianScene.allCases.filter { $0 != .debug }
        availableScenes.forEach { scene in
            sceneControl.addItem(withTitle: scene.displayName)
        }
        if let selectedScene = currentScene,
           let selectedIndex = availableScenes.firstIndex(of: selectedScene) {
            sceneControl.selectItem(at: selectedIndex)
        }
        sceneControl.target = self
        sceneControl.action = #selector(sceneChanged(_:))
        
        let sceneStack = NSStackView(views: [sceneLabel, sceneControl])
        sceneStack.orientation = .horizontal
        sceneStack.spacing = 6

        let loadSPZButton = NSButton(title: "Load local SPZ...", target: self, action: #selector(openSPZFile(_:)))
        loadSPZButton.bezelStyle = .rounded

        let showsPrimitiveControl = (renderer?.selectedIntersectionMode ?? .boundingBox) == .triangle
        var arrangedSubviews: [NSView] = [titleLabel, primitiveLabel, gaussianCountLabel, sceneStack, modeStack]
        if showsPrimitiveControl {
            arrangedSubviews.append(primitiveControlStack)
        }
        arrangedSubviews.append(contentsOf: [cameraHintLabel, shDegreeStack, maxIsectStack, strideStack, rawHitCheckbox, loadSPZButton, resetCameraButton])

        let stack = NSStackView(views: arrangedSubviews)
        stack.orientation = .vertical
        stack.spacing = 6
        stack.alignment = .leading
        stack.translatesAutoresizingMaskIntoConstraints = false

        controlsContainer.addSubview(stack)
        view.addSubview(controlsContainer)

        NSLayoutConstraint.activate([
            controlsContainer.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            controlsContainer.topAnchor.constraint(equalTo: view.topAnchor, constant: 16),

            stack.topAnchor.constraint(equalTo: controlsContainer.topAnchor, constant: 10),
            stack.leadingAnchor.constraint(equalTo: controlsContainer.leadingAnchor, constant: 12),
            stack.trailingAnchor.constraint(equalTo: controlsContainer.trailingAnchor, constant: -12),
            stack.bottomAnchor.constraint(equalTo: controlsContainer.bottomAnchor, constant: -10),
        ])
    }

    private func configureStatisticsOverlay(in view: MTKView) {
        statisticsContainer.translatesAutoresizingMaskIntoConstraints = false
        statisticsContainer.wantsLayer = true
        statisticsContainer.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.6).cgColor
        statisticsContainer.layer?.cornerRadius = 10

        statisticsLabel.translatesAutoresizingMaskIntoConstraints = false
        statisticsLabel.font = .monospacedSystemFont(ofSize: 12, weight: .medium)
        statisticsLabel.textColor = .white
        statisticsLabel.alignment = .left
        statisticsLabel.lineBreakMode = .byWordWrapping
        statisticsLabel.maximumNumberOfLines = 0
        statisticsLabel.cell?.wraps = true

        view.addSubview(statisticsContainer)
        statisticsContainer.addSubview(statisticsLabel)

        NSLayoutConstraint.activate([
            statisticsContainer.topAnchor.constraint(equalTo: view.topAnchor, constant: 16),
            statisticsContainer.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            statisticsContainer.widthAnchor.constraint(lessThanOrEqualToConstant: 260),

            statisticsLabel.topAnchor.constraint(equalTo: statisticsContainer.topAnchor, constant: 10),
            statisticsLabel.leadingAnchor.constraint(equalTo: statisticsContainer.leadingAnchor, constant: 12),
            statisticsLabel.trailingAnchor.constraint(equalTo: statisticsContainer.trailingAnchor, constant: -12),
            statisticsLabel.bottomAnchor.constraint(equalTo: statisticsContainer.bottomAnchor, constant: -10),
        ])
    }

    private func configureLoadingOverlay(in view: MTKView) {
        loadingOverlay.translatesAutoresizingMaskIntoConstraints = false
        loadingOverlay.wantsLayer = true
        loadingOverlay.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.45).cgColor
        loadingOverlay.isHidden = true

        loadingPanel.translatesAutoresizingMaskIntoConstraints = false
        loadingPanel.material = .hudWindow
        loadingPanel.blendingMode = .withinWindow
        loadingPanel.state = .active
        loadingPanel.wantsLayer = true
        loadingPanel.layer?.cornerRadius = 16

        loadingIndicator.translatesAutoresizingMaskIntoConstraints = false
        loadingIndicator.style = .spinning
        loadingIndicator.controlSize = .large
        loadingIndicator.startAnimation(nil)

        loadingLabel.translatesAutoresizingMaskIntoConstraints = false
        loadingLabel.font = .systemFont(ofSize: 14, weight: .medium)
        loadingLabel.textColor = .white
        loadingLabel.alignment = .center
        loadingLabel.maximumNumberOfLines = 2
        loadingLabel.lineBreakMode = .byWordWrapping

        let stack = NSStackView(views: [loadingIndicator, loadingLabel])
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 12

        loadingPanel.addSubview(stack)
        loadingOverlay.addSubview(loadingPanel)
        view.addSubview(loadingOverlay)

        NSLayoutConstraint.activate([
            loadingOverlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            loadingOverlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            loadingOverlay.topAnchor.constraint(equalTo: view.topAnchor),
            loadingOverlay.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            loadingPanel.centerXAnchor.constraint(equalTo: loadingOverlay.centerXAnchor),
            loadingPanel.centerYAnchor.constraint(equalTo: loadingOverlay.centerYAnchor),
            loadingPanel.widthAnchor.constraint(greaterThanOrEqualToConstant: 220),

            stack.topAnchor.constraint(equalTo: loadingPanel.topAnchor, constant: 20),
            stack.leadingAnchor.constraint(equalTo: loadingPanel.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: loadingPanel.trailingAnchor, constant: -24),
            stack.bottomAnchor.constraint(equalTo: loadingPanel.bottomAnchor, constant: -20),
        ])
    }

    private func setLoadingOverlay(visible: Bool, message: String? = nil) {
        if let message {
            pendingLoadingMessage = message
        }
        loadingLabel.stringValue = pendingLoadingMessage ?? "Loading..."
        loadingOverlay.isHidden = !visible
        if visible {
            view.window?.makeFirstResponder(nil)
        } else {
            pendingLoadingMessage = nil
        }
    }

    private func updateStatisticsOverlay(with statistics: RendererStatistics) {
        let width = Int(statistics.drawableSize.width)
        let height = Int(statistics.drawableSize.height)
        let gpuName = mtkView.device?.name ?? "Unknown GPU"
        let progressLine = currentScene?.progress.snapshot.statusLine ?? "Stride progress unavailable"
        statisticsLabel.stringValue = """
        FPS \(String(format: "%.1f", statistics.fps)) | \(String(format: "%.2f", statistics.frameTimeMilliseconds)) ms
        Gaussian RT \(String(format: "%.2f", statistics.raytracingTimeMilliseconds)) ms
        \(statistics.gaussianCount) gaussians | \(statistics.primitiveName)
        \(progressLine)
        Camera distance \(String(format: "%.2f", statistics.cameraDistance))
        \(width)x\(height) | \(gpuName)
        """
    }

    private func configureGestures(for view: MTKView) {
        let panGesture = NSPanGestureRecognizer(target: self, action: #selector(handleOrbitPan(_:)))
        let magnificationGesture = NSMagnificationGestureRecognizer(target: self, action: #selector(handleMagnification(_:)))
        view.addGestureRecognizer(panGesture)
        view.addGestureRecognizer(magnificationGesture)
    }

    @objc private func handleOrbitPan(_ gesture: NSPanGestureRecognizer) {
        guard let renderer else { return }
        let location = gesture.location(in: mtkView)
        guard !controlsContainer.frame.contains(location) else {
            gesture.setTranslation(.zero, in: mtkView)
            return
        }
        let translation = gesture.translation(in: mtkView)
        renderer.orbitCamera(deltaX: Float(translation.x), deltaY: Float(translation.y))
        gesture.setTranslation(.zero, in: mtkView)
    }

    @objc private func handleMagnification(_ gesture: NSMagnificationGestureRecognizer) {
        guard let renderer else { return }
        let location = gesture.location(in: mtkView)
        guard !controlsContainer.frame.contains(location) else {
            gesture.magnification = 0
            return
        }
        renderer.zoomCamera(magnification: Float(gesture.magnification))
        gesture.magnification = 0
    }

    @objc private func resetCamera(_ sender: NSButton) {
        renderer?.resetCamera()
    }

    @objc private func toggleRawHits(_ sender: NSButton) {
        renderer?.visualizeRawPolygonHit = (sender.state == .on)
    }
    
    @objc private func shDegreeChanged(_ sender: NSSegmentedControl) {
        renderer?.shDegree = sender.selectedSegment
    }

    @objc private func primitiveChanged(_ sender: NSPopUpButton) {
        guard sender.indexOfSelectedItem >= 0,
              sender.indexOfSelectedItem < GaussianEnclosingPrimitive.allCases.count else {
            return
        }
        renderer?.selectedPrimitive = GaussianEnclosingPrimitive.allCases[sender.indexOfSelectedItem]
    }

    @objc private func intersectionModeChanged(_ sender: NSSegmentedControl) {
        guard sender.selectedSegment >= 0,
              sender.selectedSegment < GaussianRayIntersectionMode.allCases.count else {
            return
        }
        renderer?.selectedIntersectionMode = GaussianRayIntersectionMode.allCases[sender.selectedSegment]
        configureControlsPanel(in: mtkView)
    }

    @objc private func maxIntersectionCountChanged(_ sender: NSSlider) {
        let value = Int(sender.integerValue)
        renderer?.maxIntersectionCount = value
        if let label = controlsContainer.viewWithTag(9001) as? NSTextField {
            label.stringValue = "Max Intersections: \(value)"
        }
    }

    @objc private func strideSizeChanged(_ sender: NSSlider) {
        let value = Int(sender.integerValue)
        renderer?.strideSize = value
        if let label = controlsContainer.viewWithTag(9003) as? NSTextField {
            label.stringValue = "Stride: \(value)"
        }
    }

    @objc private func sceneChanged(_ sender: NSPopUpButton) {
        let availableScenes = GaussianScene.allCases.filter { $0 != .debug }
        guard sender.indexOfSelectedItem >= 0,
              sender.indexOfSelectedItem < availableScenes.count else {
            return
        }
        let selected = availableScenes[sender.indexOfSelectedItem]
        guard selected != currentScene else { return }
        loadScene(selected)
    }

    @objc private func openSPZFile(_ sender: NSButton) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.init(filenameExtension: "spz")].compactMap({$0})
        
        panel.beginSheetModal(for: self.view.window!) { response in
            if response == .OK, let url = panel.url {
                GaussianSceneLoader.loadGaussians(from: url) { result in
                    DispatchQueue.main.async {
                        switch result {
                        case let .success(gaussians):
                            let scene = GaussianScene.custom(displayName: url.deletingPathExtension().lastPathComponent)
                            self.setLoadingOverlay(visible: true, message: "Loading \(scene.displayName)...")
                            self.currentScene = scene
                            self.installRenderer(with: gaussians)
                        case let .failure(error):
                            self.setLoadingOverlay(visible: false)
                            self.statisticsLabel.stringValue = "Failed to load SPZ: \(error.localizedDescription)"
                        }
                    }
                }
            }
        }
    }

    private func loadDefaultScene() {
        loadScene(GaussianScene.defaultScene)
    }

    private func loadScene(_ scene: GaussianScene) {
        setLoadingOverlay(visible: true, message: "Loading \(scene.displayName)...")
        statisticsLabel.stringValue = "Loading \(scene.displayName)..."
        GaussianSceneLoader.loadScene(scene) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case let .success(gaussians):
                    self.currentScene = scene
                    self.installRenderer(with: gaussians)
                    self.syncUIToRenderer()
                case let .failure(error):
                    self.setLoadingOverlay(visible: false)
                    self.statisticsLabel.stringValue = "Failed to load \(scene.displayName): \(error.localizedDescription)"
                }
            }
        }
    }

    private func syncUIToRenderer() {
        guard isViewLoaded, renderer != nil else { return }
        // Re-configure to ensure all initial values match
        configureControlsPanel(in: mtkView)
    }

    private func installRenderer(with gaussians: [GaussianSplat]) {
        if let renderer {
            renderer.progressDelegate = currentScene?.progress
            renderer.replaceGaussians(gaussians)
            configureControlsPanel(in: mtkView)
            return
        }

        guard let renderer = Renderer(metalView: mtkView,
                                      gaussians: gaussians,
                                      progressDelegate: currentScene?.progress) else {
            setLoadingOverlay(visible: false)
            statisticsLabel.stringValue = "Renderer initialization failed."
            return
        }

        self.renderer = renderer
        renderer.statisticsHandler = { [weak self] statistics in
            self?.updateStatisticsOverlay(with: statistics)
        }
        renderer.loadingStatusChanged = { [weak self] isLoading in
            DispatchQueue.main.async {
                guard let self else { return }
                self.setLoadingOverlay(visible: isLoading,
                                       message: self.pendingLoadingMessage ?? "Updating renderer...")
            }
        }
        renderer.mtkView(mtkView, drawableSizeWillChange: mtkView.drawableSize)
        mtkView.delegate = renderer
        configureControlsPanel(in: mtkView)
        setLoadingOverlay(visible: false)
    }
}
