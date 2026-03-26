//
//  IOSViewController.swift
//  Metal3DGRT
//

import MetalKit
import UIKit

final class IOSViewController: UIViewController {
    private var renderer: Renderer?
    private var currentScene: GaussianScene?
    private let mtkView = MTKView(frame: .zero)

    private let statisticsLabel = UILabel()
    private let controlsContainer = UIVisualEffectView(effect: UIBlurEffect(style: .systemThinMaterialDark))
    private let statusLabel = UILabel()
    
    private let maxIntersectionLabel = UILabel()
    private let maxIntersectionSlider = UISlider()
    private let primitiveLabel = UILabel()
    private let primitiveButton = UIButton(type: .system)
    private let primitiveStack = UIStackView()
    private let modeLabel = UILabel()
    private let modeControl = UISegmentedControl(items: GaussianRayIntersectionMode.allCases.map(\.displayName))
    private let shDegreeLabel = UILabel()
    private let shDegreeControl = UISegmentedControl(items: ["0", "1", "2", "3"])
    private let sceneLabel = UILabel()
    private let sceneButton = UIButton(type: .system)
    private let sceneStack = UIStackView()
    private let strideLabel = UILabel()
    private let strideSlider = UISlider()
    private let rawHitsLabel = UILabel()
    private let rawHitsSwitch = UISwitch()
    private let loadingOverlay = UIView()
    private let loadingPanel = UIVisualEffectView(effect: UIBlurEffect(style: .systemChromeMaterialDark))
    private let loadingIndicator = UIActivityIndicatorView(style: .large)
    private let loadingLabel = UILabel()
    private var pendingLoadingMessage: String?

    override func viewDidLoad() {
        super.viewDidLoad()

        title = "Metal3DGRT iOS"
        view.backgroundColor = .black

        configureMetalView()
        configureOverlay()
        configureGestures()
        configureRenderer()
    }

    private func configureMetalView() {
        mtkView.translatesAutoresizingMaskIntoConstraints = false
        mtkView.enableSetNeedsDisplay = false
        mtkView.isPaused = false
        mtkView.autoResizeDrawable = true
        mtkView.preferredFramesPerSecond = 60
        mtkView.framebufferOnly = false
        mtkView.clearColor = MTLClearColor(red: 0.02, green: 0.03, blue: 0.05, alpha: 1.0)

        view.addSubview(mtkView)

        NSLayoutConstraint.activate([
            mtkView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            mtkView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            mtkView.topAnchor.constraint(equalTo: view.topAnchor),
            mtkView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])

        configureLoadingOverlay()
    }

    private func configureOverlay() {
        statisticsLabel.translatesAutoresizingMaskIntoConstraints = false
        statisticsLabel.font = .monospacedSystemFont(ofSize: 12, weight: .medium)
        statisticsLabel.textColor = .white
        statisticsLabel.numberOfLines = 0
        statisticsLabel.text = "Preparing renderer..."

        controlsContainer.translatesAutoresizingMaskIntoConstraints = false
        controlsContainer.layer.cornerRadius = 16
        controlsContainer.clipsToBounds = true

        let titleLabel = UILabel()
        titleLabel.text = "Gaussian Preview"
        titleLabel.font = .preferredFont(forTextStyle: .headline)
        titleLabel.textColor = .white

        statusLabel.text = "Pan: orbit  Pinch: zoom"
        statusLabel.font = .preferredFont(forTextStyle: .footnote)
        statusLabel.textColor = UIColor.white.withAlphaComponent(0.8)
        statusLabel.numberOfLines = 0

        let resetButton = UIButton(type: .system)
        resetButton.configuration = .filled()
        resetButton.configuration?.title = "Reset Camera"
        resetButton.addTarget(self, action: #selector(resetCamera), for: .touchUpInside)

        primitiveLabel.text = "Primitive"
        primitiveLabel.font = .preferredFont(forTextStyle: .footnote)
        primitiveLabel.textColor = .white

        primitiveButton.configuration = .tinted()
        primitiveButton.showsMenuAsPrimaryAction = true
        primitiveButton.changesSelectionAsPrimaryAction = true

        primitiveStack.addArrangedSubview(primitiveLabel)
        primitiveStack.addArrangedSubview(primitiveButton)
        primitiveStack.axis = .vertical
        primitiveStack.spacing = 4

        sceneLabel.text = "Scene"
        sceneLabel.font = .preferredFont(forTextStyle: .footnote)
        sceneLabel.textColor = .white

        sceneButton.configuration = .tinted()
        sceneButton.showsMenuAsPrimaryAction = true
        sceneButton.changesSelectionAsPrimaryAction = true
        
        sceneStack.addArrangedSubview(sceneLabel)
        sceneStack.addArrangedSubview(sceneButton)
        sceneStack.axis = .vertical
        sceneStack.spacing = 4

        modeLabel.text = "Mode"
        modeLabel.font = .preferredFont(forTextStyle: .footnote)
        modeLabel.textColor = .white

        modeControl.addTarget(self, action: #selector(modeChanged(_:)), for: .valueChanged)

        let modeStack = UIStackView(arrangedSubviews: [modeLabel, modeControl])
        modeStack.axis = .vertical
        modeStack.spacing = 4

        maxIntersectionLabel.font = .preferredFont(forTextStyle: .footnote)
        maxIntersectionLabel.textColor = .white
        
        maxIntersectionSlider.minimumValue = 1
        maxIntersectionSlider.maximumValue = Float(MAX_INTERSECTION_COUNT_LIMIT)
        maxIntersectionSlider.addTarget(self, action: #selector(maxIntersectionChanged(_:)), for: .valueChanged)

        let maxIsectStack = UIStackView(arrangedSubviews: [maxIntersectionLabel, maxIntersectionSlider])
        maxIsectStack.axis = .vertical
        maxIsectStack.spacing = 4

        shDegreeLabel.text = "SH Degree"
        shDegreeLabel.font = .preferredFont(forTextStyle: .footnote)
        shDegreeLabel.textColor = .white
        
        shDegreeControl.addTarget(self, action: #selector(shDegreeChanged(_:)), for: .valueChanged)

        let shDegreeStack = UIStackView(arrangedSubviews: [shDegreeLabel, shDegreeControl])
        shDegreeStack.axis = .horizontal
        shDegreeStack.spacing = 10
        shDegreeStack.alignment = .center

        strideLabel.font = .preferredFont(forTextStyle: .footnote)
        strideLabel.textColor = .white

        strideSlider.minimumValue = 1
        strideSlider.maximumValue = Float(MAX_STRIDE_SIZE_LIMIT)
        strideSlider.addTarget(self, action: #selector(strideChanged(_:)), for: .valueChanged)

        let strideStack = UIStackView(arrangedSubviews: [strideLabel, strideSlider])
        strideStack.axis = .vertical
        strideStack.spacing = 4

        rawHitsLabel.text = "Raw Polygon Hits"
        rawHitsLabel.font = .preferredFont(forTextStyle: .footnote)
        rawHitsLabel.textColor = .white
        
        rawHitsSwitch.addTarget(self, action: #selector(rawHitsChanged(_:)), for: .valueChanged)

        let rawHitsStack = UIStackView(arrangedSubviews: [rawHitsLabel, rawHitsSwitch])
        rawHitsStack.axis = .horizontal
        rawHitsStack.spacing = 10
        rawHitsStack.alignment = .center

        let stack = UIStackView(arrangedSubviews: [titleLabel, statusLabel, sceneStack, modeStack, primitiveStack, maxIsectStack, shDegreeStack, strideStack, rawHitsStack, resetButton])
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.axis = .vertical
        stack.spacing = 10
        stack.alignment = .fill

        view.addSubview(statisticsLabel)
        view.addSubview(controlsContainer)
        controlsContainer.contentView.addSubview(stack)

        NSLayoutConstraint.activate([
            statisticsLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            statisticsLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -16),
            statisticsLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 120),

            controlsContainer.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 16),
            controlsContainer.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -16),
            controlsContainer.widthAnchor.constraint(lessThanOrEqualToConstant: 260),

            stack.topAnchor.constraint(equalTo: controlsContainer.contentView.topAnchor, constant: 14),
            stack.leadingAnchor.constraint(equalTo: controlsContainer.contentView.leadingAnchor, constant: 14),
            stack.trailingAnchor.constraint(equalTo: controlsContainer.contentView.trailingAnchor, constant: -14),
            stack.bottomAnchor.constraint(equalTo: controlsContainer.contentView.bottomAnchor, constant: -14),
        ])
    }

    private func configureLoadingOverlay() {
        loadingOverlay.translatesAutoresizingMaskIntoConstraints = false
        loadingOverlay.backgroundColor = UIColor.black.withAlphaComponent(0.45)
        loadingOverlay.isHidden = true

        loadingPanel.translatesAutoresizingMaskIntoConstraints = false
        loadingPanel.layer.cornerRadius = 18
        loadingPanel.clipsToBounds = true

        loadingIndicator.translatesAutoresizingMaskIntoConstraints = false
        loadingIndicator.hidesWhenStopped = false
        loadingIndicator.color = .white
        loadingIndicator.startAnimating()

        loadingLabel.translatesAutoresizingMaskIntoConstraints = false
        loadingLabel.font = .preferredFont(forTextStyle: .headline)
        loadingLabel.textColor = .white
        loadingLabel.numberOfLines = 2
        loadingLabel.textAlignment = .center
        loadingLabel.text = "Loading..."

        let stack = UIStackView(arrangedSubviews: [loadingIndicator, loadingLabel])
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.axis = .vertical
        stack.spacing = 12
        stack.alignment = .center

        loadingPanel.contentView.addSubview(stack)
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

            stack.topAnchor.constraint(equalTo: loadingPanel.contentView.topAnchor, constant: 20),
            stack.leadingAnchor.constraint(equalTo: loadingPanel.contentView.leadingAnchor, constant: 24),
            stack.trailingAnchor.constraint(equalTo: loadingPanel.contentView.trailingAnchor, constant: -24),
            stack.bottomAnchor.constraint(equalTo: loadingPanel.contentView.bottomAnchor, constant: -20),
        ])
    }

    private func setLoadingOverlay(visible: Bool, message: String? = nil) {
        if let message {
            pendingLoadingMessage = message
        }
        loadingLabel.text = pendingLoadingMessage ?? "Loading..."
        loadingOverlay.isHidden = !visible
        if !visible {
            pendingLoadingMessage = nil
        }
    }

    private func configureGestures() {
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        mtkView.addGestureRecognizer(panGesture)
        mtkView.addGestureRecognizer(pinchGesture)
    }

    private func configureRenderer() {
        initializeControls()
        loadScene(GaussianScene.defaultScene)
    }

    private func loadScene(_ scene: GaussianScene) {
        setLoadingOverlay(visible: true, message: "Loading \(scene.displayName)...")
        statisticsLabel.text = "Loading \(scene.displayName)..."
        GaussianSceneLoader.loadScene(scene) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case let .success(gaussians):
                    self.currentScene = scene
                    self.installRenderer(with: gaussians)
                    self.initializeControls()
                case let .failure(error):
                    self.setLoadingOverlay(visible: false)
                    self.statisticsLabel.text = "Failed to load \(scene.displayName)."
                    self.statusLabel.text = error.localizedDescription
                }
            }
        }
    }

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let renderer else { return }
        let translation = gesture.translation(in: mtkView)
        renderer.orbitCamera(deltaX: Float(translation.x), deltaY: Float(translation.y))
        gesture.setTranslation(.zero, in: mtkView)
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard let renderer else { return }
        let magnification = Float(gesture.scale - 1.0)
        renderer.zoomCamera(magnification: magnification)
        gesture.scale = 1.0
    }

    @objc private func resetCamera() {
        renderer?.resetCamera()
    }

    @objc private func maxIntersectionChanged(_ sender: UISlider) {
        let value = Int(sender.value)
        renderer?.maxIntersectionCount = value
        maxIntersectionLabel.text = "Max Intersections: \(value)"
    }

    @objc private func shDegreeChanged(_ sender: UISegmentedControl) {
        renderer?.shDegree = sender.selectedSegmentIndex
    }

    @objc private func strideChanged(_ sender: UISlider) {
        let value = Int(sender.value)
        renderer?.strideSize = value
        strideLabel.text = "Stride: \(value)"
    }

    @objc private func rawHitsChanged(_ sender: UISwitch) {
        renderer?.visualizeRawPolygonHit = sender.isOn
    }

    @objc private func modeChanged(_ sender: UISegmentedControl) {
        guard sender.selectedSegmentIndex >= 0,
              sender.selectedSegmentIndex < GaussianRayIntersectionMode.allCases.count else {
            return
        }
        renderer?.selectedIntersectionMode = GaussianRayIntersectionMode.allCases[sender.selectedSegmentIndex]
        updatePrimitiveVisibility(for: GaussianRayIntersectionMode.allCases[sender.selectedSegmentIndex])
    }

    private func updatePrimitiveMenu(selected: GaussianEnclosingPrimitive) {
        primitiveButton.configuration?.title = selected.displayName
        primitiveButton.menu = UIMenu(children: GaussianEnclosingPrimitive.allCases.map { primitive in
            UIAction(title: primitive.displayName,
                     state: primitive == selected ? .on : .off) { [weak self] _ in
                self?.renderer?.selectedPrimitive = primitive
                self?.updatePrimitiveMenu(selected: primitive)
            }
        })
    }

    private func updateSceneMenu() {
        let availableScenes = GaussianScene.allCases.filter { $0 != .debug }
        sceneButton.menu = UIMenu(children: availableScenes.map { scene in
            UIAction(title: scene.displayName,
                     state: self.currentScene == scene ? .on : .off) { [weak self] _ in
                self?.loadScene(scene)
            }
        })
    }

    private func updatePrimitiveVisibility(for mode: GaussianRayIntersectionMode) {
        primitiveStack.isHidden = (mode != .triangle)
    }

    private func initializeControls() {
        maxIntersectionSlider.value = Float(renderer?.maxIntersectionCount ?? 3)
        maxIntersectionLabel.text = "Max Intersections: \(renderer?.maxIntersectionCount ?? 3)"
        updatePrimitiveMenu(selected: renderer?.selectedPrimitive ?? .octahedron)
        updateSceneMenu()
        let selectedMode = renderer?.selectedIntersectionMode ?? .boundingBox
        modeControl.selectedSegmentIndex = GaussianRayIntersectionMode.allCases.firstIndex(of: selectedMode) ?? 0
        updatePrimitiveVisibility(for: selectedMode)
        shDegreeControl.selectedSegmentIndex = renderer?.shDegree ?? 3
        strideSlider.value = Float(renderer?.strideSize ?? 3)
        strideLabel.text = "Stride: \(renderer?.strideSize ?? 3)"
        rawHitsSwitch.isOn = renderer?.visualizeRawPolygonHit ?? false
    }

    private func updateStatisticsOverlay(with statistics: RendererStatistics) {
        let progressLine = currentScene?.progress.snapshot.statusLine ?? "Stride progress unavailable"
        statisticsLabel.text = """
        FPS \(String(format: "%.1f", statistics.fps)) | \(String(format: "%.2f", statistics.frameTimeMilliseconds)) ms
        Gaussian RT \(String(format: "%.2f", statistics.raytracingTimeMilliseconds)) ms
        \(statistics.gaussianCount) gaussians | \(statistics.primitiveName)
        \(progressLine)
        Camera distance \(String(format: "%.2f", statistics.cameraDistance))
        \(Int(statistics.drawableSize.width))x\(Int(statistics.drawableSize.height))
        """
    }

    private func installRenderer(with gaussians: [GaussianSplat]) {
        if let renderer {
            renderer.progressDelegate = currentScene?.progress
            renderer.replaceGaussians(gaussians)
            initializeControls()
            return
        }

        guard let renderer = Renderer(metalView: mtkView,
                                      gaussians: gaussians,
                                      progressDelegate: currentScene?.progress) else {
            setLoadingOverlay(visible: false)
            statisticsLabel.text = "Renderer initialization failed."
            statusLabel.text = "This device may not support the required Metal ray tracing features."
            return
        }

        self.renderer = renderer
        initializeControls()

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
        setLoadingOverlay(visible: false)
    }
}
