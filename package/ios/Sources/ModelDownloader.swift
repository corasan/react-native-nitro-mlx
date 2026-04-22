import Foundation

enum ModelDownloadError: LocalizedError {
    case invalidModelMetadata(String)
    case invalidResponse(statusCode: Int, file: String?)
    case noDownloadableFiles(String)
    case manifestMissing(String)

    var errorDescription: String? {
        switch self {
        case .invalidModelMetadata(let modelId):
            return "Could not read downloadable files for model '\(modelId)'."
        case .invalidResponse(let statusCode, let file):
            if let file {
                return "Download failed for '\(file)' with HTTP status \(statusCode)."
            }
            return "Model request failed with HTTP status \(statusCode)."
        case .noDownloadableFiles(let modelId):
            return "No downloadable files were found for model '\(modelId)'."
        case .manifestMissing(let modelId):
            return "No download manifest was found for model '\(modelId)'."
        }
    }
}

private actor ProgressAggregator {
    private var writtenBytes: [String: Int64] = [:]
    private let baseBytes: Int64
    private let totalBytes: Int64
    private let onProgress: (Double) -> Void

    init(baseBytes: Int64, totalBytes: Int64, onProgress: @escaping (Double) -> Void) {
        self.baseBytes = baseBytes
        self.totalBytes = totalBytes
        self.onProgress = onProgress
    }

    func update(file: String, bytes: Int64) {
        writtenBytes[file] = bytes
        emit()
    }

    func emit() {
        guard totalBytes > 0 else { return }
        let total = writtenBytes.values.reduce(0, +) + baseBytes
        onProgress(min(1.0, Double(total) / Double(totalBytes)))
    }
}

private final class ProgressObservationBox: @unchecked Sendable {
    var observation: NSKeyValueObservation?
}

actor ModelDownloader: NSObject {
    static let shared = ModelDownloader()
    static var debug: Bool = false

    private let fileManager = FileManager.default
    private let manifestFileName = ".download-manifest.json"

    private func log(_ message: String) {
        if Self.debug {
            print("[Downloader] \(message)")
        }
    }

    private let downloadableExtensions: Set<String> = [
        "json", "safetensors", "txt", "model", "tiktoken", "py"
    ]

    private struct DownloadManifest: Codable {
        let modelId: String
        let files: [String]
        let completedAt: Date
    }

    private func fetchFileList(modelId: String) async throws -> [String] {
        let urlString = "https://huggingface.co/api/models/\(modelId)"
        guard let url = URL(string: urlString) else {
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }
        guard httpResponse.statusCode == 200 else {
            throw ModelDownloadError.invalidResponse(statusCode: httpResponse.statusCode, file: nil)
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = json["siblings"] as? [[String: Any]]
        else {
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }

        let files = siblings.compactMap { $0["rfilename"] as? String }
            .filter { name in
                let ext = (name as NSString).pathExtension.lowercased()
                return downloadableExtensions.contains(ext)
            }

        guard !files.isEmpty else {
            throw ModelDownloadError.noDownloadableFiles(modelId)
        }

        return files
    }

    private func manifestURL(for modelDir: URL) -> URL {
        modelDir.appendingPathComponent(manifestFileName)
    }

    private func readManifest(for modelDir: URL) -> DownloadManifest? {
        let url = manifestURL(for: modelDir)
        guard let data = try? Data(contentsOf: url) else {
            return nil
        }
        return try? JSONDecoder().decode(DownloadManifest.self, from: data)
    }

    private func writeManifest(modelId: String, files: [String], to modelDir: URL) throws {
        let manifest = DownloadManifest(
            modelId: modelId,
            files: files.sorted(),
            completedAt: Date()
        )
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL(for: modelDir), options: .atomic)
    }

    private func validateDownload(at modelDir: URL, manifest: DownloadManifest) -> Bool {
        manifest.files.allSatisfy { file in
            let fileURL = modelDir.appendingPathComponent(file)
            guard fileManager.fileExists(atPath: fileURL.path) else {
                return false
            }

            guard let attributes = try? fileManager.attributesOfItem(atPath: fileURL.path),
                  let size = attributes[.size] as? NSNumber else {
                return false
            }

            return size.int64Value > 0
        }
    }

    private func hasLegacyDownload(at modelDir: URL) -> Bool {
        let configURL = modelDir.appendingPathComponent("config.json")
        guard fileManager.fileExists(atPath: configURL.path) else {
            return false
        }

        guard let enumerator = fileManager.enumerator(at: modelDir, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return false
        }

        var hasWeights = false

        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            if ext == "safetensors" || ext == "model" {
                hasWeights = true
                break
            }
        }

        return hasWeights
    }

    private func isExistingFileValid(at url: URL) -> Bool {
        guard fileManager.fileExists(atPath: url.path),
              let attributes = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attributes[.size] as? NSNumber else {
            return false
        }

        return size.int64Value > 0
    }

    private struct PendingDownload: Sendable {
        let file: String
        let destURL: URL
        let url: URL
        var expectedBytes: Int64
    }

    private func fetchContentLength(url: URL) async -> Int64 {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse, httpResponse.expectedContentLength > 0 {
                return httpResponse.expectedContentLength
            }
        } catch {
            // best-effort; fall back to 0 if unavailable
        }
        return 0
    }

    func download(
        modelId: String,
        progressCallback: @escaping (Double) -> Void
    ) async throws -> URL {
        let files = try await fetchFileList(modelId: modelId)
        let modelDir = getModelDirectory(modelId: modelId)
        try fileManager.createDirectory(at: modelDir, withIntermediateDirectories: true)

        if let manifest = readManifest(for: modelDir),
           manifest.modelId == modelId,
           manifest.files.sorted() == files.sorted(),
           validateDownload(at: modelDir, manifest: manifest) {
            progressCallback(1.0)
            log("Using existing validated download for: \(modelId)")
            return modelDir
        }

        log("Model directory: \(modelDir.path)")
        log("Files to download: \(files)")

        var pending: [PendingDownload] = []
        var baseBytes: Int64 = 0

        for file in files {
            let destURL = modelDir.appendingPathComponent(file)
            try fileManager.createDirectory(at: destURL.deletingLastPathComponent(), withIntermediateDirectories: true)

            let urlString = "https://huggingface.co/\(modelId)/resolve/main/\(file)"
            guard let url = URL(string: urlString) else {
                log("Invalid URL: \(urlString)")
                continue
            }

            if isExistingFileValid(at: destURL) {
                log("File exists, skipping: \(file)")
                let size = (try? fileManager.attributesOfItem(atPath: destURL.path)[.size] as? NSNumber)?.int64Value ?? 0
                baseBytes += size
            } else {
                pending.append(PendingDownload(file: file, destURL: destURL, url: url, expectedBytes: 0))
            }
        }

        // Prefetch expected sizes in parallel so progress is byte-accurate.
        if !pending.isEmpty {
            await withTaskGroup(of: (Int, Int64).self) { group in
                for (index, p) in pending.enumerated() {
                    let url = p.url
                    group.addTask { (index, await self.fetchContentLength(url: url)) }
                }
                for await (index, bytes) in group {
                    pending[index].expectedBytes = bytes
                }
            }
        }

        let totalBytes = baseBytes + pending.map(\.expectedBytes).reduce(0, +)

        let aggregator = ProgressAggregator(
            baseBytes: baseBytes,
            totalBytes: totalBytes,
            onProgress: progressCallback
        )
        await aggregator.emit()

        // Download with bounded concurrency so progress streams smoothly for large files.
        let maxConcurrency = 4

        try await withThrowingTaskGroup(of: Void.self) { group in
            var iterator = pending.makeIterator()
            var inFlight = 0

            while inFlight < maxConcurrency, let next = iterator.next() {
                group.addTask { try await self.performDownload(next, modelId: modelId, aggregator: aggregator) }
                inFlight += 1
            }

            while try await group.next() != nil {
                if let next = iterator.next() {
                    group.addTask { try await self.performDownload(next, modelId: modelId, aggregator: aggregator) }
                }
            }
        }

        try writeManifest(modelId: modelId, files: files, to: modelDir)
        progressCallback(1.0)
        return modelDir
    }

    private func performDownload(
        _ pending: PendingDownload,
        modelId: String,
        aggregator: ProgressAggregator
    ) async throws {
        log("Downloading: \(pending.file)")

        let file = pending.file
        let observationBox = ProgressObservationBox()
        defer { observationBox.observation?.invalidate() }

        let (tempURL, response): (URL, URLResponse) = try await withCheckedThrowingContinuation { continuation in
            let task = URLSession.shared.downloadTask(with: pending.url) { tempURL, response, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let tempURL, let response else {
                    continuation.resume(throwing: URLError(.badServerResponse))
                    return
                }
                // The system deletes tempURL after this closure returns, so preserve it now.
                let preservedURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("mlx-download-\(UUID().uuidString)")
                do {
                    try FileManager.default.moveItem(at: tempURL, to: preservedURL)
                    continuation.resume(returning: (preservedURL, response))
                } catch {
                    continuation.resume(throwing: error)
                }
            }

            // Observe the task's real byte counter. task.progress uses 0-100 unit counts
            // (not bytes) for downloads; countOfBytesReceived is the actual byte count.
            observationBox.observation = task.observe(\.countOfBytesReceived, options: [.new]) { task, _ in
                let bytes = task.countOfBytesReceived
                Task { await aggregator.update(file: file, bytes: bytes) }
            }

            task.resume()
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            try? fileManager.removeItem(at: tempURL)
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }

        log("Response status: \(httpResponse.statusCode) for \(pending.file)")

        guard httpResponse.statusCode == 200 else {
            try? fileManager.removeItem(at: tempURL)
            throw ModelDownloadError.invalidResponse(statusCode: httpResponse.statusCode, file: pending.file)
        }

        if fileManager.fileExists(atPath: pending.destURL.path) {
            try fileManager.removeItem(at: pending.destURL)
        }
        try fileManager.moveItem(at: tempURL, to: pending.destURL)
        log("Saved: \(pending.file)")

        // Ensure final byte count matches expected (avoids drift if the delegate under-reports).
        if pending.expectedBytes > 0 {
            await aggregator.update(file: pending.file, bytes: pending.expectedBytes)
        }
    }

    func isDownloaded(modelId: String) -> Bool {
        let modelDir = getModelDirectory(modelId: modelId)

        let isComplete: Bool
        if let manifest = readManifest(for: modelDir), manifest.modelId == modelId {
            isComplete = validateDownload(at: modelDir, manifest: manifest)
        } else {
            isComplete = hasLegacyDownload(at: modelDir)
        }

        log("isDownloaded(\(modelId)): \(isComplete)")
        return isComplete
    }

    func getDownloadManifestContents(modelId: String) throws -> String {
        let modelDir = getModelDirectory(modelId: modelId)
        let manifestURL = manifestURL(for: modelDir)

        guard fileManager.fileExists(atPath: manifestURL.path) else {
            throw ModelDownloadError.manifestMissing(modelId)
        }

        let data = try Data(contentsOf: manifestURL)
        guard let contents = String(data: data, encoding: .utf8) else {
            throw ModelDownloadError.invalidModelMetadata(modelId)
        }

        return contents
    }

    func getModelDirectory(modelId: String) -> URL {
        let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docsDir
            .appendingPathComponent("huggingface/models")
            .appendingPathComponent(modelId.replacingOccurrences(of: "/", with: "_"))
    }

    func deleteModel(modelId: String) throws {
        let modelDir = getModelDirectory(modelId: modelId)
        if fileManager.fileExists(atPath: modelDir.path) {
            try fileManager.removeItem(at: modelDir)
        }
    }
}
