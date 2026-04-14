import Foundation
import NitroModules
internal import MLXLMCommon
internal import MLXLLM

class HybridModelManager: HybridModelManagerSpec {
    private let fileManager = FileManager.default

    var debug: Bool {
        get { ModelDownloader.debug }
        set { ModelDownloader.debug = newValue }
    }

    private func log(_ message: String) {
        if debug {
            print("[MLXReactNative.HybridModelManager] \(message)")
        }
    }

    func download(
        modelId: String,
        progressCallback: @escaping (Double) -> Void
    ) throws -> Promise<String> {
        return Promise.async { [self] in
            log("Starting download for: \(modelId)")

            let modelDir = try await ModelDownloader.shared.download(
                modelId: modelId,
                progressCallback: progressCallback
            )

            log("Download complete: \(modelDir.path)")
            return modelDir.path
        }
    }

    func isDownloaded(modelId: String) throws -> Promise<Bool> {
        return Promise.async {
            return await ModelDownloader.shared.isDownloaded(modelId: modelId)
        }
    }

    func getDownloadedModels() throws -> Promise<[String]> {
        return Promise.async { [self] in
            let docsDir = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
            let modelsDir = docsDir.appendingPathComponent("huggingface/models")

            guard fileManager.fileExists(atPath: modelsDir.path) else {
                return []
            }

            let contents = try fileManager.contentsOfDirectory(
                at: modelsDir,
                includingPropertiesForKeys: [.isDirectoryKey]
            )

            let modelIds = contents
                .filter { url in
                    var isDir: ObjCBool = false
                    return fileManager.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
                }
                .map { $0.lastPathComponent.replacingOccurrences(of: "_", with: "/") }

            var downloadedModels: [String] = []
            for modelId in modelIds {
                if await ModelDownloader.shared.isDownloaded(modelId: modelId) {
                    downloadedModels.append(modelId)
                }
            }

            return downloadedModels
        }
    }

    func deleteModel(modelId: String) throws -> Promise<Void> {
        return Promise.async {
            try await ModelDownloader.shared.deleteModel(modelId: modelId)
        }
    }

    func getModelPath(modelId: String) throws -> Promise<String> {
        return Promise.async {
            return await ModelDownloader.shared.getModelDirectory(modelId: modelId).path
        }
    }
}
