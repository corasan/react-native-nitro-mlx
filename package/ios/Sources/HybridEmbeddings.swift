import Foundation
import NitroModules
internal import MLX
internal import MLXEmbedders
internal import MLXLMCommon
internal import Tokenizers

enum EmbeddingsError: Error {
  case notLoaded
  case emptyInput
}

class HybridEmbeddings: HybridEmbeddingsSpec {
  private var container: EmbedderModelContainer?
  private var loadTask: Task<Void, Error>?
  private var currentTask: Task<Any, Error>?
  private let tokenizerLoader: any TokenizerLoader = EmbeddingsTokenizerLoader()

  private var cachedDimension: Int = 0
  private var cachedMaxSeqLen: Int = 0

  var isLoaded: Bool { container != nil }
  var dimension: Double { Double(cachedDimension) }
  var maxSequenceLength: Double { Double(cachedMaxSeqLen) }

  private func readConfigInt(at dir: URL, keys: [String]) -> Int? {
    let url = dir.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: url),
      let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      return nil
    }
    for key in keys {
      if let v = obj[key] as? Int { return v }
      if let v = obj[key] as? NSNumber { return v.intValue }
    }
    return nil
  }

  private func floatsToArrayBuffer(_ floats: [Float]) -> ArrayBuffer {
    let byteSize = floats.count * MemoryLayout<Float>.size
    let buffer = ArrayBuffer.allocate(size: byteSize)
    floats.withUnsafeBytes { srcPtr in
      if let base = srcPtr.baseAddress {
        UnsafeMutableRawPointer(buffer.data).copyMemory(
          from: base,
          byteCount: byteSize
        )
      }
    }
    return buffer
  }

  func load(modelId: String, options: EmbeddingsLoadOptions?) throws -> Promise<Void> {
    loadTask?.cancel()

    return Promise.async { [self] in
      let task = Task { @MainActor in
        currentTask?.cancel()
        currentTask = nil
        container = nil
        cachedDimension = 0
        cachedMaxSeqLen = 0
        MLX.Memory.clearCache()

        if !(await ModelDownloader.shared.isDownloaded(modelId: modelId)) {
          _ = try await ModelDownloader.shared.download(
            modelId: modelId,
            progressCallback: { progress in
              options?.onProgress?(progress)
            }
          )
        }

        let modelDir = await ModelDownloader.shared.getModelDirectory(modelId: modelId)
        let loadedContainer = try await EmbedderModelFactory.shared.loadContainer(
          from: modelDir,
          using: tokenizerLoader
        )

        try Task.checkCancellation()

        self.container = loadedContainer
        cachedDimension = readConfigInt(at: modelDir, keys: ["hidden_size", "dim"]) ?? 0
        cachedMaxSeqLen =
          readConfigInt(at: modelDir, keys: ["max_position_embeddings"]) ?? 0
        options?.onProgress?(1.0)
      }

      loadTask = task
      try await task.value
    }
  }

  private func computeEmbeddings(texts: [String]) async throws -> [[Float]] {
    guard let container else { throw EmbeddingsError.notLoaded }

    return await container.perform { context in
      let tokenizer = context.tokenizer
      let model = context.model
      let pooling = context.pooling

      let inputs = texts.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }
      // Pad to longest, min 16 for Apple Silicon alignment (matches MLXEmbedders README).
      let maxLength = inputs.reduce(into: 16) { acc, elem in
        acc = max(acc, elem.count)
      }

      let padToken = tokenizer.eosTokenId ?? 0
      let padded = stacked(
        inputs.map { elem in
          MLXArray(
            elem + Array(repeating: padToken, count: maxLength - elem.count))
        })
      let mask = (padded .!= padToken)
      let tokenTypes = MLXArray.zeros(like: padded)
      let result = pooling(
        model(
          padded,
          positionIds: nil,
          tokenTypeIds: tokenTypes,
          attentionMask: mask),
        mask: mask,
        normalize: true
      )
      result.eval()
      return result.map { $0.asArray(Float.self) }
    }
  }

  func embed(text: String) throws -> Promise<ArrayBuffer> {
    guard container != nil else { throw EmbeddingsError.notLoaded }
    guard !text.isEmpty else { throw EmbeddingsError.emptyInput }

    return Promise.async { [self] in
      let task = Task<Any, Error> {
        let vectors = try await computeEmbeddings(texts: [text])
        guard let vec = vectors.first else { throw EmbeddingsError.emptyInput }
        return floatsToArrayBuffer(vec) as Any
      }

      currentTask = task
      defer { currentTask = nil }

      return try await task.value as! ArrayBuffer
    }
  }

  func embedBatch(texts: [String]) throws -> Promise<[ArrayBuffer]> {
    guard container != nil else { throw EmbeddingsError.notLoaded }
    guard !texts.isEmpty else { throw EmbeddingsError.emptyInput }

    return Promise.async { [self] in
      let task = Task<Any, Error> {
        let vectors = try await computeEmbeddings(texts: texts)
        return vectors.map { floatsToArrayBuffer($0) } as Any
      }

      currentTask = task
      defer { currentTask = nil }

      return try await task.value as! [ArrayBuffer]
    }
  }

  func unload() throws {
    loadTask?.cancel()
    loadTask = nil
    currentTask?.cancel()
    currentTask = nil
    container = nil
    cachedDimension = 0
    cachedMaxSeqLen = 0
    MLX.Memory.clearCache()
  }
}

/// Loads a Hugging Face tokenizer from a local directory and bridges it to
/// `MLXLMCommon.Tokenizer`. The mlx-swift-lm 3.x API requires an explicit
/// `TokenizerLoader`; this mirrors the expansion of `#huggingFaceTokenizerLoader()`.
private struct EmbeddingsTokenizerLoader: TokenizerLoader {
  func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
    let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
    return EmbeddingsTokenizerBridge(upstream)
  }
}

private struct EmbeddingsTokenizerBridge: MLXLMCommon.Tokenizer {
  private let upstream: any Tokenizers.Tokenizer

  init(_ upstream: any Tokenizers.Tokenizer) {
    self.upstream = upstream
  }

  func encode(text: String, addSpecialTokens: Bool) -> [Int] {
    upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
  }

  // swift-transformers uses `decode(tokens:)` instead of `decode(tokenIds:)`.
  func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
    upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
  }

  func convertTokenToId(_ token: String) -> Int? {
    upstream.convertTokenToId(token)
  }

  func convertIdToToken(_ id: Int) -> String? {
    upstream.convertIdToToken(id)
  }

  var bosToken: String? { upstream.bosToken }
  var eosToken: String? { upstream.eosToken }
  var unknownToken: String? { upstream.unknownToken }

  func applyChatTemplate(
    messages: [[String: any Sendable]],
    tools: [[String: any Sendable]]?,
    additionalContext: [String: any Sendable]?
  ) throws -> [Int] {
    do {
      return try upstream.applyChatTemplate(
        messages: messages, tools: tools, additionalContext: additionalContext)
    } catch Tokenizers.TokenizerError.missingChatTemplate {
      throw MLXLMCommon.TokenizerError.missingChatTemplate
    }
  }
}
