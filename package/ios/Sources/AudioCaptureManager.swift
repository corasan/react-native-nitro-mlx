import AVFoundation
import Foundation
internal import MLX

class AudioCaptureManager {
  private let audioEngine = AVAudioEngine()
  private var audioBuffer: [Float] = []
  private let bufferLock = NSLock()
  private let targetSampleRate: Double = 16000

  var isCapturing: Bool { audioEngine.isRunning }

  func startCapturing() async throws {
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.record, mode: .measurement)
    try session.setActive(true)

    let inputNode = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)
    let outputFormat = AVAudioFormat(
      commonFormat: .pcmFormatFloat32,
      sampleRate: targetSampleRate,
      channels: 1,
      interleaved: false
    )!

    guard
      let converter = AVAudioConverter(
        from: inputFormat, to: outputFormat)
    else {
      throw NSError(
        domain: "AudioCaptureManager",
        code: -1,
        userInfo: [
          NSLocalizedDescriptionKey:
            "Failed to create audio converter"
        ]
      )
    }

    bufferLock.lock()
    audioBuffer.removeAll()
    bufferLock.unlock()

    inputNode.installTap(
      onBus: 0, bufferSize: 4096, format: inputFormat
    ) { [weak self] buffer, _ in
      guard let self else { return }

      let frameCount = AVAudioFrameCount(
        targetSampleRate * Double(buffer.frameLength)
          / inputFormat.sampleRate
      )
      guard
        let convertedBuffer = AVAudioPCMBuffer(
          pcmFormat: outputFormat, frameCapacity: frameCount)
      else { return }

      var error: NSError?
      converter.convert(to: convertedBuffer, error: &error) {
        _, outStatus in
        outStatus.pointee = .haveData
        return buffer
      }

      if error == nil, let channelData = convertedBuffer.floatChannelData {
        let frames = Int(convertedBuffer.frameLength)
        self.bufferLock.lock()
        self.audioBuffer.append(
          contentsOf: UnsafeBufferPointer(
            start: channelData[0], count: frames))
        self.bufferLock.unlock()
      }
    }

    audioEngine.prepare()
    try audioEngine.start()
  }

  func snapshotAndClear() -> MLXArray? {
    bufferLock.lock()
    let samples = audioBuffer
    audioBuffer.removeAll()
    bufferLock.unlock()

    guard samples.count >= 8000 else { return nil }

    // Silence gate: skip chunks whose peak amplitude is near the noise
    // floor so the ASR model doesn't hallucinate ("The.", "...") on
    // silence. Peak-based because measurement-mode capture disables AGC,
    // making RMS of quiet speech close to ambient noise.
    var peak: Float = 0
    for s in samples {
      let a = s < 0 ? -s : s
      if a > peak { peak = a }
    }
    guard peak >= 0.005 else { return nil }

    return MLXArray(samples)
  }

  func snapshot() -> MLXArray? {
    // Take exclusive ownership of the accumulated buffer so the audio tap
    // gets fresh empty storage to append into; the expensive MLXArray copy
    // then happens off the audio path. Samples are merged back afterward
    // so the buffer keeps accumulating across calls.
    bufferLock.lock()
    var samples = audioBuffer
    audioBuffer.removeAll()
    bufferLock.unlock()

    defer {
      bufferLock.lock()
      samples.append(contentsOf: audioBuffer)
      audioBuffer = samples
      bufferLock.unlock()
    }

    guard samples.count >= 16000 else { return nil }
    return MLXArray(samples)
  }

  func stopCapturing() -> MLXArray {
    audioEngine.inputNode.removeTap(onBus: 0)
    audioEngine.stop()
    try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])

    bufferLock.lock()
    let samples = audioBuffer
    audioBuffer.removeAll()
    bufferLock.unlock()

    return MLXArray(samples)
  }
}
