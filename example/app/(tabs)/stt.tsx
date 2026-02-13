import { useFocusEffect } from 'expo-router'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native'
import { MLXModel, STT } from 'react-native-nitro-mlx'
import { SafeAreaView } from 'react-native-safe-area-context'

const MODEL_ID = MLXModel.GLM_ASR_Nano_4bit

type Status = 'idle' | 'loading' | 'ready' | 'listening' | 'transcribing'

export default function STTScreen() {
  const [status, setStatus] = useState<Status>('idle')
  const [transcript, setTranscript] = useState('')
  const [streamingText, setStreamingText] = useState('')
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  const bgColor = colorScheme === 'dark' ? 'black' : 'white'
  const isLoadingRef = useRef(false)
  const streamingRef = useRef('')
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isTranscribingChunk = useRef(false)

  const loadModel = useCallback(async () => {
    if (STT.isLoaded) {
      setStatus('ready')
      return
    }
    if (isLoadingRef.current) return
    isLoadingRef.current = true
    setStatus('loading')
    try {
      await STT.load(MODEL_ID)
      setStatus('ready')
    } catch (error) {
      console.error('Error loading STT model:', error)
      setStatus('idle')
    } finally {
      isLoadingRef.current = false
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      loadModel()
    }, [loadModel]),
  )

  const transcribeChunk = useCallback(async () => {
    if (isTranscribingChunk.current) return
    isTranscribingChunk.current = true
    try {
      const text = await STT.transcribeBuffer()
      if (text) {
        streamingRef.current = text
        setStreamingText(text)
      }
    } catch {
      // buffer too small or not listening, skip
    } finally {
      isTranscribingChunk.current = false
    }
  }, [])

  const startPolling = useCallback(() => {
    intervalRef.current = setInterval(transcribeChunk, 1000)
  }, [transcribeChunk])

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  const handleToggleListening = async () => {
    if (status === 'listening') {
      try {
        stopPolling()
        setStatus('transcribing')
        const finalText = await STT.stopListening()
        setTranscript(finalText || streamingRef.current)
        setStreamingText('')
        streamingRef.current = ''
        setStatus('ready')
      } catch (error) {
        console.error('STT stopListening error:', error)
        setStatus('ready')
      }
    } else if (status === 'ready') {
      setTranscript('')
      setStreamingText('')
      streamingRef.current = ''
      try {
        await STT.startListening()
        setStatus('listening')
        startPolling()
      } catch (error) {
        console.error('STT startListening error:', error)
        setStatus('ready')
      }
    }
  }

  const handleClear = () => {
    setTranscript('')
    setStreamingText('')
    streamingRef.current = ''
  }

  const statusText: Record<Status, string> = {
    idle: '',
    loading: 'Downloading & loading model...',
    ready: 'Ready',
    listening: 'Listening...',
    transcribing: 'Transcribing...',
  }

  const displayText = streamingText || transcript

  if (status === 'idle' || status === 'loading') {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <ActivityIndicator size="large" />
        <Text style={[styles.statusLabel, { color: textColor }]}>
          {statusText[status] || 'Preparing...'}
        </Text>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: bgColor }]}
      edges={['bottom', 'top']}
    >
      <View
        style={[
          styles.header,
          {
            borderBottomColor: colorScheme === 'dark' ? '#333' : '#eee',
          },
        ]}
      >
        <Text style={[styles.headerTitle, { color: textColor }]}>MLX STT</Text>
        <Text style={[styles.statusBadge]}>{statusText[status]}</Text>
      </View>

      <View style={styles.content}>
        <ScrollView
          style={[styles.transcriptBox]}
          contentContainerStyle={styles.transcriptContent}
        >
          {displayText ? (
            <Text style={[styles.transcriptText, { color: textColor }]}>
              {displayText}
            </Text>
          ) : (
            <Text style={styles.placeholderText}>
              Tap the microphone to start listening...
            </Text>
          )}
        </ScrollView>

        <View style={styles.controls}>
          <TouchableOpacity
            style={[
              styles.micButton,
              status === 'listening' && styles.micButtonActive,
              status === 'transcribing' && styles.micButtonDisabled,
            ]}
            onPress={handleToggleListening}
            disabled={status === 'transcribing'}
          >
            {status === 'transcribing' ? (
              <ActivityIndicator color="white" />
            ) : (
              <Text style={styles.micButtonText}>
                {status === 'listening' ? '⏹' : '🎙'}
              </Text>
            )}
          </TouchableOpacity>

          {displayText.length > 0 && status === 'ready' && (
            <TouchableOpacity style={styles.clearButton} onPress={handleClear}>
              <Text style={styles.clearButtonText}>Clear</Text>
            </TouchableOpacity>
          )}
        </View>
      </View>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingBottom: 50,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    padding: 16,
    borderBottomWidth: 1,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  statusBadge: {
    fontSize: 13,
    color: '#007AFF',
    fontWeight: '500',
  },
  statusLabel: {
    marginTop: 12,
    fontSize: 16,
  },
  content: {
    flex: 1,
    padding: 20,
    gap: 16,
  },
  transcriptBox: {
    flex: 1,
    backgroundColor: '#c4c4c62f',
    borderRadius: 12,
    padding: 16,
  },
  transcriptContent: {
    flexGrow: 1,
  },
  transcriptText: {
    fontSize: 16,
    lineHeight: 24,
  },
  placeholderText: {
    fontSize: 16,
    color: '#999',
  },
  controls: {
    alignItems: 'center',
    gap: 12,
  },
  micButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
  },
  micButtonActive: {
    backgroundColor: '#FF3B30',
  },
  micButtonDisabled: {
    opacity: 0.5,
  },
  micButtonText: {
    fontSize: 28,
  },
  clearButton: {
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#c4c4c62f',
  },
  clearButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#999',
  },
})
