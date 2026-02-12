import { LegendList, type LegendListRef } from '@legendapp/list'
import * as Crypto from 'expo-crypto'
import { router, useFocusEffect } from 'expo-router'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ActivityIndicator,
  LayoutAnimation,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  useColorScheme,
  View,
} from 'react-native'
import { KeyboardAvoidingView } from 'react-native-keyboard-controller'
import { createTool, LLM, MLXModel, ModelManager, type StreamEvent } from 'react-native-nitro-mlx'
import { SafeAreaView } from 'react-native-safe-area-context'
import { z } from 'zod'
import { useBenchmark } from '../../components/benchmark-context'

const MODEL_ID = MLXModel.Qwen3_1_7B_4bit

const WEATHER_API_KEY = process.env.EXPO_PUBLIC_WEATHER_API_KEY
const BASE_URL = 'https://api.openweathermap.org/data/2.5/weather?units=imperial'

const weatherTool = createTool({
  name: 'weather_tool',
  description:
    'Get current weather for a SINGLE city. Call this tool once per city. If comparing multiple cities, make separate calls for each.',
  arguments: z.object({
    city: z.string().describe('A single city name'),
  }),
  handler: async args => {
    try {
      const url = `${BASE_URL}&q=${args.city}&APPID=${WEATHER_API_KEY}`
      const res = await fetch(url, { method: 'GET' })
      const result = await res.json()

      if (!result.main) {
        console.error('Invalid API response:', result)
        return {
          temperature: 0,
          humidity: 0,
          precipitation: 'Unknown',
          units: 'imperial',
        }
      }

      return {
        temperature: result.main.temp,
        humidity: result.main.humidity || 0,
        precipitation: result.weather?.[0]?.description || 'Unknown',
        units: 'imperial',
      }
    } catch (error) {
      console.error('Weather tool error:', error)
      return { temperature: 0, humidity: 0, precipitation: 'Unknown', units: 'imperial' }
    }
  },
})

type ThinkingBlockData = {
  type: 'thinking'
  content: string
}

type ToolCallBlockData = {
  type: 'tool_call'
  id?: string
  name: string
  args: Record<string, unknown>
  completed?: boolean
}

type MessageBlock = ThinkingBlockData | ToolCallBlockData

function parseThinkingBlocks(text: string): { thinking: string; content: string } {
  const thinkRegex = /<think>([\s\S]*?)<\/think>/g
  const thinkingParts: string[] = []
  let content = text

  let match: RegExpExecArray | null = thinkRegex.exec(text)
  while (match !== null) {
    thinkingParts.push(match[1].trim())
    match = thinkRegex.exec(text)
  }

  content = text.replace(/<think>[\s\S]*?<\/think>/g, '').trim()

  return {
    thinking: thinkingParts.join('\n\n'),
    content,
  }
}

type Message = {
  id: string
  content: string
  blocks?: MessageBlock[]
  currentThinking?: string
  isCurrentlyThinking?: boolean
  isUser: boolean
}

const ToolCallBlock = ({ toolCall }: { toolCall: ToolCallBlockData }) => {
  const [expanded, setExpanded] = useState(false)
  const colorScheme = useColorScheme()

  const toggleExpanded = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut)
    setExpanded(!expanded)
  }

  const toolDisplayName = toolCall.name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())

  return (
    <TouchableOpacity onPress={toggleExpanded} style={styles.toolCallBlock}>
      <View style={styles.toolCallHeader}>
        <Text style={styles.toolCallIcon}>🔧</Text>
        <Text style={styles.toolCallLabel}>
          {toolCall.completed ? 'Used' : 'Using'} {toolDisplayName}
        </Text>
        {toolCall.completed ? (
          <Text style={styles.toolCallComplete}>✓</Text>
        ) : (
          <ActivityIndicator
            size="small"
            color="#007AFF"
            style={styles.toolCallSpinner}
          />
        )}
      </View>
      {expanded && (
        <Text
          style={[
            styles.toolCallArgs,
            { color: colorScheme === 'dark' ? '#aaa' : '#666' },
          ]}
        >
          {JSON.stringify(toolCall.args, null, 2)}
        </Text>
      )}
    </TouchableOpacity>
  )
}

const ThinkingBlock = ({ thinking, isStreaming }: { thinking: string; isStreaming?: boolean }) => {
  const [expanded, setExpanded] = useState(false)
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? '#aaa' : '#666'

  const toggleExpanded = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut)
    setExpanded(!expanded)
  }

  return (
    <TouchableOpacity onPress={toggleExpanded} style={styles.thinkingBlock}>
      <View style={styles.thinkingHeader}>
        <Text style={[styles.thinkingLabel, { color: textColor }]}>
          {expanded ? '▼' : '▶'} Thinking
        </Text>
        {isStreaming && <ActivityIndicator size="small" color="#888" style={{ marginLeft: 8 }} />}
      </View>
      {expanded && (
        <Text style={[styles.thinkingText, { color: textColor }]}>{thinking}</Text>
      )}
    </TouchableOpacity>
  )
}

const MessageItem = ({ content, blocks, currentThinking, isCurrentlyThinking, isUser }: Message) => {
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'

  if (isUser) {
    return (
      <View style={styles.userMessage}>
        <Text style={[styles.messageText, { color: 'white' }]}>{content}</Text>
      </View>
    )
  }

  return (
    <View style={styles.message}>
      {blocks?.map((block, index) =>
        block.type === 'thinking' ? (
          <ThinkingBlock key={`block-${index}`} thinking={block.content} />
        ) : (
          <ToolCallBlock key={`block-${index}`} toolCall={block} />
        ),
      )}
      {isCurrentlyThinking && (
        <ThinkingBlock thinking={currentThinking || 'Processing...'} isStreaming />
      )}
      {content?.trim() ? (
        <Text style={[styles.messageText, { color: textColor }]}>{content.trim()}</Text>
      ) : null}
    </View>
  )
}

export default function ChatScreen() {
  const [isChecking, setIsChecking] = useState(true)
  const [isDownloaded, setIsDownloaded] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [loadProgress, setLoadProgress] = useState(0)
  const [isReady, setIsReady] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const colorScheme = useColorScheme()
  const textColor = colorScheme === 'dark' ? 'white' : 'black'
  const bgColor = colorScheme === 'dark' ? 'black' : 'white'
  const [messages, setMessages] = useState<Message[]>([])
  const listRef = useRef<LegendListRef>(null)
  const inputRef = useRef<TextInput>(null)
  const isLoadingRef = useRef(false)
  const { addResult } = useBenchmark()

  LLM.debug = true

  // useEffect(() => {
  //   return () => {
  //     LLM.unload()
  //   }
  // }, [])

  const openSettings = () => {
    router.push('/settings-modal')
  }

  const checkDownloaded = useCallback(async () => {
    setIsChecking(true)
    try {
      const downloaded = await ModelManager.isDownloaded(MODEL_ID)
      setIsDownloaded(downloaded)
    } catch (error) {
      console.error('Error checking download:', error)
    } finally {
      setIsChecking(false)
    }
  }, [])

  useFocusEffect(
    useCallback(() => {
      checkDownloaded()
    }, [checkDownloaded]),
  )

  useEffect(() => {
    if (!isDownloaded || isReady || isLoadingRef.current) return

    const loadModel = async () => {
      isLoadingRef.current = true
      setIsLoading(true)
      setLoadProgress(0)
      try {
        LLM.systemPrompt =
          'You are a helpful assistant. When users ask about weather, use the weather_tool to get current information. IMPORTANT: If asked about multiple cities, you MUST call weather_tool separately for each city - never assume they have the same weather.'
        await LLM.load(MODEL_ID, {
          onProgress: setLoadProgress,
          manageHistory: true,
          tools: [weatherTool],
        })
        setIsReady(true)
      } catch (error) {
        console.error('Error loading model:', error)
      } finally {
        setIsLoading(false)
        isLoadingRef.current = false
      }
    }

    loadModel()
  }, [isDownloaded, isReady])

  const sendPrompt = async () => {
    if (!isReady || !prompt.trim() || isGenerating) return

    const currentPrompt = prompt
    const assistantMessageId = Crypto.randomUUID()
    const tempAssistantMessage: Message = {
      id: assistantMessageId,
      content: '',
      blocks: [],
      isUser: false,
    }

    setMessages(prev => [
      ...prev,
      { id: Crypto.randomUUID(), content: currentPrompt, isUser: true },
      tempAssistantMessage,
    ])
    setPrompt('')
    inputRef.current?.blur()
    setIsGenerating(true)

    let content = ''
    let currentThinking = ''

    const handleEvent = (event: StreamEvent) => {
      switch (event.type) {
        case 'thinking_start':
          currentThinking = ''
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId
                ? { ...msg, currentThinking: '', isCurrentlyThinking: true }
                : msg,
            ),
          )
          break

        case 'thinking_chunk':
          currentThinking += event.chunk
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId ? { ...msg, currentThinking } : msg,
            ),
          )
          break

        case 'thinking_end':
          setMessages(prev =>
            prev.map(msg => {
              if (msg.id !== assistantMessageId) return msg
              const thinkingBlock: ThinkingBlockData = { type: 'thinking', content: event.content }
              return {
                ...msg,
                blocks: [...(msg.blocks || []), thinkingBlock],
                currentThinking: undefined,
                isCurrentlyThinking: false,
              }
            }),
          )
          break

        case 'token':
          content += event.token
          setMessages(prev =>
            prev.map(msg =>
              msg.id === assistantMessageId ? { ...msg, content } : msg,
            ),
          )
          break

        case 'tool_call_start':
          setMessages(prev =>
            prev.map(msg => {
              if (msg.id !== assistantMessageId) return msg
              const toolBlock: ToolCallBlockData = {
                type: 'tool_call',
                id: event.id,
                name: event.name,
                args: JSON.parse(event.arguments),
                completed: false,
              }
              return {
                ...msg,
                blocks: [...(msg.blocks || []), toolBlock],
              }
            }),
          )
          break

        case 'tool_call_completed':
        case 'tool_call_failed':
          setMessages(prev =>
            prev.map(msg => {
              if (msg.id !== assistantMessageId || !msg.blocks) return msg
              return {
                ...msg,
                blocks: msg.blocks.map(block =>
                  block.type === 'tool_call' && block.id === event.id
                    ? { ...block, completed: true }
                    : block,
                ),
              }
            }),
          )
          break

        case 'generation_end':
          addResult({
            tokensPerSecond: event.stats.tokensPerSecond,
            timeToFirstToken: event.stats.timeToFirstToken,
            totalTokens: event.stats.tokenCount,
            totalTime: event.stats.totalTime,
            toolExecutionTime: event.stats.toolExecutionTime,
            timestamp: new Date(),
          })
          break
      }
    }

    try {
      await LLM.streamWithEvents(currentPrompt, handleEvent)
    } catch (error) {
      console.error('Error generating:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  const openDownloadModal = () => {
    router.push('/download-modal')
  }

  const deleteModel = async () => {
    try {
      LLM.unload()
      await ModelManager.deleteModel(MODEL_ID)
      setIsDownloaded(false)
      setIsReady(false)
      setMessages([])
      isLoadingRef.current = false
    } catch (error) {
      console.error('Error deleting model:', error)
    }
  }

  const syncFromHistory = useCallback((preserveToolCalls = false) => {
    try {
      const history = LLM.getHistory()
      setMessages(prev => {
        const blocksMap = new Map<number, MessageBlock[]>()
        if (preserveToolCalls) {
          prev.forEach((msg, idx) => {
            if (msg.blocks) {
              blocksMap.set(idx, msg.blocks)
            }
          })
        }

        return history.map((msg, index) => {
          if (msg.role === 'user') {
            return {
              id: `history-${index}`,
              content: msg.content,
              isUser: true,
            }
          }

          const { thinking, content } = parseThinkingBlocks(msg.content)
          const existingBlocks = blocksMap.get(index)
          const blocks: MessageBlock[] = existingBlocks || (thinking ? [{ type: 'thinking', content: thinking }] : [])

          return {
            id: `history-${index}`,
            content,
            blocks,
            isUser: false,
          }
        })
      })
    } catch (error) {
      console.error('Error syncing from history:', error)
    }
  }, [])

  const logHistory = () => {
    try {
      const history = LLM.getHistory()
      console.log('Message History:', history)
      console.log('Total messages:', history.length)
    } catch (error) {
      console.error('Error getting history:', error)
    }
  }

  const handleClearHistory = () => {
    try {
      LLM.clearHistory()
      setMessages([])
      console.log('History cleared')
    } catch (error) {
      console.error('Error clearing history:', error)
    }
  }

  useEffect(() => {
    if (isReady) {
      syncFromHistory()
    }
  }, [isReady, syncFromHistory])

  if (isChecking) {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <ActivityIndicator size="large" />
        <Text style={[styles.statusText, { color: textColor }]}>Checking model...</Text>
      </SafeAreaView>
    )
  }

  if (!isDownloaded) {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <Text style={[styles.title, { color: textColor }]}>MLX Chat</Text>
        <Text style={[styles.subtitle, { color: textColor }]}>
          Download the model to get started
        </Text>
        <TouchableOpacity style={styles.downloadButton} onPress={openDownloadModal}>
          <Text style={styles.downloadButtonText}>Download Model</Text>
        </TouchableOpacity>
        <Text style={[styles.modelId, { color: textColor }]}>{MODEL_ID}</Text>
      </SafeAreaView>
    )
  }

  if (isLoading) {
    return (
      <SafeAreaView style={[styles.centered, { backgroundColor: bgColor }]}>
        <ActivityIndicator size="large" />
        <Text style={[styles.statusText, { color: textColor }]}>
          Loading model... {(loadProgress * 100).toFixed(0)}%
        </Text>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: bgColor }]}
      edges={['bottom', 'top']}
    >
      <KeyboardAvoidingView
        style={[styles.container, { backgroundColor: bgColor }]}
        behavior="padding"
        keyboardVerticalOffset={Platform.select({ ios: 0, default: 0 })}
      >
        <View
          style={[
            styles.header,
            { borderBottomColor: colorScheme === 'dark' ? '#333' : '#eee' },
          ]}
        >
          <TouchableOpacity onPress={openSettings}>
            <Text style={[styles.headerButton, { color: '#007AFF' }]}>Benchmark</Text>
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: textColor }]}>MLX Chat</Text>
          <View style={styles.headerButtons}>
            <TouchableOpacity style={styles.historyButton} onPress={logHistory}>
              <Text style={styles.historyButtonText}>Log</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.clearButton} onPress={handleClearHistory}>
              <Text style={styles.clearButtonText}>Clear</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.deleteButton} onPress={deleteModel}>
              <Text style={styles.deleteButtonText}>Delete</Text>
            </TouchableOpacity>
          </View>
        </View>

        <LegendList<Message>
          ref={listRef}
          data={messages}
          keyExtractor={item => item.id}
          estimatedItemSize={100}
          renderItem={({ item }) => <MessageItem key={item.id} {...item} />}
          alignItemsAtEnd
          maintainScrollAtEnd
          maintainVisibleContentPosition
        />

        <View style={styles.inputContainer}>
          <TextInput
            ref={inputRef}
            value={prompt}
            onChangeText={setPrompt}
            placeholder="Type a message..."
            placeholderTextColor="#999"
            style={[styles.input, { color: textColor }]}
            editable={!isGenerating}
            onSubmitEditing={sendPrompt}
            returnKeyType="send"
          />
          <TouchableOpacity
            style={[
              styles.sendButton,
              (!prompt.trim() || isGenerating) && styles.sendButtonDisabled,
            ]}
            onPress={sendPrompt}
            disabled={!prompt.trim() || isGenerating}
          >
            <Text style={styles.sendButtonText}>{isGenerating ? '...' : 'Send'}</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
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
  headerButton: {
    fontSize: 14,
    fontWeight: '500',
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 6,
  },
  historyButton: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: '#34C759',
  },
  historyButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  clearButton: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: '#FF9500',
  },
  clearButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  deleteButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: '#FF3B30',
  },
  deleteButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.7,
    marginBottom: 24,
    textAlign: 'center',
  },
  statusText: {
    marginTop: 12,
    fontSize: 16,
  },
  modelId: {
    marginTop: 16,
    fontSize: 12,
    opacity: 0.5,
  },
  downloadButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  downloadButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  message: {
    padding: 16,
    paddingHorizontal: 20,
  },
  userMessage: {
    padding: 12,
    paddingHorizontal: 16,
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end',
    borderRadius: 16,
    marginRight: 12,
    marginVertical: 4,
    maxWidth: '80%',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    gap: 8,
  },
  input: {
    flex: 1,
    backgroundColor: '#c4c4c62f',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 16,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  sendButtonDisabled: {
    opacity: 0.5,
  },
  sendButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  thinkingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  thinkingIndicatorText: {
    fontSize: 14,
    fontStyle: 'italic',
    opacity: 0.7,
  },
  toolCallBlock: {
    backgroundColor: '#007AFF15',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#007AFF',
  },
  toolCallHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  toolCallIcon: {
    fontSize: 14,
    marginRight: 6,
  },
  toolCallLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#007AFF',
    flex: 1,
  },
  toolCallSpinner: {
    marginLeft: 8,
  },
  toolCallComplete: {
    color: '#34C759',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  toolCallArgs: {
    fontSize: 11,
    fontFamily: 'Menlo',
    marginTop: 8,
  },
  thinkingBlock: {
    backgroundColor: '#8881',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
  },
  thinkingHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  thinkingLabel: {
    fontSize: 12,
    fontWeight: '600',
  },
  thinkingText: {
    fontSize: 13,
    lineHeight: 18,
    marginTop: 8,
    fontStyle: 'italic',
  },
})
