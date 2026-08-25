import { z } from 'zod'
import { LLM } from './llm'
import type { AbortSignalLike } from './runtime'
import type { LLMGenerationConfig, StreamEvent } from './specs/LLM.nitro'
import type {
  LLMMessage,
  LLMToolCall,
  LLMTurnOutcome,
  LLMTurnRequest,
  ToolSchema,
} from './turn'

const ERROR_PREFIX = '[react-native-nitro-mlx]'

export interface ToolExecutorResult {
  content: string
  isError?: boolean
}

/**
 * Executes one tool call and returns its result. A plain string return is
 * shorthand for `{ content }`. Throwing is safe: the error message becomes an
 * `isError` tool message, because the model must see failures to recover.
 */
export type ToolExecutor = (
  call: LLMToolCall,
) => ToolExecutorResult | string | Promise<ToolExecutorResult | string>

export interface RunToolLoopOptions {
  /** Warm Turn Context id. Exclusive with the cold-turn fields below. */
  contextId?: string
  /** Cold-turn fields, forwarded to `runTurn` when `contextId` is absent. */
  instructions?: string
  history?: LLMMessage[]
  tools?: ToolSchema[]
  generationConfig?: LLMGenerationConfig
  /** Maximum number of generation turns before the loop stops. @default 6 */
  maxSteps?: number
  /** Cancels the loop; behaves like `LLM.runTurn`'s signal on the active turn. */
  signal?: AbortSignalLike
  /** Forwarded to every turn. */
  onEvent?: (event: StreamEvent) => void
  /** Called after each executed tool call with its normalized result. */
  onToolResult?: (call: LLMToolCall, result: ToolExecutorResult) => void
}

export interface ToolLoopResult {
  /** The terminal turn: a plain answer, or the last turn when the loop stopped early. */
  outcome: LLMTurnOutcome
  /** Number of generation turns that ran. */
  steps: number
  /** True when the loop hit `maxSteps` with tool calls still pending. */
  stoppedAtMaxSteps: boolean
}

const stringResultSchema = z.string()
const functionSchema = z.instanceof(Function)

function normalizeExecutorResult(value: ToolExecutorResult | string): ToolExecutorResult {
  const parsed = stringResultSchema.safeParse(value)
  if (parsed.success) {
    return { content: parsed.data }
  }
  // SAFETY: the executor contract is `ToolExecutorResult | string`; a value
  // that is not a string is the result-object arm.
  return value as ToolExecutorResult
}

async function executeCall(
  execute: ToolExecutor | Record<string, ToolExecutor>,
  call: LLMToolCall,
): Promise<ToolExecutorResult> {
  try {
    if (functionSchema.safeParse(execute).success) {
      // SAFETY: the parameter contract is executor-or-map; a Function here is
      // the single-executor arm.
      return normalizeExecutorResult(await (execute as ToolExecutor)(call))
    }
    // SAFETY: not a Function, so this is the by-name map arm of the contract.
    const executor = (execute as Record<string, ToolExecutor>)[call.name]
    if (!executor) {
      return { content: `Unknown tool: ${call.name}`, isError: true }
    }
    return normalizeExecutorResult(await executor(call))
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { content: message, isError: true }
  }
}

/**
 * Run the standard multi-turn tool loop over `LLM.runTurn`: generate, execute
 * any requested tool calls, feed the results back as tool messages, and
 * generate again — until the model answers in plain text, the turn terminates
 * early (stopped / failed / length), or `maxSteps` is reached.
 *
 * @param initialMessages - Messages for the first turn (typically one user message).
 * @param execute - A single executor for every tool, or a map of executors by
 *   tool name. Executor errors and unknown tool names become `isError` tool
 *   messages instead of aborting the loop.
 *
 * @example
 * ```ts
 * const { outcome } = await runToolLoop(
 *   [{ role: 'user', content: 'What time is it in Tokyo?' }],
 *   { get_time: call => lookupTime(call.arguments) },
 *   { contextId: ctx.id },
 * )
 * console.log(outcome.content)
 * ```
 */
export async function runToolLoop(
  initialMessages: LLMMessage[],
  execute: ToolExecutor | Record<string, ToolExecutor>,
  options: RunToolLoopOptions = {},
): Promise<ToolLoopResult> {
  const maxSteps = options.maxSteps ?? 6
  if (!Number.isInteger(maxSteps) || maxSteps < 1) {
    throw new TypeError(
      `${ERROR_PREFIX} runToolLoop maxSteps must be a positive integer.`,
    )
  }

  const baseRequest: Omit<LLMTurnRequest, 'messages'> = {
    contextId: options.contextId,
    instructions: options.instructions,
    history: options.history,
    tools: options.tools,
    generationConfig: options.generationConfig,
  }

  let messages: LLMMessage[] = initialMessages
  let outcome: LLMTurnOutcome | undefined
  let steps = 0

  while (steps < maxSteps) {
    outcome = await LLM.runTurn({ ...baseRequest, messages }, options.onEvent, {
      signal: options.signal,
    })
    steps += 1

    if (outcome.finishReason !== 'tool_calls' || outcome.toolCalls.length === 0) {
      return { outcome, steps, stoppedAtMaxSteps: false }
    }

    const results: LLMMessage[] = []
    for (const call of outcome.toolCalls) {
      const result = await executeCall(execute, call)
      try {
        options.onToolResult?.(call, result)
      } catch {
        // observers cannot affect the loop
      }
      results.push({
        role: 'tool',
        toolCallId: call.id,
        name: call.name,
        content: result.content,
        isError: result.isError || undefined,
      })
    }
    messages = results
  }

  // SAFETY: maxSteps >= 1, so the loop body ran and `outcome` is assigned.
  return { outcome: outcome as LLMTurnOutcome, steps, stoppedAtMaxSteps: true }
}
