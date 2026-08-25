import { z } from 'zod'
import { LLM } from './llm'
import type { AbortSignalLike } from './runtime'
import { ERROR_PREFIX, functionSchema, stringSchema } from './runtime'
import type { LLMGenerationConfig, StreamEvent } from './specs/LLM.nitro'
import type {
  LLMMessage,
  LLMToolCall,
  LLMTurnOutcome,
  LLMTurnRequest,
  ToolSchema,
} from './turn'

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

const executorResultSchema = z.looseObject({
  content: z.string(),
  isError: z.boolean().optional(),
})

function normalizeExecutorResult(value: ToolExecutorResult | string): ToolExecutorResult {
  const asString = stringSchema.safeParse(value)
  if (asString.success) {
    return { content: asString.data }
  }
  const asResult = executorResultSchema.safeParse(value)
  if (asResult.success) {
    return { content: asResult.data.content, isError: asResult.data.isError }
  }
  // A void or malformed executor return (undefined, {}, a number) must not
  // crash the loop; surface it to the model as a failure, the same way
  // executor exceptions are handled.
  return {
    content:
      'Tool executor returned no usable result (expected a string or { content }).',
    isError: true,
  }
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
  let steps = 0

  for (;;) {
    const outcome = await LLM.runTurn({ ...baseRequest, messages }, options.onEvent, {
      signal: options.signal,
    })
    steps += 1

    if (outcome.finishReason !== 'tool_calls' || outcome.toolCalls.length === 0) {
      return { outcome, steps, stoppedAtMaxSteps: false }
    }

    if (steps >= maxSteps) {
      // Deliberately do NOT execute this round's tool calls: nothing would
      // feed their results back, so side effects would be wasted — and in
      // warm mode the context would be left waiting for results that never
      // arrive. The unexecuted calls are on `outcome.toolCalls` for the
      // caller to handle.
      return { outcome, steps, stoppedAtMaxSteps: true }
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

    if (options.contextId !== undefined) {
      // Warm mode: the Turn Context retains the transcript natively, so the
      // next request carries only the new tool results.
      messages = results
    } else {
      // Cold mode: every native turn is stateless, so the request itself must
      // carry the whole exchange — the prior messages, the assistant turn
      // that made the calls, and the results. Sending only the results would
      // hand the model orphaned tool outputs with no question attached.
      messages = [
        ...messages,
        { role: 'assistant', content: outcome.content, toolCalls: outcome.toolCalls },
        ...results,
      ]
    }
  }
}
