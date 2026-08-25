import type { AnyMap } from 'react-native-nitro-modules'
import { z } from 'zod'
import type { JsonObject } from './json'
import type { ToolDefinition, ToolParameter, ToolParameterType } from './specs/LLM.nitro'

type ZodObjectSchema = z.ZodObject
type InferArgs<T extends ZodObjectSchema> = z.infer<T>

export interface TypeSafeToolDefinition<T extends ZodObjectSchema> {
  name: string
  description: string
  arguments: T
  handler: (args: InferArgs<T>) => Promise<JsonObject>
}

const toolParameterTypeSchema = z.enum(['string', 'number', 'boolean', 'array', 'object'])

const jsonSchemaPropertySchema = z.looseObject({
  type: z.string().optional(),
  description: z.string().optional(),
})

const objectJsonSchemaSchema = z.looseObject({
  properties: z.record(z.string(), jsonSchemaPropertySchema).optional(),
  required: z.array(z.string()).optional(),
})

function toToolParameterType(type: string | undefined): ToolParameterType {
  const parsed = toolParameterTypeSchema.safeParse(type === 'integer' ? 'number' : type)
  return parsed.success ? parsed.data : 'string'
}

function zodSchemaToParameters(schema: ZodObjectSchema): ToolParameter[] {
  const jsonSchema = objectJsonSchemaSchema.parse(
    z.toJSONSchema(schema, { io: 'input', unrepresentable: 'any' }),
  )
  const required = new Set(jsonSchema.required ?? [])
  return Object.entries(jsonSchema.properties ?? {}).map(([name, property]) => ({
    name,
    type: toToolParameterType(property.type),
    description: property.description ?? '',
    required: required.has(name),
  }))
}

export function createTool<T extends ZodObjectSchema>(
  definition: TypeSafeToolDefinition<T>,
): ToolDefinition {
  return {
    name: definition.name,
    description: definition.description,
    parameters: zodSchemaToParameters(definition.arguments),
    handler: async (args: AnyMap) => {
      const parsed = definition.arguments.safeParse(args)
      if (!parsed.success) {
        // This message is fed back to the model as the tool failure; a compact
        // directive beats a raw ZodError's multi-line JSON issue dump.
        throw new Error(
          `Invalid arguments for tool "${definition.name}": ${z.prettifyError(parsed.error)}. Fix the arguments and call the tool again.`,
        )
      }
      return definition.handler(parsed.data)
    },
  }
}
