import type { AnyMap } from 'react-native-nitro-modules'
import type { z } from 'zod'
import type { ToolDefinition, ToolParameter, ToolParameterType } from './specs/LLM.nitro'

type ZodObjectSchema = z.ZodObject<z.core.$ZodShape>
type InferArgs<T extends ZodObjectSchema> = z.infer<T>

export interface TypeSafeToolDefinition<T extends ZodObjectSchema> {
  name: string
  description: string
  arguments: T
  handler: (args: InferArgs<T>) => Promise<Record<string, unknown>>
}

function getZodTypeString(zodType: z.ZodType): ToolParameterType {
  const typeName = zodType._zod.def.type
  switch (typeName) {
    case 'string':
      return 'string'
    case 'number':
    case 'int':
      return 'number'
    case 'boolean':
      return 'boolean'
    case 'array':
      return 'array'
    case 'object':
      return 'object'
    case 'optional':
      return getZodTypeString((zodType as z.ZodOptional<z.ZodType>)._zod.def.innerType)
    case 'default':
      return getZodTypeString((zodType as z.ZodDefault<z.ZodType>)._zod.def.innerType)
    default:
      return 'string'
  }
}

function isZodOptional(zodType: z.ZodType): boolean {
  const typeName = zodType._zod.def.type
  return typeName === 'optional' || typeName === 'default'
}

function zodSchemaToParameters(schema: ZodObjectSchema): ToolParameter[] {
  const shape = schema._zod.def.shape
  const parameters: ToolParameter[] = []

  for (const [key, zodType] of Object.entries(shape)) {
    const zType = zodType as z.ZodType
    parameters.push({
      name: key,
      type: getZodTypeString(zType),
      description: zType.description ?? '',
      required: !isZodOptional(zType),
    })
  }

  return parameters
}

export function createTool<T extends ZodObjectSchema>(
  definition: TypeSafeToolDefinition<T>,
): ToolDefinition {
  return {
    name: definition.name,
    description: definition.description,
    parameters: zodSchemaToParameters(definition.arguments),
    handler: async (args: AnyMap) => {
      const argsObj = args as unknown as Record<string, unknown>
      const parsedArgs = definition.arguments.parse(argsObj)
      const result = await definition.handler(parsedArgs)
      return result as unknown as AnyMap
    },
  }
}
