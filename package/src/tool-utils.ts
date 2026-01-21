import type { AnyMap } from 'react-native-nitro-modules'
import { z } from 'zod'
import type { ToolDefinition, ToolParameter, ToolParameterType } from './specs/LLM.nitro'

type ZodObjectSchema = z.ZodObject<z.ZodRawShape>
type InferArgs<T extends ZodObjectSchema> = z.infer<T>

export interface TypeSafeToolDefinition<T extends ZodObjectSchema> {
  name: string
  description: string
  arguments: T
  handler: (args: InferArgs<T>) => Promise<Record<string, unknown>>
}

function getZodTypeString(zodType: z.ZodTypeAny): ToolParameterType {
  const typeName = zodType._def.typeName
  switch (typeName) {
    case z.ZodFirstPartyTypeKind.ZodString:
      return 'string'
    case z.ZodFirstPartyTypeKind.ZodNumber:
      return 'number'
    case z.ZodFirstPartyTypeKind.ZodBoolean:
      return 'boolean'
    case z.ZodFirstPartyTypeKind.ZodArray:
      return 'array'
    case z.ZodFirstPartyTypeKind.ZodObject:
      return 'object'
    case z.ZodFirstPartyTypeKind.ZodOptional:
      return getZodTypeString((zodType as z.ZodOptional<z.ZodTypeAny>)._def.innerType)
    case z.ZodFirstPartyTypeKind.ZodDefault:
      return getZodTypeString((zodType as z.ZodDefault<z.ZodTypeAny>)._def.innerType)
    default:
      return 'string'
  }
}

function isZodOptional(zodType: z.ZodTypeAny): boolean {
  const typeName = zodType._def.typeName
  return (
    typeName === z.ZodFirstPartyTypeKind.ZodOptional ||
    typeName === z.ZodFirstPartyTypeKind.ZodDefault
  )
}

function zodSchemaToParameters(schema: ZodObjectSchema): ToolParameter[] {
  const shape = schema.shape
  const parameters: ToolParameter[] = []

  for (const [key, zodType] of Object.entries(shape)) {
    const zType = zodType as z.ZodTypeAny
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
