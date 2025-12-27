export enum ModelFamily {
  Llama = 'Llama',
  Qwen = 'Qwen',
  Gemma = 'Gemma',
  Phi = 'Phi',
  SmolLM = 'SmolLM',
  OpenELM = 'OpenELM',
}

export enum ModelProvider {
  Meta = 'Meta',
  Alibaba = 'Alibaba',
  Google = 'Google',
  Microsoft = 'Microsoft',
  HuggingFace = 'HuggingFace',
  Apple = 'Apple',
}

export type ModelQuantization = '4bit' | '8bit'

export interface ModelInfo {
  id: MLXModel
  family: ModelFamily
  provider: ModelProvider
  parameters: string
  quantization: ModelQuantization
  displayName: string
  downloadSize: number
}

export enum MLXModel {
  // Llama 3.2 (Meta) - 1B and 3B variants
  Llama_3_2_1B_Instruct_4bit = 'mlx-community/Llama-3.2-1B-Instruct-4bit',
  Llama_3_2_1B_Instruct_8bit = 'mlx-community/Llama-3.2-1B-Instruct-8bit',
  Llama_3_2_3B_Instruct_4bit = 'mlx-community/Llama-3.2-3B-Instruct-4bit',
  Llama_3_2_3B_Instruct_8bit = 'mlx-community/Llama-3.2-3B-Instruct-8bit',

  // Qwen 2.5 (Alibaba) - 0.5B, 1.5B, 3B variants
  Qwen2_5_0_5B_Instruct_4bit = 'mlx-community/Qwen2.5-0.5B-Instruct-4bit',
  Qwen2_5_0_5B_Instruct_8bit = 'mlx-community/Qwen2.5-0.5B-Instruct-8bit',
  Qwen2_5_1_5B_Instruct_4bit = 'mlx-community/Qwen2.5-1.5B-Instruct-4bit',
  Qwen2_5_1_5B_Instruct_8bit = 'mlx-community/Qwen2.5-1.5B-Instruct-8bit',
  Qwen2_5_3B_Instruct_4bit = 'mlx-community/Qwen2.5-3B-Instruct-4bit',
  Qwen2_5_3B_Instruct_8bit = 'mlx-community/Qwen2.5-3B-Instruct-8bit',

  // Qwen 3 - 1.7B variant
  Qwen3_1_7B_4bit = 'mlx-community/Qwen3-1.7B-4bit',
  Qwen3_1_7B_8bit = 'mlx-community/Qwen3-1.7B-8bit',

  // Gemma 3 (Google) - 1B variant
  Gemma_3_1B_IT_4bit = 'mlx-community/gemma-3-1b-it-4bit',
  Gemma_3_1B_IT_8bit = 'mlx-community/gemma-3-1b-it-8bit',

  // Phi 3.5 Mini (Microsoft) - ~3.8B but runs well on mobile
  Phi_3_5_Mini_Instruct_4bit = 'mlx-community/Phi-3.5-mini-instruct-4bit',
  Phi_3_5_Mini_Instruct_8bit = 'mlx-community/Phi-3.5-mini-instruct-8bit',

  // Phi 4 Mini (Microsoft)
  Phi_4_Mini_Instruct_4bit = 'mlx-community/Phi-4-mini-instruct-4bit',
  Phi_4_Mini_Instruct_8bit = 'mlx-community/Phi-4-mini-instruct-8bit',

  // SmolLM (HuggingFace) - 1.7B
  SmolLM_1_7B_Instruct_4bit = 'mlx-community/SmolLM-1.7B-Instruct-4bit',
  SmolLM_1_7B_Instruct_8bit = 'mlx-community/SmolLM-1.7B-Instruct-8bit',

  // SmolLM2 (HuggingFace) - 1.7B
  SmolLM2_1_7B_Instruct_4bit = 'mlx-community/SmolLM2-1.7B-Instruct-4bit',
  SmolLM2_1_7B_Instruct_8bit = 'mlx-community/SmolLM2-1.7B-Instruct-8bit',

  // OpenELM (Apple) - 1.1B and 3B
  OpenELM_1_1B_4bit = 'mlx-community/OpenELM-1_1B-4bit',
  OpenELM_1_1B_8bit = 'mlx-community/OpenELM-1_1B-8bit',
  OpenELM_3B_4bit = 'mlx-community/OpenELM-3B-4bit',
  OpenELM_3B_8bit = 'mlx-community/OpenELM-3B-8bit',
}

export const MLXModels: ModelInfo[] = [
  {
    id: MLXModel.Llama_3_2_1B_Instruct_4bit,
    family: ModelFamily.Llama,
    provider: ModelProvider.Meta,
    parameters: '1B',
    quantization: '4bit',
    displayName: 'Llama 3.2 1B Instruct (4-bit)',
    downloadSize: 1407777762,
  },
  {
    id: MLXModel.Llama_3_2_1B_Instruct_8bit,
    family: ModelFamily.Llama,
    provider: ModelProvider.Meta,
    parameters: '1B',
    quantization: '8bit',
    displayName: 'Llama 3.2 1B Instruct (8-bit)',
    downloadSize: 1313157436,
  },
  {
    id: MLXModel.Llama_3_2_3B_Instruct_4bit,
    family: ModelFamily.Llama,
    provider: ModelProvider.Meta,
    parameters: '3B',
    quantization: '4bit',
    displayName: 'Llama 3.2 3B Instruct (4-bit)',
    downloadSize: 2019397474,
  },
  {
    id: MLXModel.Llama_3_2_3B_Instruct_8bit,
    family: ModelFamily.Llama,
    provider: ModelProvider.Meta,
    parameters: '3B',
    quantization: '8bit',
    displayName: 'Llama 3.2 3B Instruct (8-bit)',
    downloadSize: 3413784042,
  },
  {
    id: MLXModel.Qwen2_5_0_5B_Instruct_4bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '0.5B',
    quantization: '4bit',
    displayName: 'Qwen 2.5 0.5B Instruct (4-bit)',
    downloadSize: 278064920,
  },
  {
    id: MLXModel.Qwen2_5_0_5B_Instruct_8bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '0.5B',
    quantization: '8bit',
    displayName: 'Qwen 2.5 0.5B Instruct (8-bit)',
    downloadSize: 525045902,
  },
  {
    id: MLXModel.Qwen2_5_1_5B_Instruct_4bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '1.5B',
    quantization: '4bit',
    displayName: 'Qwen 2.5 1.5B Instruct (4-bit)',
    downloadSize: 868628559,
  },
  {
    id: MLXModel.Qwen2_5_1_5B_Instruct_8bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '1.5B',
    quantization: '8bit',
    displayName: 'Qwen 2.5 1.5B Instruct (8-bit)',
    downloadSize: 1640414038,
  },
  {
    id: MLXModel.Qwen2_5_3B_Instruct_4bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '3B',
    quantization: '4bit',
    displayName: 'Qwen 2.5 3B Instruct (4-bit)',
    downloadSize: 1736293090,
  },
  {
    id: MLXModel.Qwen2_5_3B_Instruct_8bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '3B',
    quantization: '8bit',
    displayName: 'Qwen 2.5 3B Instruct (8-bit)',
    downloadSize: 3279142142,
  },
  {
    id: MLXModel.Qwen3_1_7B_4bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '1.7B',
    quantization: '4bit',
    displayName: 'Qwen 3 1.7B (4-bit)',
    downloadSize: 979502864,
  },
  {
    id: MLXModel.Qwen3_1_7B_8bit,
    family: ModelFamily.Qwen,
    provider: ModelProvider.Alibaba,
    parameters: '1.7B',
    quantization: '8bit',
    displayName: 'Qwen 3 1.7B (8-bit)',
    downloadSize: 1839729195,
  },
  {
    id: MLXModel.Gemma_3_1B_IT_4bit,
    family: ModelFamily.Gemma,
    provider: ModelProvider.Google,
    parameters: '1B',
    quantization: '4bit',
    displayName: 'Gemma 3 1B IT (4-bit)',
    downloadSize: 770650946,
  },
  {
    id: MLXModel.Gemma_3_1B_IT_8bit,
    family: ModelFamily.Gemma,
    provider: ModelProvider.Google,
    parameters: '1B',
    quantization: '8bit',
    displayName: 'Gemma 3 1B IT (8-bit)',
    downloadSize: 1421522471,
  },
  {
    id: MLXModel.Phi_3_5_Mini_Instruct_4bit,
    family: ModelFamily.Phi,
    provider: ModelProvider.Microsoft,
    parameters: '3.8B',
    quantization: '4bit',
    displayName: 'Phi 3.5 Mini Instruct (4-bit)',
    downloadSize: 2150195856,
  },
  {
    id: MLXModel.Phi_3_5_Mini_Instruct_8bit,
    family: ModelFamily.Phi,
    provider: ModelProvider.Microsoft,
    parameters: '3.8B',
    quantization: '8bit',
    displayName: 'Phi 3.5 Mini Instruct (8-bit)',
    downloadSize: 4060636056,
  },
  {
    id: MLXModel.Phi_4_Mini_Instruct_4bit,
    family: ModelFamily.Phi,
    provider: ModelProvider.Microsoft,
    parameters: '3.8B',
    quantization: '4bit',
    displayName: 'Phi 4 Mini Instruct (4-bit)',
    downloadSize: 2173624891,
  },
  {
    id: MLXModel.Phi_4_Mini_Instruct_8bit,
    family: ModelFamily.Phi,
    provider: ModelProvider.Microsoft,
    parameters: '3.8B',
    quantization: '8bit',
    displayName: 'Phi 4 Mini Instruct (8-bit)',
    downloadSize: 4091536167,
  },
  {
    id: MLXModel.SmolLM_1_7B_Instruct_4bit,
    family: ModelFamily.SmolLM,
    provider: ModelProvider.HuggingFace,
    parameters: '1.7B',
    quantization: '4bit',
    displayName: 'SmolLM 1.7B Instruct (4-bit)',
    downloadSize: 962855374,
  },
  {
    id: MLXModel.SmolLM_1_7B_Instruct_8bit,
    family: ModelFamily.SmolLM,
    provider: ModelProvider.HuggingFace,
    parameters: '1.7B',
    quantization: '8bit',
    displayName: 'SmolLM 1.7B Instruct (8-bit)',
    downloadSize: 1818493993,
  },
  {
    id: MLXModel.SmolLM2_1_7B_Instruct_4bit,
    family: ModelFamily.SmolLM,
    provider: ModelProvider.HuggingFace,
    parameters: '1.7B',
    quantization: '4bit',
    displayName: 'SmolLM2 1.7B Instruct (4-bit)',
    downloadSize: 980000000,
  },
  {
    id: MLXModel.SmolLM2_1_7B_Instruct_8bit,
    family: ModelFamily.SmolLM,
    provider: ModelProvider.HuggingFace,
    parameters: '1.7B',
    quantization: '8bit',
    displayName: 'SmolLM2 1.7B Instruct (8-bit)',
    downloadSize: 1850000000,
  },
  {
    id: MLXModel.OpenELM_1_1B_4bit,
    family: ModelFamily.OpenELM,
    provider: ModelProvider.Apple,
    parameters: '1.1B',
    quantization: '4bit',
    displayName: 'OpenELM 1.1B (4-bit)',
    downloadSize: 608162655,
  },
  {
    id: MLXModel.OpenELM_1_1B_8bit,
    family: ModelFamily.OpenELM,
    provider: ModelProvider.Apple,
    parameters: '1.1B',
    quantization: '8bit',
    displayName: 'OpenELM 1.1B (8-bit)',
    downloadSize: 1148048397,
  },
  {
    id: MLXModel.OpenELM_3B_4bit,
    family: ModelFamily.OpenELM,
    provider: ModelProvider.Apple,
    parameters: '3B',
    quantization: '4bit',
    displayName: 'OpenELM 3B (4-bit)',
    downloadSize: 1650000000,
  },
  {
    id: MLXModel.OpenELM_3B_8bit,
    family: ModelFamily.OpenELM,
    provider: ModelProvider.Apple,
    parameters: '3B',
    quantization: '8bit',
    displayName: 'OpenELM 3B (8-bit)',
    downloadSize: 3100000000,
  },
]
