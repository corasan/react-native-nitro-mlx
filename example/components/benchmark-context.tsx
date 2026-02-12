import { createContext, type ReactNode, useContext, useState } from 'react'

export type BenchmarkResult = {
  tokensPerSecond: number
  timeToFirstToken: number
  totalTokens: number
  totalTime: number
  toolExecutionTime: number
  timestamp: Date
}

type BenchmarkContextType = {
  results: BenchmarkResult[]
  addResult: (result: BenchmarkResult) => void
  clearResults: () => void
}

const BenchmarkContext = createContext<BenchmarkContextType | null>(null)

export function BenchmarkProvider({ children }: { children: ReactNode }) {
  const [results, setResults] = useState<BenchmarkResult[]>([])

  const addResult = (result: BenchmarkResult) => {
    setResults(prev => [...prev, result])
  }

  const clearResults = () => {
    setResults([])
  }

  return (
    <BenchmarkContext.Provider
      value={{
        results,
        addResult,
        clearResults,
      }}
    >
      {children}
    </BenchmarkContext.Provider>
  )
}

export function useBenchmark() {
  const context = useContext(BenchmarkContext)
  if (!context) {
    throw new Error('useBenchmark must be used within a BenchmarkProvider')
  }
  return context
}
