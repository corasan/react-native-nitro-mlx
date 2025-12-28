const { getDefaultConfig } = require('expo/metro-config')
const path = require('path')

const projectRoot = __dirname
const monorepoRoot = path.resolve(projectRoot, '../..')
const isDev = process.env.NODE_ENV !== 'production'

const config = getDefaultConfig(projectRoot)

const watchFolders = [monorepoRoot]
const nodeModulesPaths = [
  path.resolve(projectRoot, 'node_modules'),
  path.resolve(monorepoRoot, 'node_modules'),
]

if (isDev) {
  const nitroMlxRoot = path.resolve(projectRoot, '../package')
  watchFolders.push(nitroMlxRoot)
  nodeModulesPaths.unshift(nitroMlxRoot)
}

config.watchFolders = watchFolders
config.resolver.nodeModulesPaths = nodeModulesPaths

config.resolver.assetExts.push('pte')
config.resolver.assetExts.push('bin')
config.resolver.sourceExts.push('sql')

module.exports = config
