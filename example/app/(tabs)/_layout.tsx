import { Icon, Label, NativeTabs } from 'expo-router/unstable-native-tabs'

export default function TabsLayout() {
  return (
    <NativeTabs>
      <NativeTabs.Trigger name="index">
        <Label>Home</Label>
        <Icon sf="house.fill" />
      </NativeTabs.Trigger>

      <NativeTabs.Trigger name="tts">
        <Label>TTS</Label>
        <Icon sf="speaker.1.fill" />
      </NativeTabs.Trigger>

      <NativeTabs.Trigger name="stt">
        <Label>STT</Label>
        <Icon sf="mic.fill" />
      </NativeTabs.Trigger>
    </NativeTabs>
  )
}
