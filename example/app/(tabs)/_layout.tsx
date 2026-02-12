import { Tabs } from 'expo-router'

export default function TabsLayout() {
  return (
    <Tabs>
      <Tabs.Screen
        name="index"
        options={{ title: 'Chat', headerShown: false }}
      />
      <Tabs.Screen
        name="tts"
        options={{ title: 'TTS', headerShown: false }}
      />
    </Tabs>
  )
}
