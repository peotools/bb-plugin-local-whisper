// bb-plugin-local-whisper — serves Cloudroom voice input with whisper.cpp on
// this machine; the work happens in host.ts. Select it with
// BB_TRANSCRIPTION=whisper/<model>, e.g. whisper/small.en.
import type { BbPluginApi } from "@get-bb/plugin-sdk";

export default function plugin(bb: BbPluginApi) {
  bb.experimental_aiServices.register({
    id: "whisper",
    displayName: "Local Whisper (whisper.cpp)",
    kinds: ["voice"],
  } as never);
}
