# bb-plugin-local-whisper

Cloudroom voice input transcribed by [whisper.cpp](https://github.com/ggml-org/whisper.cpp) on your own machine. Your recording goes to a local `whisper-cli`, not a cloud service.

Long dictation works. On a Ryzen 9 5950X with `small.en`, five minutes of nonstop talking comes back in about 12 seconds.

## Setup

1. Install whisper.cpp and ffmpeg so `whisper-cli` and `ffmpeg` are on your PATH. On macOS: `brew install whisper-cpp ffmpeg`. On Linux, build whisper.cpp from source or use your distro's package.
2. Download a model:

   ```
   mkdir -p ~/.cache/whisper.cpp
   curl -L -o ~/.cache/whisper.cpp/ggml-small.en.bin \
     https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
   ```

3. Install the plugin:

   ```
   room-cli plugin install git:https://github.com/peotools/bb-plugin-local-whisper
   ```

4. Point Cloudroom's voice input at it. The part after `whisper/` picks the model file, so `whisper/small.en` reads `ggml-small.en.bin`:

   ```
   node <bb-app>/dist/bb-app.js --data-dir ~/.gui-cloudroom config set BB_TRANSCRIPTION whisper/small.en
   ```

5. Test it: `room-cli voice transcribe some-clip.webm`

## Options

Everything is found automatically. If you need to override something, create `~/.config/bb-local-whisper.json`:

```json
{
  "whisperCli": "~/whisper.cpp/build/bin/whisper-cli",
  "ffmpeg": "/usr/bin/ffmpeg",
  "modelsDir": "~/.cache/whisper.cpp",
  "threads": 8
}
```

Threads default to half your logical cores, between 4 and 16.

## How long recordings fit

Cloudroom gives voice transcription 10 seconds per attempt and retries once on a timeout with the same audio. The plugin keys each job by its audio and keeps whisper running through the first timeout, so the retry collects the result. That gives about 20 seconds of whisper time.

Recordings up to 4 minutes use beam search, whisper.cpp's default. Longer ones use greedy decoding, which is about 30% faster, so up to roughly 7 to 8 minutes of speech fits on a fast desktop CPU. Slower machines fit less. A smaller model such as `base.en` is faster still.

If local transcription fails and the recording is under 4 MB, Cloudroom falls back to its own cloud transcription. That is Cloudroom's behavior, not this plugin's.

Each transcription is logged with timing, audio length, and decoding mode in `~/.local/state/bb-local-whisper/transcriptions.log`. The log never contains your words.

## Development

```
npm install
room-cli plugin install ./
room-cli plugin build . && room-cli plugin reload local-whisper
```

Tested on Linux (KDE Plasma, Ryzen 9 5950X). macOS should work with Homebrew's whisper-cpp but is untested.

MIT license.
