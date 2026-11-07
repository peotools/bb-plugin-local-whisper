// Transcribes Cloudroom voice input with whisper.cpp on this machine.
// ffmpeg converts each recording to 16 kHz mono WAV; whisper-cli reads it
// with <modelsDir>/ggml-<model>.bin, where <model> is the model segment of
// BB_TRANSCRIPTION (whisper/small.en).
//
// Long recordings: Cloudroom gives each attempt 10 s and retries once on
// "timeout" with the same audio. So a job is keyed by its audio and keeps
// running past the first deadline; the retry picks up the same job. That
// gives ~20 s of whisper time. Measured on a Ryzen 9 5950X, 5 min of dense
// speech, small.en, 16 threads: beam search 16.4 s, greedy 11.5 s.
//
// Optional ~/.config/bb-local-whisper.json:
//   { "whisperCli": "/path/to/whisper-cli", "ffmpeg": "/path/to/ffmpeg",
//     "modelsDir": "/path/to/models", "threads": 8 }
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { access, appendFile, mkdir, mkdtemp, readFile, realpath, rm, stat, writeFile } from "node:fs/promises";
import { availableParallelism, homedir, tmpdir } from "node:os";
import { delimiter, dirname, join } from "node:path";
import { defineRpcContract } from "@get-bb/plugin-sdk";
import { experimental_defineHostEntry } from "@get-bb/plugin-sdk/host";
import { z } from "zod";

const HOME = homedir();
const CONFIG_FILE = join(HOME, ".config/bb-local-whisper.json");
const LOG_DIR = join(HOME, ".local/state/bb-local-whisper");
// GUI apps often start with a short PATH, so look in the usual places too.
const EXTRA_BIN_DIRS = [join(HOME, ".local/bin"), "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"];
const BEAM_MAX_SECONDS = 240; // longer clips decode greedy to fit Cloudroom's two 10 s attempts
const JOB_LIMIT_MS = 120_000; // a job nobody waits for still ends
const ANSWER_MARGIN_MS = 300; // answer before Cloudroom's own deadline
const KEEP_DONE_MS = 60_000;

const configSchema = z
  .object({
    whisperCli: z.string().min(1).optional(),
    ffmpeg: z.string().min(1).optional(),
    modelsDir: z.string().min(1).optional(),
    threads: z.number().int().positive().optional(),
  })
  .strict();
type Config = z.infer<typeof configSchema>;

async function readConfig(): Promise<Config> {
  let raw: string;
  try {
    raw = await readFile(CONFIG_FILE, "utf8");
  } catch {
    return {};
  }
  return configSchema.parse(JSON.parse(raw));
}

const expand = (path: string) => path.replace(/^~(?=\/|$)/, HOME);

async function findExecutable(name: string, configured: string | undefined) {
  const candidates = configured
    ? [expand(configured)]
    : [...(process.env.PATH ?? "").split(delimiter), ...EXTRA_BIN_DIRS].filter(Boolean).map((dir) => join(dir, name));
  for (const path of candidates) {
    try {
      await access(path, constants.X_OK);
      return path;
    } catch {}
  }
  return null;
}

// Physical cores, roughly: whisper.cpp gains little from SMT threads.
const defaultThreads = () => Math.max(4, Math.min(16, Math.floor(availableParallelism() / 2)));

// Mirrors the SDK's experimental_aiServicesHostContract voice method, which
// the published SDK (0.6.49) does not export yet.
const failure = z
  .object({
    ok: z.literal(false),
    code: z.enum(["timeout", "rate_limited", "service_unavailable", "auth_required", "request_failed", "invalid_response"]),
    message: z.string().min(1),
  })
  .strict();
const contract = defineRpcContract({
  "ai.voice.transcribe": {
    input: z
      .object({
        serviceId: z.string().min(1),
        model: z.string().min(1),
        audioBase64: z.string().min(1),
        mimeType: z.string().min(1),
        filename: z.string().min(1),
        prompt: z.string().nullable(),
        timeoutMs: z.number().int().positive(),
      })
      .strict(),
    output: z.union([z.object({ ok: z.literal(true), model: z.string().min(1), text: z.string() }).strict(), failure]),
  },
});

type Tools = { whisperCli: string; ffmpeg: string; modelsDir: string; threads: number; env: NodeJS.ProcessEnv };

async function resolveTools(): Promise<Tools | string> {
  const config = await readConfig();
  const whisperCli = await findExecutable("whisper-cli", config.whisperCli);
  if (!whisperCli) return "whisper-cli not found. Install whisper.cpp or set whisperCli in " + CONFIG_FILE;
  const ffmpeg = await findExecutable("ffmpeg", config.ffmpeg);
  if (!ffmpeg) return "ffmpeg not found. Install it or set ffmpeg in " + CONFIG_FILE;
  // A whisper.cpp unpacked outside the system keeps its libraries next to bin/.
  const root = dirname(dirname(await realpath(whisperCli)));
  const libs = [join(root, "lib"), join(root, "lib/x86_64-linux-gnu"), join(root, "lib/aarch64-linux-gnu")];
  return {
    whisperCli,
    ffmpeg,
    modelsDir: expand(config.modelsDir ?? "~/.cache/whisper.cpp"),
    threads: config.threads ?? defaultThreads(),
    env: { ...process.env, LD_LIBRARY_PATH: [...libs, process.env.LD_LIBRARY_PATH].filter(Boolean).join(delimiter) },
  };
}

function run(file: string, args: string[], env: NodeJS.ProcessEnv, timeoutMs: number) {
  return new Promise<void>((resolve, reject) => {
    execFile(file, args, { env, timeout: timeoutMs, maxBuffer: 8 * 1024 * 1024 }, (error, _stdout, stderr) => {
      if (!error) return resolve();
      reject(new Error(`${file} failed: ${String(stderr || error.message).trim().slice(-400)}`));
    });
  });
}

async function log(line: string) {
  await mkdir(LOG_DIR, { recursive: true });
  await appendFile(join(LOG_DIR, "transcriptions.log"), `${new Date().toISOString()} ${line}\n`);
}

async function transcribe(tools: Tools, modelName: string, model: string, audioBase64: string, prompt: string | null) {
  const started = Date.now();
  const dir = await mkdtemp(join(tmpdir(), "bb-whisper-"));
  try {
    const audio = join(dir, "input");
    const wav = join(dir, "audio.wav");
    await writeFile(audio, Buffer.from(audioBase64, "base64"));
    await run(tools.ffmpeg, ["-loglevel", "error", "-y", "-i", audio, "-ar", "16000", "-ac", "1", "-f", "wav", wav], tools.env, JOB_LIMIT_MS);
    const seconds = Math.round(((await stat(wav)).size - 44) / 32000); // 16 kHz, 16-bit, mono
    const greedy = seconds > BEAM_MAX_SECONDS;
    const lang = modelName.endsWith(".en") ? "en" : "auto";
    const args = ["-m", model, "-l", lang, "-t", String(tools.threads), "-nt", "-np", "-otxt", "-of", join(dir, "out"), "-f", wav];
    if (greedy) args.push("-bs", "1", "-bo", "1");
    if (prompt) args.push("--prompt", prompt.slice(0, 800));
    await run(tools.whisperCli, args, tools.env, JOB_LIMIT_MS - (Date.now() - started));
    const text = (await readFile(join(dir, "out.txt"), "utf8")).replace(/\s+/g, " ").trim();
    await log(`${modelName} ${Date.now() - started} ms ${text.length} chars ${seconds} s audio ${greedy ? "greedy" : "beam"}`);
    return text;
  } catch (error) {
    await log(`${modelName} FAILED after ${Date.now() - started} ms: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// One job per recording, shared by Cloudroom's first attempt and its retry.
const jobs = new Map<string, Promise<string>>();

function job(tools: Tools, modelName: string, model: string, audioBase64: string, prompt: string | null) {
  const key = createHash("sha256").update(modelName).update("\0").update(audioBase64).digest("hex");
  let promise = jobs.get(key);
  if (!promise) {
    promise = transcribe(tools, modelName, model, audioBase64, prompt);
    jobs.set(key, promise);
    const forget = () => setTimeout(() => jobs.delete(key), KEEP_DONE_MS).unref();
    promise.then(forget, forget);
  }
  return promise;
}

export default experimental_defineHostEntry({
  contract,
  handlers: {
    "ai.voice.transcribe": async (input) => {
      if (!/^[\w.-]+$/.test(input.model)) {
        return { ok: false as const, code: "request_failed" as const, message: `Bad model name "${input.model}".` };
      }
      let tools: Tools | string;
      try {
        tools = await resolveTools();
      } catch (error) {
        tools = `Bad ${CONFIG_FILE}: ${error instanceof Error ? error.message : String(error)}`;
      }
      if (typeof tools === "string") return { ok: false as const, code: "service_unavailable" as const, message: tools };
      const model = join(tools.modelsDir, `ggml-${input.model}.bin`);
      try {
        await access(model);
      } catch {
        return { ok: false as const, code: "service_unavailable" as const, message: `Missing model ${model}` };
      }
      const pending = job(tools, input.model, model, input.audioBase64, input.prompt);
      let timer: NodeJS.Timeout | undefined;
      const deadline = new Promise<null>((resolve) => {
        timer = setTimeout(() => resolve(null), Math.max(500, input.timeoutMs - ANSWER_MARGIN_MS));
      });
      try {
        const text = await Promise.race([pending, deadline]);
        if (text === null) {
          return { ok: false as const, code: "timeout" as const, message: "Still transcribing; the retry collects the result." };
        }
        if (!text) return { ok: false as const, code: "invalid_response" as const, message: "Whisper returned an empty transcript." };
        return { ok: true as const, model: input.model, text };
      } catch (error) {
        return { ok: false as const, code: "request_failed" as const, message: error instanceof Error ? error.message : String(error) };
      } finally {
        clearTimeout(timer);
      }
    },
  },
});
