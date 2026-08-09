/**
 * auto-pi — automatic tool-call review for pi.
 *
 * Every proposed tool call is scored by a local encoder (ProCreations/auto-1b-bf16 or
 * auto-0.4b) before it runs. Safe calls execute untouched; unsafe ones are blocked and the
 * agent is told why, so it can adjust instead of silently failing.
 *
 * The model is not a general LLM being asked politely — it is a classifier trained on one
 * exact serialization. Any drift in that format costs real accuracy, so buildModelInput()
 * below reproduces it byte for byte.
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = resolve(HERE, "..");
const HOME_DIR = join(homedir(), ".pi", "auto-pi");
const CONFIG_PATH = join(HOME_DIR, "config.json");
const VENV_PYTHON = join(HOME_DIR, "venv", "bin", "python3");
const SERVER_PY = join(PKG_ROOT, "python", "auto_server.py");

const MODELS: Record<string, string> = {
	"auto-1b": "ProCreations/auto-1b-bf16",
	"auto-1b-bf16": "ProCreations/auto-1b-bf16",
	"auto-0.4b": "ProCreations/auto-0.4b",
};

type Config = {
	enabled: boolean;
	model: string;
	threshold: number;
	device: string;
	failClosed: boolean;
	maxHistorySteps: number;
	maxResultChars: number;
	maxInputChars: number;
	exemptTools: string[];
	notifyOnApprove: boolean;
};

const DEFAULTS: Config = {
	enabled: true,
	model: "ProCreations/auto-1b-bf16",
	threshold: 0.5,
	device: "auto",
	// Fail open by default: a missing model or a crashed sidecar should not brick the agent.
	// Set true to refuse every call the gate could not score.
	failClosed: false,
	maxHistorySteps: 20,
	maxResultChars: 2000,
	maxInputChars: 120000,
	exemptTools: [],
	notifyOnApprove: false,
};

function loadConfig(): Config {
	try {
		if (existsSync(CONFIG_PATH)) {
			return { ...DEFAULTS, ...JSON.parse(readFileSync(CONFIG_PATH, "utf8")) };
		}
	} catch {
		/* fall through to defaults — a corrupt config must not stop pi from starting */
	}
	return { ...DEFAULTS };
}

function saveConfig(cfg: Config): void {
	mkdirSync(HOME_DIR, { recursive: true });
	writeFileSync(CONFIG_PATH, JSON.stringify(cfg, null, 2));
}

/** Render tool arguments the way the training corpus did. */
function renderArgs(input: unknown): string {
	if (typeof input === "string") return input;
	if (input && typeof input === "object") {
		const obj = input as Record<string, unknown>;
		// Shell calls dominate the corpus and appear there as the bare command string, not
		// as JSON. Matching that shape matters more than being uniform.
		if (typeof obj.command === "string") return obj.command;
		try {
			return JSON.stringify(obj);
		} catch {
			return String(input);
		}
	}
	return String(input ?? "");
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		return content
			.filter((c: any) => c && c.type === "text" && typeof c.text === "string")
			.map((c: any) => c.text)
			.join("\n");
	}
	return "";
}

type HistoryStep = { tool: string; args: string; result: string };

/**
 * Walk the active branch and rebuild (a) the user's most recent request and (b) the
 * completed tool calls leading up to now.
 *
 * Only calls that already have a result are included, which conveniently excludes both the
 * call under judgement and any un-executed siblings from the same assistant message.
 */
function readSession(ctx: any, currentToolCallId: string) {
	let entries: any[] = [];
	try {
		entries = ctx.sessionManager?.buildContextEntries?.() ?? ctx.sessionManager?.getBranch?.() ?? [];
	} catch {
		entries = [];
	}

	let userRequest = "";
	const pending = new Map<string, { tool: string; args: string }>();
	const steps: HistoryStep[] = [];

	for (const entry of entries) {
		const msg = entry?.message ?? entry;
		if (!msg || typeof msg !== "object") continue;

		if (msg.role === "user") {
			const t = textOf(msg.content).trim();
			// Keep the latest user turn: it is what authorises (or fails to authorise) the call.
			if (t) userRequest = t;
		} else if (msg.role === "assistant" && Array.isArray(msg.content)) {
			for (const block of msg.content) {
				if (block?.type === "toolCall" && typeof block.id === "string") {
					pending.set(block.id, { tool: String(block.name ?? "?"), args: renderArgs(block.arguments) });
				}
			}
		} else if (msg.role === "toolResult" && typeof msg.toolCallId === "string") {
			const call = pending.get(msg.toolCallId);
			if (!call || msg.toolCallId === currentToolCallId) continue;
			steps.push({ tool: call.tool, args: call.args, result: textOf(msg.content) });
			pending.delete(msg.toolCallId);
		} else if (msg.role === "bashExecution" && typeof msg.command === "string") {
			// User-run `!` commands are real agent-visible context too.
			steps.push({ tool: "bash", args: msg.command, result: String(msg.output ?? "") });
		}
	}

	return { userRequest, steps };
}

/**
 * Reproduce the exact serialization auto was trained on.
 *
 * Order is deliberate and load-bearing: the proposed call and the user request come first
 * so they survive truncation from the tail; history follows. The section headers are
 * literal training tokens — do not reword them.
 */
function buildModelInput(
	call: { tool: string; args: string },
	userRequest: string,
	steps: HistoryStep[],
	cfg: Config,
): string {
	const parts = [
		"### PROPOSED TOOL CALL",
		`tool: ${call.tool}`,
		`args: ${call.args}`,
		"",
		"### USER REQUEST",
		userRequest || "(no user request recorded)",
		"",
		"### AGENT HISTORY",
	];

	const recent = steps.slice(-Math.max(0, cfg.maxHistorySteps));
	if (recent.length === 0) {
		parts.push("(no prior actions)");
	} else {
		recent.forEach((s, i) => {
			let result = s.result ?? "";
			if (result.length > cfg.maxResultChars) {
				result = `${result.slice(0, cfg.maxResultChars)}… [truncated]`;
			}
			parts.push(`[${i + 1}] ${s.tool}(${s.args})\n-> ${result}`);
		});
	}

	let text = parts.join("\n");
	if (text.length > cfg.maxInputChars) {
		// Trim the tail (oldest history is at the end of the section) rather than the head,
		// so the call and the request always survive.
		text = `${text.slice(0, cfg.maxInputChars)}\n… [history truncated]`;
	}
	return text;
}

/** Owns the python sidecar: lazy spawn, request multiplexing, restart on death. */
class Scorer {
	private proc: ChildProcessWithoutNullStreams | null = null;
	private buf = "";
	private nextId = 1;
	private waiting = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
	private starting: Promise<void> | null = null;
	info: any = null;
	lastError: string | null = null;

	constructor(private cfg: Config) {}

	private pythonBin(): string {
		if (existsSync(VENV_PYTHON)) return VENV_PYTHON;
		return process.env.AUTO_PI_PYTHON || "python3";
	}

	isRunning(): boolean {
		return this.proc !== null && !this.proc.killed;
	}

	async start(): Promise<void> {
		if (this.isRunning()) return;
		if (this.starting) return this.starting;

		this.starting = new Promise<void>((resolveStart, rejectStart) => {
			if (!existsSync(SERVER_PY)) {
				const e = new Error(`scorer script missing at ${SERVER_PY}`);
				this.lastError = e.message;
				rejectStart(e);
				return;
			}

			let proc: ChildProcessWithoutNullStreams;
			try {
				proc = spawn(
					this.pythonBin(),
					[SERVER_PY, "--model", this.cfg.model, "--device", this.cfg.device],
					{ stdio: ["pipe", "pipe", "pipe"] },
				);
			} catch (err: any) {
				this.lastError = `could not start python: ${err?.message ?? err}`;
				rejectStart(new Error(this.lastError));
				return;
			}

			this.proc = proc;
			let settled = false;

			// The sidecar must never keep pi alive. In non-interactive runs (`pi -p ...`)
			// session_shutdown is not guaranteed to fire before pi tries to exit, and a live
			// child with open stdio pipes pins the event loop open until the harness kills it.
			// unref lets pi exit; the exit hooks below stop that leaving an orphan holding GPU memory.
			proc.unref();
			proc.stdout.unref?.();
			proc.stderr.unref?.();
			proc.stdin.unref?.();

			const reap = () => {
				try {
					proc.kill();
				} catch {
					/* already gone */
				}
			};
			process.once("exit", reap);
			process.once("SIGINT", reap);
			process.once("SIGTERM", reap);
			proc.once("exit", () => {
				process.off("exit", reap);
				process.off("SIGINT", reap);
				process.off("SIGTERM", reap);
			});

			proc.stdout.setEncoding("utf8");
			proc.stdout.on("data", (chunk: string) => {
				this.buf += chunk;
				let nl: number;
				while ((nl = this.buf.indexOf("\n")) >= 0) {
					const line = this.buf.slice(0, nl).trim();
					this.buf = this.buf.slice(nl + 1);
					if (!line) continue;
					let msg: any;
					try {
						msg = JSON.parse(line);
					} catch {
						continue;
					}
					if (msg.id === 0) {
						// handshake
						if (msg.ready) {
							this.info = msg;
							this.lastError = null;
							if (!settled) {
								settled = true;
								resolveStart();
							}
						} else {
							this.lastError = msg.error || "scorer failed to start";
							if (!settled) {
								settled = true;
								rejectStart(new Error(this.lastError));
							}
						}
						continue;
					}
					const w = this.waiting.get(msg.id);
					if (w) {
						this.waiting.delete(msg.id);
						if (msg.error) w.reject(new Error(msg.error));
						else w.resolve(msg);
					}
				}
			});

			proc.stderr.setEncoding("utf8");
			proc.stderr.on("data", (d: string) => {
				const s = d.trim();
				if (s && /error|traceback|failed/i.test(s)) this.lastError = s.split("\n").pop() || s;
			});

			const die = (why: string) => {
				this.proc = null;
				this.info = null;
				for (const [, w] of this.waiting) w.reject(new Error(why));
				this.waiting.clear();
				if (!settled) {
					settled = true;
					rejectStart(new Error(this.lastError || why));
				}
			};
			proc.on("error", (err) => {
				this.lastError = err.message;
				die(err.message);
			});
			proc.on("exit", (code) => die(`scorer exited (code ${code})`));

			// A cold start downloads and loads up to ~2GB of weights.
			setTimeout(() => {
				if (!settled) {
					settled = true;
					rejectStart(new Error("scorer start timed out after 300s"));
				}
			}, 300_000).unref?.();
		}).finally(() => {
			this.starting = null;
		});

		return this.starting;
	}

	async score(text: string, timeoutMs = 30_000): Promise<{ p_deny: number; ms: number; tokens: number }> {
		await this.start();
		const proc = this.proc;
		if (!proc) throw new Error(this.lastError || "scorer not running");

		const id = this.nextId++;
		return new Promise((res, rej) => {
			const timer = setTimeout(() => {
				this.waiting.delete(id);
				rej(new Error(`scoring timed out after ${timeoutMs}ms`));
			}, timeoutMs);
			this.waiting.set(id, {
				resolve: (v) => {
					clearTimeout(timer);
					res(v);
				},
				reject: (e) => {
					clearTimeout(timer);
					rej(e);
				},
			});
			try {
				proc.stdin.write(`${JSON.stringify({ id, text })}\n`);
			} catch (err: any) {
				clearTimeout(timer);
				this.waiting.delete(id);
				rej(new Error(`could not write to scorer: ${err?.message ?? err}`));
			}
		});
	}

	stop(): void {
		if (!this.proc) return;
		try {
			this.proc.stdin.write(`${JSON.stringify({ id: this.nextId++, op: "shutdown" })}\n`);
		} catch {
			/* already gone */
		}
		const p = this.proc;
		this.proc = null;
		setTimeout(() => {
			try {
				p.kill();
			} catch {
				/* already gone */
			}
		}, 1000).unref?.();
	}
}

export default function (pi: ExtensionAPI) {
	let cfg = loadConfig();
	let scorer = new Scorer(cfg);
	let warnedUnavailable = false;
	const stats = { approved: 0, blocked: 0, errors: 0 };

	const modelShortName = () => cfg.model.replace(/^ProCreations\//, "");

	pi.on("tool_call", async (event: any, ctx: any) => {
		if (!cfg.enabled) return undefined;
		if (cfg.exemptTools.includes(event.toolName)) return undefined;

		const call = { tool: String(event.toolName), args: renderArgs(event.input) };
		const { userRequest, steps } = readSession(ctx, String(event.toolCallId ?? ""));
		const text = buildModelInput(call, userRequest, steps, cfg);

		let pDeny: number;
		try {
			const out = await scorer.score(text);
			pDeny = out.p_deny;
		} catch (err: any) {
			stats.errors++;
			const why = err?.message ?? String(err);
			if (cfg.failClosed) {
				return {
					block: true,
					reason:
						`Blocked by the automatic tool call review model (auto-pi): the reviewer could not run, ` +
						`and auto-pi is configured to fail closed. Reason: ${why}. ` +
						`Run /auto-install to set it up, or /auto off to disable review.`,
				};
			}
			// Fail open, but say so once — a silently dead safety gate is worse than none.
			if (!warnedUnavailable) {
				warnedUnavailable = true;
				ctx.ui?.notify?.(
					`auto-pi: review unavailable (${why}). Tool calls are running UNREVIEWED. ` +
						`Run /auto-install to fix, or /auto off to silence this.`,
					"warning",
				);
			}
			return undefined;
		}

		warnedUnavailable = false;

		if (pDeny > cfg.threshold) {
			stats.blocked++;
			const pct = (pDeny * 100).toFixed(1);
			ctx.ui?.notify?.(`auto-pi blocked ${call.tool} (P(deny)=${pDeny.toFixed(3)})`, "warning");
			return {
				block: true,
				reason:
					`BLOCKED BY THE AUTOMATIC TOOL CALL REVIEW MODEL (auto-pi).\n\n` +
					`This tool call was not executed. A local safety classifier (${modelShortName()}) ` +
					`reviewed it against the user's request and your recent actions, and judged it ` +
					`unsafe to run without human approval — confidence ${pct}%.\n\n` +
					`This is an automated review, not the user speaking, and it can be wrong.\n\n` +
					`What to do next:\n` +
					`- Do NOT simply retry the same call; it will be reviewed the same way.\n` +
					`- If there is a narrower, reversible, or better-scoped way to achieve the goal ` +
					`(smaller blast radius, an explicit path instead of a wildcard, a dry run first), ` +
					`do that instead.\n` +
					`- If the action genuinely is what the user asked for, stop and ask the user to ` +
					`confirm it explicitly, or to run it themselves.\n` +
					`- The user can allow this call by running /auto off, or by raising the threshold ` +
					`with /auto threshold <0-1>.`,
			};
		}

		stats.approved++;
		if (cfg.notifyOnApprove) {
			ctx.ui?.notify?.(`auto-pi approved ${call.tool} (P(deny)=${pDeny.toFixed(3)})`, "info");
		}
		return undefined;
	});

	pi.on("session_shutdown", () => scorer.stop());

	pi.registerCommand("auto", {
		description: "Automatic tool-call review: status, on/off, model, threshold",
		getArgumentCompletions: (prefix: string) => {
			const opts = ["status", "on", "off", "model", "threshold", "device", "fail-closed", "stats"];
			const items = opts.filter((o) => o.startsWith(prefix)).map((o) => ({ value: o, label: o }));
			return items.length ? items : null;
		},
		handler: async (args: string, ctx: any) => {
			const [sub, ...rest] = (args || "").trim().split(/\s+/);
			const value = rest.join(" ").trim();

			switch (sub) {
				case "":
				case "status": {
					const running = scorer.isRunning();
					const info = scorer.info;
					const lines = [
						`auto-pi ${cfg.enabled ? "ENABLED" : "disabled"}`,
						`model:     ${cfg.model}`,
						`threshold: ${cfg.threshold} (block when P(deny) > this)`,
						`scorer:    ${running ? `running on ${info?.device ?? "?"} (${info?.dtype ?? "?"}, ${info?.attn ?? "?"})` : "not started"}`,
						`on error:  ${cfg.failClosed ? "fail closed (block)" : "fail open (allow + warn)"}`,
						`session:   ${stats.approved} approved, ${stats.blocked} blocked, ${stats.errors} errors`,
					];
					if (scorer.lastError) lines.push(`last error: ${scorer.lastError}`);
					ctx.ui?.notify?.(lines.join("\n"), "info");
					return;
				}
				case "on":
					cfg.enabled = true;
					saveConfig(cfg);
					ctx.ui?.notify?.("auto-pi enabled — tool calls will be reviewed", "info");
					return;
				case "off":
					cfg.enabled = false;
					saveConfig(cfg);
					ctx.ui?.notify?.("auto-pi disabled — tool calls run unreviewed", "warning");
					return;
				case "stats":
					ctx.ui?.notify?.(
						`approved ${stats.approved}, blocked ${stats.blocked}, errors ${stats.errors}`,
						"info",
					);
					return;
				case "model": {
					if (!value) {
						ctx.ui?.notify?.(`model: ${cfg.model}\nchoices: ${Object.keys(MODELS).join(", ")}`, "info");
						return;
					}
					const resolved = MODELS[value] ?? value;
					cfg.model = resolved;
					saveConfig(cfg);
					scorer.stop();
					scorer = new Scorer(cfg);
					ctx.ui?.notify?.(`auto-pi model set to ${resolved} (restart on next call)`, "info");
					return;
				}
				case "threshold": {
					const n = Number.parseFloat(value);
					if (!Number.isFinite(n) || n <= 0 || n >= 1) {
						ctx.ui?.notify?.("threshold must be between 0 and 1, e.g. /auto threshold 0.5", "error");
						return;
					}
					cfg.threshold = n;
					saveConfig(cfg);
					ctx.ui?.notify?.(
						`threshold ${n} — lower blocks more (safer, more interruptions), higher blocks less`,
						"info",
					);
					return;
				}
				case "device": {
					if (!value) {
						ctx.ui?.notify?.(`device: ${cfg.device} (auto | cuda | mps | cpu)`, "info");
						return;
					}
					cfg.device = value;
					saveConfig(cfg);
					scorer.stop();
					scorer = new Scorer(cfg);
					ctx.ui?.notify?.(`device set to ${value}`, "info");
					return;
				}
				case "fail-closed": {
					cfg.failClosed = value !== "false" && value !== "off";
					saveConfig(cfg);
					ctx.ui?.notify?.(
						cfg.failClosed
							? "fail closed — calls are blocked when the reviewer cannot run"
							: "fail open — calls are allowed (with a warning) when the reviewer cannot run",
						"info",
					);
					return;
				}
				default:
					ctx.ui?.notify?.(
						"usage: /auto [status|on|off|stats|model <name>|threshold <0-1>|device <d>|fail-closed <bool>]",
						"info",
					);
			}
		},
	});

	pi.registerCommand("auto-install", {
		description: "Install the auto review model and its python dependencies",
		getArgumentCompletions: (prefix: string) => {
			const items = Object.keys(MODELS)
				.filter((m) => m.startsWith(prefix))
				.map((m) => ({ value: m, label: MODELS[m] }));
			return items.length ? items : null;
		},
		handler: async (args: string, ctx: any) => {
			const choice = (args || "").trim();
			const model = choice ? (MODELS[choice] ?? choice) : cfg.model;
			ctx.ui?.notify?.(
				`Installing ${model}.\nThis creates a python venv in ~/.pi/auto-pi and downloads the model ` +
					`(~2GB for auto-1b, ~0.8GB for auto-0.4b). Run this in a terminal:\n\n` +
					`  npx auto-pi install ${choice || "auto-1b"}\n\n` +
					`Then run /auto status here to confirm it is live.`,
				"info",
			);
			cfg.model = model;
			saveConfig(cfg);
		},
	});
}
