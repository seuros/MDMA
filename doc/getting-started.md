# Getting started: three 16 GB machines, one private fabric

You own a MacBook with an M1 and 16 GB, a Linux PC with 16 GB, and a FreeBSD
home server with 16 GB. Each can run one mid-size model on its own, and each
model sits on its own machine. mdma runs a model on whatever hardware each
machine has, tells you before loading whether it fits, and lets the Mac use all
three over mutually authenticated TLS: no cloud, no shared account, no central
scheduler. By the end, one browser tab on the Mac reaches three models on three
operating systems.

| Machine | Backend | Model | File |
|---|---|---|---|
| MacBook, M1, 16 GB | Metal (GPU) | Gemma 4 E2B instruct | 3.1 GB |
| Linux PC, 16 GB | CPU | Qwen3.5 9B | 5.7 GB |
| FreeBSD server, 16 GB | CPU | Gemma 4 E4B instruct | 5.0 GB |

The examples use `192.168.1.10` (Mac), `192.168.1.11` (Linux) and
`192.168.1.12` (FreeBSD); use your own addresses. Every machine keeps its files
in `~/mdma`.

**Tested** on 2026-10-14 with mdma 0.4.0 and again on 2026-10-15 with 0.5.0,
both installed from crates.io with mise's Rust 1.99, on an M1 Max (macOS 15.7),
an Intel Core i7-12700H (Arch Linux) and an AMD Ryzen 7 5800H (FreeBSD
16.0-CURRENT). Each server's tensor budget was capped at 12 GiB to stand in for
a 16 GB machine. The old front door below ran the FreeBSD build, copied over,
on a 2011 Intel Core i5-2415M with 16 GB.

## Why not one big GPU?

A GPU with 48 to 96 GB costs thousands and spends most of its life idle; at
home you will rarely earn it back. mdma starts from what you already have and
borrows the rest when you need it.

- **The front door can be old, and empty.** We ran it on a 2011 Intel Core
  i5-2415M with 16 GB, on FreeBSD, with an empty models folder. It
  authenticated the client and routed the Qwen3.5 9B and Gemma 4 E4B requests
  to the Linux and FreeBSD machines; it never loads a model itself.
- **Workers come and go.** With the Linux machine switched off, its route
  failed within a second (`Unavailable`) while the FreeBSD model kept
  answering. The first request after the Linux machine came back succeeded.
- **Rent the big one for an afternoon.** A rented GPU server, an H200 say,
  joins like any other machine: install, `mdma-fed init`, exchange bundles,
  route, use it, stop it. Keep its `fed` directory between rentals or give
  each rental its own id: a known id with new keys is refused. The rented
  machine sees the prompts you route to it, and only those. The CUDA backend
  is tested on an RTX 3070 Ti laptop GPU, not yet on rented datacenter GPUs.

Each request still runs whole on one machine. mdma does not split one model
across machines, so a model has to fit the machine that runs it.

## 1. Install

mdma needs Rust 1.99 or newer. Distribution packages lag behind (FreeBSD's
and Arch's `rust` were at 1.97), so [mise](https://mise.jdx.dev) provides it
for the build without touching the system's Rust. Everything goes in `~/mdma`.

macOS (the Command Line Tools provide the compiler and linker):

```sh
xcode-select --install
curl https://mise.run | sh
~/.local/bin/mise exec rust@1.99 -- cargo install --locked --root ~/mdma mdma --features metal
```

Linux needs mise, a C toolchain and CMake: `pacman -S mise base-devel cmake`
on Arch; elsewhere `build-essential cmake` from the distribution and
`curl https://mise.run | sh` (mise is then `~/.local/bin/mise`).

```sh
mise exec rust@1.99 -- cargo install --locked --root ~/mdma mdma
```

FreeBSD:

```sh
pkg install mise protobuf cmake
PROTOC=/usr/local/bin/protoc mise exec rust@1.99 -- cargo install --locked --root ~/mdma mdma
```

Then, on each machine:

```sh
export PATH=$HOME/mdma/bin:$PATH
mdma doctor
```

The Mac reports `metal: ready`, the other two `cpu: ready`. Building takes
a few minutes (about six on the FreeBSD machine).

## 2. Get the models

```sh
mkdir -p ~/mdma/models && cd ~/mdma/models
# Mac
curl -LO https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf
# Linux
curl -LO https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
# FreeBSD
fetch https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf
```

Only the model file is needed, not the `mmproj` vision projectors next to it.
The optional qwen35 checkpoint gate uses a separate pinned fixture; the
Qwen3.5-9B file downloaded here is a runnable example, but it is not accepted
by that exact-token test unless its SHA-256 matches the pinned value.

## 3. Will it fit?

`mdma plan` reads the file's metadata, without loading weights, and adds up
weights, KV cache and scratch for a context and prompt size:

```console
linux$ mdma plan ~/mdma/models/Qwen3.5-9B-Q4_K_M.gguf --ctx 4096 --prompt-tokens 512 --ram-bytes 17179869184
Model: /home/you/mdma/models/Qwen3.5-9B-Q4_K_M.gguf
Backend: cpu
Fits: true
Host required: 6503909376 bytes / budget: 17179869184 bytes
```

The same model on a machine with 4 GiB to spare:

```console
linux$ mdma plan ~/mdma/models/Qwen3.5-9B-Q4_K_M.gguf --ctx 4096 --prompt-tokens 512 --ram-bytes 4294967296
Fits: false
Host required: 6503909376 bytes / budget: 4294967296 bytes
Error: "model/request does not fit the supplied budgets"
```

The servers enforce the same accounting: a request that would not fit is
refused with `RESOURCE_EXHAUSTED` before anything is allocated, and the server
keeps running.

## 4. A first answer on each machine

`mdma run` loads a model and generates, without a server. Prompts are raw
text, so they carry the model's chat markers (`$'…'` turns `\n` into
newlines in zsh, bash and FreeBSD's `sh`):

```sh
# Mac
mdma run ~/mdma/models/gemma-4-E2B-it-Q4_K_M.gguf --backend metal --max-tokens 96 \
  $'<|turn>user\nWhy is the sky blue? Answer in two sentences.<turn|>\n<|turn>model\n'
# FreeBSD
mdma run ~/mdma/models/gemma-4-E4B-it-Q4_K_M.gguf --max-tokens 96 \
  $'<|turn>user\nWhy is the sky blue? Answer in two sentences.<turn|>\n<|turn>model\n'
# Linux; the empty think block asks Qwen3.5 to answer without reasoning first
mdma run ~/mdma/models/Qwen3.5-9B-Q4_K_M.gguf --max-tokens 96 \
  $'<|im_start|>user\nWhy is the sky blue? Answer in two sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

> **Why the `<|turn>` markers?** mdma is early. `mdma run` sends your prompt
> exactly as written, so you add the model's chat markers yourself. The
> servers in the rest of this guide apply each model's chat format for you
> (`MDMA_CHAT=1`). Rough edges like this are where
> [help is welcome](https://github.com/seuros/mdma/issues).

Each prints its answer, for example:

```text
The sky appears blue because Earth's atmosphere scatters shorter wavelengths of sunlight, such as blue and violet, more effectively than longer wavelengths like red. This phenomenon, known as Rayleigh scattering, causes the blue light to spread out in all directions, making the sky look blue to our eyes.
```

`mdma-bench MODEL 128 16 3` measures prefill and decode speed
(`MDMA_BACKEND=metal` on the Mac). On the tested machines the CPU servers
decoded at about 1.1 tokens/s (Qwen3.5 9B, i7-12700H) and 1.5 tokens/s
(Gemma 4 E4B, Ryzen 7 5800H): fine for background work, slow for chat. The
Mac's GPU is the interactive one. The CPU decoders are written for
correctness first; their speed is work in progress.

## 5. Introduce the machines to each other

Each machine gets a federation identity: its name, where peers reach it, and
signing keys. On each machine, in `~/mdma`:

```sh
cd ~/mdma
mdma-certs                                   # ./certs: a local CA, server and client certificates
mdma-fed init fed --id mac.home --endpoint https://192.168.1.10:6365       # Mac
mdma-fed init fed --id linux.home --endpoint https://192.168.1.11:6365     # Linux
mdma-fed init fed --id freebsd.home --endpoint https://192.168.1.12:6365   # FreeBSD
mdma-fed export fed > mac.home.json          # likewise linux.home.json, freebsd.home.json
```

`export` prints the bundle's digest. Copy the bundles across (scp, a USB
stick, anything) so the Mac has the Linux and FreeBSD bundles and both of them
have the Mac's:

```sh
# on the Mac
scp mac.home.json 192.168.1.11:mdma/ && scp mac.home.json 192.168.1.12:mdma/
scp 192.168.1.11:mdma/linux.home.json . && scp 192.168.1.12:mdma/freebsd.home.json .
```

On Linux and FreeBSD, let the Mac see and run the models in the models folder:

```sh
mdma-fed enroll fed mac.home.json --execute all --models models
```

```text
enrolled mac.home at https://192.168.1.10:6365 from bundle sha256:4478…
  sees Qwen3.5-9B-Q4_K_M
  runs Qwen3.5-9B-Q4_K_M
```

On the Mac, trust both machines without granting them anything:

```sh
mdma-fed enroll fed linux.home.json
mdma-fed enroll fed freebsd.home.json
```

Each `enroll` prints the digest it read; it must match what `export` printed on
the other machine. Trust is explicit and one-way: the Linux machine runs the
Mac's requests, the Mac runs nothing for it. Seeing a model, running it,
routing to it and sending your prompt to it are separate grants.

`mdma-fed` changes a node only while its server is stopped.

## 6. Start the Linux and FreeBSD servers

On each, in `~/mdma`:

```sh
MDMA_MODELS_DIR=$HOME/mdma/models \
MDMA_FEDERATION_CONFIG=$HOME/mdma/fed/runtime.toml \
MDMA_TLS_CERT=certs/server.pem MDMA_TLS_KEY=certs/server-key.pem MDMA_TLS_CA=certs/ca.pem \
MDMA_NATIVE_TENSOR_BYTES=12884901888 \
  mdma-serve
```

```text
mdma native tensor budget: 0/12884901888 bytes reserved (not a process/VRAM cap)
mdma model Qwen3.5-9B-Q4_K_M: gguf qwen35 (5417 MB on disk)
mdma models load on first use (1 warm-up rounds each)
mdma listening on 127.0.0.1:6362 (mTLS required, backend=cpu)
mdma federation listening on 0.0.0.0:6365 (strict mTLS, execution=true)
```

`MDMA_NATIVE_TENSOR_BYTES` (12 GiB here) is what this machine lends to models;
the rest stays for the operating system and its other work. Ordinary clients
reach the server only on `127.0.0.1:6362`; the Mac comes in on port 6365, with
its own certificate. One `mdma-serve` runs per machine; a second refuses to
start.

## 7. Route from the Mac, then start it

Run the Mac's commands from Terminal or over SSH: macOS lets programs started
there reach the local network. A program started from another app, such as an
editor's built-in terminal, counts as that app and cannot reach the other
machines unless that app is allowed.

With the Mac's server not running yet, read each peer's signed catalog and
route its models for the Mac's client certificate:

```sh
cd ~/mdma
mdma-fed routes fed --peer linux.home --client-cert certs/client.pem
mdma-fed routes fed --peer freebsd.home --client-cert certs/client.pem
```

```text
linux-Qwen3.5-9B-Q4_K_M -> linux.home Qwen3.5-9B-Q4_K_M
freebsd-gemma-4-E4B-it-Q4_K_M -> freebsd.home gemma-4-E4B-it-Q4_K_M
```

Each route is pinned to one machine and one exact model file. Then start the
Mac's server, on its GPU:

```sh
MDMA_BACKEND=metal \
MDMA_MODELS_DIR=$HOME/mdma/models \
MDMA_FEDERATION_CONFIG=$HOME/mdma/fed/runtime.toml \
MDMA_TLS_CERT=certs/server.pem MDMA_TLS_KEY=certs/server-key.pem MDMA_TLS_CA=certs/ca.pem \
MDMA_NATIVE_TENSOR_BYTES=12884901888 \
  mdma-serve
```

```text
mdma: metal device Apple M1 Max
mdma listening on 127.0.0.1:6362 (mTLS required, backend=metal)
mdma federation catalog poll peer=freebsd.home verified=true category=none io_kind=None
mdma federation catalog poll peer=linux.home verified=true category=none io_kind=None
```

To run it as a service instead, see the [notes](#notes).

## 8. Use all three from the Mac

In a second Terminal on the Mac, in `~/mdma`:

```sh
# MDMA_CHAT=1 puts the question in each model's own chat format
export MDMA_TLS_CERT=certs/client.pem MDMA_TLS_KEY=certs/client-key.pem MDMA_TLS_CA=certs/ca.pem MDMA_CHAT=1

# the Mac's own model
MDMA_MODEL=gemma-4-E2B-it-Q4_K_M mdma-client "Why is the sky blue? Answer in two sentences."

# the Linux machine's model
MDMA_MODEL=linux-Qwen3.5-9B-Q4_K_M MDMA_RPC_TIMEOUT_MS=300000 MDMA_MAX_TOKENS=300 \
  mdma-client "Why is the sky blue? Answer in two sentences."

# the FreeBSD machine's model
MDMA_MODEL=freebsd-gemma-4-E4B-it-Q4_K_M MDMA_RPC_TIMEOUT_MS=300000 MDMA_MAX_TOKENS=120 \
  mdma-client "Why is the sky blue? Answer in two sentences."
```

The routed models get the same chat formatting as the Mac's own: the Mac wraps
the question in the format each machine's signed catalog names, then forwards
it. A routed call needs a deadline (`MDMA_RPC_TIMEOUT_MS`, at most five
minutes). Qwen3.5 reasons before it answers and prints that block first, here
empty:

```text
<think>

</think>

Mercury, Venus, and Earth are three planets in our solar system.
```

The first call to a model loads it on its machine, which takes longer; later
calls find it resident.

For a browser, start the web UI:

```sh
mdma-web
```

```text
mdma-web listening on http://127.0.0.1:6364/?token=…
mdma-web: connected to server (state=serving, 3 models)
```

Open the printed URL. The **Playground** lists all three models; pick one,
turn chat on, write a prompt and watch the reply stream, from the Mac or from
the other machines alike. **Memory & placement** shows what each model holds
in the Mac's budget.

## What you just built

- Three operating systems and two kinds of hardware serve one client. Each
  machine runs the model that suits it: the Mac's GPU handles interactive chat,
  and the other two machines lend their RAM and CPU to bigger models.
- Every hop is TLS 1.3 with mutual authentication and pinned keys. There is no
  shared certificate authority: each machine made its own, and the machines
  trust each other's identities, not a third party.
- Each machine decides what others may see and run, and the Mac decides which
  of its clients may use which route. Nothing is shared until you grant it.
- Memory is admitted before it is used. Restarting the FreeBSD server with a
  4 GiB budget, smaller than Gemma 4 E4B's weights, made the Mac's routed call
  fail with `ResourceExhausted` before anything loaded, and the server kept
  serving.
- A request runs whole on one machine. If it fails, mdma reports the failure;
  it never retries the request somewhere else behind your back.

## Notes

- **Stopping and changing**: stop a server with Ctrl-C. `mdma-fed status fed`
  shows peers, grants and routes, also while it runs. After a peer's models
  change, stop the Mac's server, run `mdma-fed routes` again and restart it,
  or have the running server import them with `mdma-client --routes PEER`
  from a client pinned in its `MDMA_ADMIN_SPKI`.
- **Machines you do not log into**: enroll a peer with `--install 20G
  --installed` and the node may start with an empty models folder; push GGUFs
  to it with `mdma-fed push`, or from a running server with `mdma-client
  --push`. See
  [pushing models](federation/routing-runtime.md#pushing-models-to-a-peer).
- **Certificates expire** after 30 days by default (`mdma-fed init --days`, up
  to 397). Renewing is a new `init`, then `mdma-fed enroll --replace` of the
  new bundle on each peer.
- **macOS services**: Local Network privacy remembers its answer per signed
  binary, and a program started from another app counts as that app. To run
  `mdma-serve` under launchd, sign it with an Apple-issued identity; see the
  [routing runtime](federation/routing-runtime.md#deployment-notes).
- **Other models**: Llama-family models and Gemma 4 run on Metal; Qwen2/3,
  GLM-4 and Qwen3.5 run on CPU there. See [architectures](architectures.md) for
  what each backend supports, [serving](serving.md) for every server setting,
  and [federation](federation.md) for the protocol.
