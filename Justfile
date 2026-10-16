set positional-arguments

# Opt in to the shared compiler cache with `just cargo=mbx <recipe>`.
cargo := "cargo"
# Backend feature set: `just features=metal test`, `just features=vulkan build`.
# Empty keeps the default (cpu).
features := ""
feature_flag := if features == "" { "" } else { "--features " + features }

# Display help
help:
    just -l

# Build every binary (debug profile).
build *args:
    {{cargo}} build {{feature_flag}} {{args}}

# Build every binary (release profile, fat LTO, stripped).
release *args:
    {{cargo}} build --release {{feature_flag}} {{args}}

# Sign the binaries that reach federation peers, so macOS can grant them
# Local Network access: `just features=metal sign "Apple Development: Name (TEAMID)"`.
sign identity:
    {{cargo}} build --release {{feature_flag}} --bin mdma-serve --bin mdma-fed
    for bin in mdma-serve mdma-fed; do \
        codesign --force --timestamp=none --sign "$1" --identifier "org.seuros.$bin" "target/release/$bin" && \
        codesign --verify --strict "target/release/$bin" && \
        codesign -dvv "target/release/$bin" 2>&1 | grep -E "^(Identifier|Authority|TeamIdentifier)"; \
    done

# Run mdma-serve (release). Pass env via the shell: MDMA_ADDR=... just serve
alias s := serve
serve *args:
    {{cargo}} run --release {{feature_flag}} --bin mdma-serve -- {{args}}

# Run mdma-web (debug) against a running mdma-serve. Needs MDMA_TLS_CERT/KEY/CA.
alias w := web
web *args:
    {{cargo}} run --bin mdma-web -- {{args}}

# Run the mdma CLI (debug).
alias m := mdma
mdma *args:
    {{cargo}} run --bin mdma -- {{args}}

# Run mdma-client (debug).
client *args:
    {{cargo}} run --bin mdma-client -- {{args}}

# Run mdma-bench (release).
bench *args:
    {{cargo}} run --release {{feature_flag}} --bin mdma-bench -- {{args}}

# Install all binaries into ~/.cargo/bin (release + target-cpu=native).
install:
    RUSTFLAGS="-C target-cpu=native" {{cargo}} install --path . --locked --force {{feature_flag}}

# Format code
fmt:
    cargo fmt

# Clippy on every target, deny warnings
clippy:
    {{cargo}} clippy --all-targets {{feature_flag}} -- -D warnings

# Check compilation without building
check:
    {{cargo}} check --all-targets {{feature_flag}}

# Run tests (release: the kernel tests are too slow in debug)
test *args:
    {{cargo}} nextest run --release --no-fail-fast {{feature_flag}} "$@"

# Lint + check + clippy (no tests)
qq: fmt check clippy

# Full QA: qq + rust tests + web tests, then drop stale build artifacts
qa: qq test web-test sweep

# Fix clippy warnings automatically
fix:
    {{cargo}} clippy --fix --all-targets {{feature_flag}} --allow-dirty

# List outdated deps via cargo-edit
outdated:
    cargo upgrade --dry-run

# ---- web/ (three.js stage bundled into src/bin/mdma-web/assets/scene.js) ----

# Install front-end deps
web-install:
    cd web && bun install

# Run the front-end unit tests
web-test:
    cd web && bun test

# Federation view on a stub fleet with the shipped bundle: open the printed URL, add &federated=1.
web-preview: web-build
    cd web && bun serve-preview.js

# Rebuild scene.js. mdma-web embeds it with include_str!, so rebuild the bin after.
web-build: web-install
    cd web && bun run build

# Full web cycle: deps, tests, bundle, then rebuild mdma-web
web-all: web-install web-test web-build
    {{cargo}} build --bin mdma-web

# Drop stale build artifacts Cargo never removes: old toolchains', and unused for 2 weeks (cargo install cargo-sweep)
sweep:
    cargo sweep --installed
    cargo sweep --time 14

# Remove build output and the repo-local temp root.
clean:
    rm -rf target .tmp web/node_modules
