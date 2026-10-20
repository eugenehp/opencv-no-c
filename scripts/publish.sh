#!/usr/bin/env bash
# scripts/publish.sh — publish every opencv-no-c crate to crates.io.
#
# Modelled on rlx's scripts/publish.sh, with one structural difference: the
# publish order is derived from `cargo metadata` (a topological sort of the
# workspace's own dependency edges) instead of a hand-kept tier list, so a
# new crate or a new edge cannot leave the order stale.
#
# What keeps it safe to leave running unattended:
#
#   * Rate-limit floor: `MIN_INTERVAL` seconds (default 65) between uploads.
#     crates.io allows roughly one new crate / new version per minute after a
#     short burst; all 18 crates here are new on the first release.
#   * Index readiness: after each upload, poll the sparse index
#     (https://index.crates.io/...) until the new version is served. Until
#     then the next crate's `cargo publish` cannot resolve it.
#   * Skip published: a crate whose local version is already on the index is
#     skipped, so a re-run resumes where the last one stopped.
#   * Retries: 429 / 408 / 5xx / network errors sleep until crates.io's
#     "try again after <GMT>" (+ pad) when given, else a status-specific
#     backoff, then retry the same crate. "No matching package" for a sibling
#     waits for that sibling to appear on the index, then retries.
#
# Pre-flight gates (skip with --no-gate):
#
#   1. Fresh resolve, and rlx must come from crates.io. The lockfile is
#      regenerated so the gates build what a new user would resolve.
#      /Users/Shared/.cargo/config.toml patches every rlx* crate to a local
#      checkout, and cargo finds that file by walking up from the *working
#      directory*. So every cargo call here runs from `/` with
#      --manifest-path, outside its reach; the check fails if any rlx* crate
#      still resolves to a path.
#   2. cargo clippy --workspace --all-targets -- -D warnings
#   3. cargo check -p opencv-no-c-accel --features rlx,onnx (the RLX path is
#      off by default, so nothing else compiles it)
#   4. cargo test --workspace --release --no-fail-fast
#
#   `cargo fmt --check` is not a gate: the tree is not rustfmt-formatted.
#
# Usage:
#
#   scripts/publish.sh --list                  # publish order + index status
#   scripts/publish.sh --plan                  # only crates needing publish
#   scripts/publish.sh --dry-run               # cargo publish --workspace --dry-run
#   scripts/publish.sh                         # gates, confirm, publish
#   scripts/publish.sh --yes                   # no confirm prompt
#   scripts/publish.sh --no-gate               # skip the pre-flight gates
#   scripts/publish.sh --start-crate NAME      # resume from NAME
#   scripts/publish.sh --min-interval 90       # rate-limit floor (sec)
#   scripts/publish.sh --poll-interval 10      # index poll cadence (sec)
#   scripts/publish.sh --poll-timeout 600      # index poll cap per crate (sec)
#   scripts/publish.sh --max-retries 5         # cap unparsed backoffs (0 = unlimited)
#   scripts/publish.sh --rate-limit-pad 15     # pad after server retry-after (sec)
#   scripts/publish.sh --no-verify             # skip cargo's packaged rebuild
#
# Dry run: `cargo publish --workspace --dry-run` packages every crate and
# builds each against its siblings' packaged tarballs, so unlike a per-crate
# dry run it validates the non-leaf crates before anything is on crates.io.
#
# Environment: CARGO_REGISTRY_TOKEN, or run `cargo login` first.

set -euo pipefail

DRY_RUN=0
LIST_ONLY=0
PLAN_ONLY=0
NO_GATE=0
NO_VERIFY=0
ASSUME_YES=0
MIN_INTERVAL=65
POLL_INTERVAL=10
POLL_TIMEOUT=600
MAX_RETRIES=0
RATE_LIMIT_PAD=15
START_CRATE=""

usage() {
    local last
    last="$(grep -n -m1 '^set -euo pipefail' "$0" | cut -d: -f1)"
    sed -n "2,$((last - 2))p" "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while (( $# > 0 )); do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --list)           LIST_ONLY=1; shift ;;
        --plan)           PLAN_ONLY=1; shift ;;
        --no-gate)        NO_GATE=1; shift ;;
        --no-verify)      NO_VERIFY=1; shift ;;
        --yes|-y)         ASSUME_YES=1; shift ;;
        --min-interval)   MIN_INTERVAL="$2"; shift 2 ;;
        --poll-interval)  POLL_INTERVAL="$2"; shift 2 ;;
        --poll-timeout)   POLL_TIMEOUT="$2"; shift 2 ;;
        --max-retries)    MAX_RETRIES="$2"; shift 2 ;;
        --rate-limit-pad) RATE_LIMIT_PAD="$2"; shift 2 ;;
        --start-crate)    START_CRATE="$2"; shift 2 ;;
        --help|-h)        usage ;;
        *)
            echo "unknown arg: $1 (run with --help)" >&2
            exit 2
            ;;
    esac
done

cd "$(dirname "$0")/.."
ROOT="$PWD"
MANIFEST="$ROOT/Cargo.toml"

# Cargo reads .cargo/config.toml from the working directory upward, not from
# the manifest's location. Running from `/` keeps a parent config's [patch]
# (see gate 1) out of every resolve, build and publish below. The subcommand
# goes first so --manifest-path lands before any `--` passthrough.
cargo_ws() {
    local sub="$1"; shift
    (cd / && cargo "$sub" --manifest-path "$MANIFEST" "$@")
}

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

for tool in cargo curl python3; do
    command -v "$tool" >/dev/null 2>&1 || { red "$tool not found"; exit 1; }
done

# ── Publish order ───────────────────────────────────────────────
# One "name<TAB>version" line per publishable member, dependencies first.
# Every intra-workspace edge counts, dev-dependencies included: cargo keeps a
# dev-dep that carries a version in the published manifest and resolves it.
# Ties break alphabetically so the order is stable across runs.
PLAN_FILE="$(mktemp)"
trap 'rm -f "$PLAN_FILE"' EXIT
cargo_ws metadata --format-version 1 --no-deps | python3 -c '
import json, sys
m = json.load(sys.stdin)
members = set(m["workspace_members"])
pkgs = {p["name"]: p for p in m["packages"] if p["id"] in members}
# `publish = false` comes through as an empty registry list.
pub = {n: p for n, p in pkgs.items() if p.get("publish") != []}
deps = {n: {d["name"] for d in p["dependencies"] if d["name"] in pub and d["name"] != n}
        for n, p in pub.items()}
for n, p in pub.items():
    for d in p["dependencies"]:
        if d["name"] in pkgs and d["name"] not in pub and d.get("kind") != "dev":
            sys.exit(n + " depends on " + d["name"] + ", which has publish = false")
order, done = [], set()
while len(order) < len(pub):
    ready = sorted(n for n in pub if n not in done and deps[n] <= done)
    if not ready:
        sys.exit("dependency cycle among: " + ", ".join(sorted(set(pub) - done)))
    order.append(ready[0]); done.add(ready[0])
for n in order:
    print(n + "\t" + pub[n]["version"])
' > "$PLAN_FILE"

CRATES=()
while IFS=$'\t' read -r name _; do
    CRATES+=("$name")
done < "$PLAN_FILE"

crate_version() {
    awk -F'\t' -v c="$1" '$1 == c { print $2; exit }' "$PLAN_FILE"
}

if [[ -n "$START_CRATE" && -z "$(crate_version "$START_CRATE")" ]]; then
    red "--start-crate $START_CRATE is not a publishable workspace crate"
    exit 2
fi

# ── Sparse index ────────────────────────────────────────────────
sparse_index_path() {
    local name
    name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    local n="${#name}"
    if   (( n == 1 )); then printf '1/%s\n'     "$name"
    elif (( n == 2 )); then printf '2/%s\n'     "$name"
    elif (( n == 3 )); then printf '3/%s/%s\n'  "${name:0:1}" "$name"
    else                    printf '%s/%s/%s\n' "${name:0:2}" "${name:2:2}" "$name"
    fi
}

# 0 = version served, 1 = crate known but not this version, 2 = crate unknown.
check_index() {
    local body
    body="$(curl -fsS --max-time 10 "https://index.crates.io/$(sparse_index_path "$1")" 2>/dev/null || true)"
    [[ -z "$body" ]] && return 2
    printf '%s' "$body" | grep -Fq "\"vers\":\"$2\"" && return 0
    return 1
}

on_crates_io() { check_index "$1" "$2"; }

list_order() {
    bold "Publish order (dependencies first):"
    local i=0 c v status
    for c in "${CRATES[@]}"; do
        v="$(crate_version "$c")"
        if on_crates_io "$c" "$v"; then status="on crates.io"; else status="needs publish"; fi
        i=$(( i + 1 ))
        printf "  %2d. %-24s %-8s (%s)\n" "$i" "$c" "$v" "$status"
    done
}

print_plan() {
    local c v n=0
    for c in "${CRATES[@]}"; do
        v="$(crate_version "$c")"
        if ! on_crates_io "$c" "$v"; then
            (( n == 0 )) && bold "Crates needing publish:"
            echo "  - $c@$v"
            n=$(( n + 1 ))
        fi
    done
    (( n == 0 )) && green "Nothing to publish — every crate is on crates.io at its local version."
    return 0
}

if (( LIST_ONLY )); then list_order; exit 0; fi
if (( PLAN_ONLY )); then print_plan; exit 0; fi

# ── Pre-flight gates ────────────────────────────────────────────
check_rlx_source() {
    # --all-features, or the optional rlx deps are not in the graph at all.
    # A path source for any rlx* package means a [patch] redirected it.
    local meta local_rlx
    # Cargo.lock is gitignored, and a build run inside /Users/Shared writes
    # the patched versions into it; re-resolve from scratch every time.
    cargo_ws update --quiet
    if ! meta="$(cargo_ws metadata --format-version 1 --all-features 2>&1 >/dev/null)"; then
        red "cargo cannot resolve the rlx features from crates.io:"
        printf '%s\n' "$meta" | head -12
        exit 1
    fi
    local_rlx="$(cargo_ws metadata --format-version 1 --all-features 2>/dev/null | python3 -c '
import json, sys
m = json.load(sys.stdin)
for p in m["packages"]:
    if p["name"].startswith("rlx") and p["source"] is None:
        print("  " + p["name"] + " " + p["version"] + " -> " + p["manifest_path"])
')"
    if [[ -n "$local_rlx" ]]; then
        red "rlx resolves to a local checkout, not crates.io:"
        printf '%s\n' "$local_rlx"
        red "Something still patches it even from \`/\` — check \$CARGO_HOME/config.toml"
        red "and CARGO_* environment overrides. The gates would not test the"
        red "published dependency."
        exit 1
    fi
}

if (( ! NO_GATE )); then
    bold "[1/4] fresh resolve; rlx comes from crates.io"
    check_rlx_source
    bold "[2/4] cargo clippy --workspace --all-targets -- -D warnings"
    cargo_ws clippy --workspace --all-targets -- -D warnings
    bold "[3/4] cargo check -p opencv-no-c-accel --features rlx,onnx"
    cargo_ws check -p opencv-no-c-accel --features rlx,onnx
    # --no-fail-fast: a release gate wants every crate's result in one run.
    bold "[4/4] cargo test --workspace --release --no-fail-fast"
    cargo_ws test --workspace --release --no-fail-fast
    green "Pre-flight gates passed."
fi

# ── Dry run ─────────────────────────────────────────────────────
if (( DRY_RUN )); then
    list_order
    echo
    args=(--workspace --dry-run --allow-dirty)
    (( NO_VERIFY )) && args+=(--no-verify)
    bold ">> cargo publish ${args[*]}"
    cargo_ws publish "${args[@]}"
    green "Dry run passed: every crate packages and builds against its siblings."
    exit 0
fi

# ── Confirmation ────────────────────────────────────────────────
list_order
echo
print_plan
echo
yellow "Mode: REAL PUBLISH — uploads to crates.io. Published versions cannot be deleted."
yellow "Rate-limit floor:  ${MIN_INTERVAL}s between uploads."
yellow "Index polling:     every ${POLL_INTERVAL}s, cap ${POLL_TIMEOUT}s per crate."
if (( MAX_RETRIES == 0 )); then
    yellow "Registry retries:  automatic, unlimited."
else
    yellow "Registry retries:  up to ${MAX_RETRIES} unparsed backoffs per crate."
fi
[[ -z "${CARGO_REGISTRY_TOKEN:-}" ]] && yellow "CARGO_REGISTRY_TOKEN not set — relying on \`cargo login\` credentials."
echo

if (( ! ASSUME_YES )); then
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        red "Aborted."
        exit 1
    fi
fi

# ── Waiting ─────────────────────────────────────────────────────
sleep_with_countdown() {
    local remaining="$1" label="$2"
    while (( remaining > 0 )); do
        printf "\r  %s — %4ds remaining " "$label" "$remaining"
        sleep 1
        remaining=$(( remaining - 1 ))
    done
    printf "\r  %s — done.                   \n" "$label"
}

wait_for_index() {
    local crate="$1" version="$2" elapsed=0
    yellow "  polling the sparse index for $crate@$version (every ${POLL_INTERVAL}s, cap ${POLL_TIMEOUT}s)..."
    while (( elapsed < POLL_TIMEOUT )); do
        if check_index "$crate" "$version"; then
            green "  index ready after ${elapsed}s — $crate@$version resolvable."
            return 0
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$(( elapsed + POLL_INTERVAL ))
    done
    yellow "  index did not serve $crate@$version within ${POLL_TIMEOUT}s — continuing;"
    yellow "  a dependent's publish will wait for it if needed."
}

# ── Error classification ────────────────────────────────────────
http_status_of() {
    grep -oiE 'status [0-9]{3}' "$1" 2>/dev/null | tail -1 | grep -oE '[0-9]{3}' || true
}

is_already_uploaded() {
    grep -qiE 'already exists on crates\.io|already exists on the registry|version `[^`]+` is already uploaded' "$1"
}

is_missing_dep() {
    grep -qE 'no matching package named|location searched: crates\.io index' "$1"
}

missing_dep_name() {
    grep -oE 'no matching package named `[^`]+`' "$1" 2>/dev/null | head -1 \
        | sed -E 's/no matching package named `([^`]+)`/\1/'
}

is_retryable() {
    case "$(http_status_of "$1")" in
        408|429|500|502|503|504) return 0 ;;
    esac
    grep -qiE 'Too Many Requests|rate.?limit|published too many|try again after|temporarily unavailable|service unavailable|bad gateway|gateway timeout|timed out|connection reset|connection refused|unexpected eof|broken pipe|error sending request|dns error' "$1"
}

http_date_to_epoch() {
    if [[ "$(uname -s)" == Darwin ]]; then
        date -j -u -f '%a, %d %b %Y %H:%M:%S GMT' "$1" '+%s' 2>/dev/null
    else
        date -u -d "$1" '+%s' 2>/dev/null
    fi
}

# Prints "<seconds> <parsed>": parsed=1 when the wait came from the server or
# a recognised status, so the caller should not escalate its own backoff.
retry_wait() {
    local log="$1" status when epoch wait
    status="$(http_status_of "$log")"
    when="$(grep -oiE 'try again after [A-Za-z]{3}, [0-9]+ [A-Za-z]+ [0-9]+ [0-9:]+ GMT' "$log" 2>/dev/null \
        | tail -1 | sed -E 's/^[Tt]ry again after //')"
    if [[ -n "$when" ]] && epoch="$(http_date_to_epoch "$when")" && [[ -n "$epoch" ]]; then
        wait=$(( epoch - $(date -u '+%s') + RATE_LIMIT_PAD ))
        (( wait < MIN_INTERVAL )) && wait=$MIN_INTERVAL
        (( wait > 86400 )) && wait=86400
        echo "$wait 1"; return
    fi
    case "$status" in
        429) echo "$(( MIN_INTERVAL * 2 )) 1" ;;
        408) echo "60 1" ;;
        500|502|503|504) echo "90 1" ;;
        *) echo "$MIN_INTERVAL 0" ;;
    esac
}

# ── Publish ─────────────────────────────────────────────────────
# Returns 0 when uploaded, 2 when crates.io already has the version.
publish_one() {
    local crate="$1" attempt=1 backoff=$MIN_INTERVAL log rc wait parsed missing
    local args=(--package "$crate")
    (( NO_VERIFY )) && args+=(--no-verify)
    while true; do
        bold ">> cargo publish ${args[*]}"
        log="$(mktemp)"
        # cargo's exit code alone is not trusted: classify from its output.
        set +e
        cargo_ws publish "${args[@]}" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        set -e

        if is_already_uploaded "$log"; then
            rm -f "$log"; return 2
        fi
        if (( rc == 0 )); then
            rm -f "$log"; return 0
        fi
        if is_missing_dep "$log"; then
            missing="$(missing_dep_name "$log")"
            rm -f "$log"
            if [[ -n "$missing" && -n "$(crate_version "$missing")" ]]; then
                yellow "  $crate needs $missing on the index first — waiting, then retrying."
                wait_for_index "$missing" "$(crate_version "$missing")"
                sleep_with_countdown "$POLL_INTERVAL" "index settle"
                continue
            fi
            red "Publish blocked for $crate: a dependency is not on crates.io (${missing:-unknown})."
            exit 1
        fi
        if is_retryable "$log"; then
            read -r wait parsed < <(retry_wait "$log")
            rm -f "$log"
            if (( parsed )); then
                yellow "  registry asked $crate to wait — ${wait}s, then retry."
                attempt=1; backoff=$MIN_INTERVAL
            else
                if (( MAX_RETRIES > 0 && attempt > MAX_RETRIES )); then
                    red "Publish failed for $crate after $MAX_RETRIES retries."
                    red "Resume with: scripts/publish.sh --start-crate $crate"
                    exit 1
                fi
                wait=$backoff
                yellow "  transient error for $crate (attempt $attempt) — backoff ${wait}s."
                backoff=$(( backoff * 2 )); (( backoff > 600 )) && backoff=600
                attempt=$(( attempt + 1 ))
            fi
            sleep_with_countdown "$wait" "crates.io cooldown"
            continue
        fi
        red "Publish failed for $crate (exit $rc, not retryable). Log: $log"
        red "Resume with: scripts/publish.sh --start-crate $crate"
        exit 1
    done
}

PUBLISHED=()
SKIPPED=()
uploaded=0
resuming=0
[[ -n "$START_CRATE" ]] && resuming=1

for crate in "${CRATES[@]}"; do
    version="$(crate_version "$crate")"
    if (( resuming )); then
        if [[ "$crate" != "$START_CRATE" ]]; then
            yellow "  skip $crate (resuming at $START_CRATE)"
            continue
        fi
        resuming=0
    fi
    if on_crates_io "$crate" "$version"; then
        green "  skip $crate@$version (already on crates.io)"
        SKIPPED+=("$crate@$version")
        continue
    fi
    # Floor only between real uploads, not after skips.
    (( uploaded > 0 )) && sleep_with_countdown "$MIN_INTERVAL" "rate-limit floor"

    rc=0
    publish_one "$crate" || rc=$?
    if (( rc == 2 )); then
        green "  $crate@$version already uploaded (cargo confirmed) — skip."
        SKIPPED+=("$crate@$version")
        continue
    fi
    PUBLISHED+=("$crate@$version")
    uploaded=$(( uploaded + 1 ))
    wait_for_index "$crate" "$version"
done

echo
if (( ${#PUBLISHED[@]} > 0 )); then
    green "Published this run (${#PUBLISHED[@]}):"
    for c in "${PUBLISHED[@]}"; do echo "    ✓ $c"; done
fi
if (( ${#SKIPPED[@]} > 0 )); then
    yellow "Already on crates.io (${#SKIPPED[@]}):"
    for c in "${SKIPPED[@]}"; do echo "    ○ $c"; done
fi
green "Publish run finished."
