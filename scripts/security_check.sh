#!/usr/bin/env bash
# Read-only public-release checks. Match values are never printed.
set -u

REPOSITORY_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPOSITORY_ROOT" || exit 2

warnings=0
blockers=0
git_available=0

say() {
  printf '%s\n' "$1"
}

scan_backend="none"
if command -v rg >/dev/null 2>&1; then
  scan_backend="rg"
elif command -v grep >/dev/null 2>&1; then
  scan_backend="grep"
  say "INFO  ripgrep is unavailable; using grep -E for content scans."
else
  say "FAIL  neither ripgrep nor grep is available; content scans cannot run."
  blockers=$((blockers + 1))
fi

public_text_files() {
  find . \
    \( \
      -path './.git' -o \
      -path './.venv' -o \
      -path './.ruff_cache' -o \
      -path './data/cache*' -o \
      -path './reports/output*' -o \
      -path './artifacts' -o \
      -path './progress_log.md' -o \
      -path './FINAL_REPORT.md' -o \
      -path './RESEARCH_SUMMARY.md' -o \
      -path './docs/codex_task_prompt*.md' -o \
      -path './docs/handover_v6.md' -o \
      -path './docs/trading_bot_dev_prompt.md' -o \
      -path './scripts/security_check.sh' -o \
      -name '__pycache__' \
    \) -prune -o \
    -type f \
    ! -name '*.parquet' \
    ! -name '*.feather' \
    ! -name '*.pkl' \
    ! -name '*.joblib' \
    -print0
}

redacted_content_scan() {
  label="$1"
  pattern="$2"

  if [ "$scan_backend" = "none" ]; then
    say "FAIL  $label: no supported content scanner is available."
    return
  fi

  if [ "$scan_backend" = "rg" ]; then
    matches="$(
      rg -n -i --hidden \
        --glob '!.git/**' \
        --glob '!.venv/**' \
        --glob '!.ruff_cache/**' \
        --glob '!**/__pycache__/**' \
        --glob '!data/cache*/**' \
        --glob '!reports/output*/**' \
        --glob '!artifacts/**' \
        --glob '!progress_log.md' \
        --glob '!FINAL_REPORT.md' \
        --glob '!RESEARCH_SUMMARY.md' \
        --glob '!docs/codex_task_prompt*.md' \
        --glob '!docs/handover_v6.md' \
        --glob '!docs/trading_bot_dev_prompt.md' \
        --glob '!scripts/security_check.sh' \
        --glob '!*.parquet' \
        --glob '!*.feather' \
        --glob '!*.pkl' \
        --glob '!*.joblib' \
        "$pattern" . 2>/dev/null |
        awk -F: '!seen[$1 ":" $2]++ {print $1 ":" $2 " [MATCH REDACTED]"; count += 1; if (count == 20) exit}'
    )"
  else
    matches="$(
      public_text_files |
        xargs -0 grep -n -E -i -- "$pattern" 2>/dev/null |
        awk -F: '!seen[$1 ":" $2]++ {print $1 ":" $2 " [MATCH REDACTED]"; count += 1; if (count == 20) exit}'
    )"
  fi
  if [ -n "$matches" ]; then
    say "WARN  $label: candidate matches found (values redacted):"
    printf '%s\n' "$matches"
    warnings=$((warnings + 1))
  else
    say "PASS  $label: no candidate matches in the public working-tree scope."
  fi
}

say "Security check scope: $REPOSITORY_ROOT"
say "Excluded from content scanning: .git, .venv, ignored data caches, local report outputs, and generated artifacts."

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_available=1
  say "INFO  Git repository detected; tracked-file checks enabled."
else
  say "WARN  No Git repository detected; history and tracked-file checks are unavailable."
  warnings=$((warnings + 1))
fi

redacted_content_scan \
  "high-signal secret patterns" \
  '(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[bp]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|[?&](access[_-]?token|api[_-]?key|token)=[^&[:space:]]+)'

redacted_content_scan \
  "credential assignments" \
  '(api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|authorization|bearer)[[:space:]]*[:=][[:space:]]*[^[:space:]#]{8,}'

redacted_content_scan \
  "local absolute paths" \
  '(/Users/[^/[:space:]]+|/home/[^/[:space:]]+|[A-Za-z]:\\Users\\[^\\[:space:]]+)'

if find . -path './.git' -prune -o -path './.venv' -prune -o -type f \
  \( -name '*.pkl' -o -name '*.pickle' -o -name '*.joblib' -o -name '*.pt' -o -name '*.pth' \
  -o -name '*.ckpt' -o -name '*.h5' -o -name '*.hdf5' -o -name '*.onnx' \) -print |
  grep -q .; then
  say "WARN  Serialized model/object artifacts exist outside .venv; inspect before release."
  warnings=$((warnings + 1))
else
  say "PASS  No risky serialized model/object artifacts found outside .venv."
fi

if find . -path './.git' -prune -o -path './.venv' -prune -o -type f -name '*.ipynb' -print |
  grep -q .; then
  say "WARN  Notebook files exist; clear and inspect outputs before release."
  warnings=$((warnings + 1))
else
  say "PASS  No notebook files found."
fi

if find . -path './.git' -prune -o -path './.venv' -prune -o -type f \
  \( -name '.env' -o -name '.env.local' -o -name '.env.production' \) -print |
  grep -q .; then
  say "WARN  Local .env file(s) exist. They are ignored but require manual review."
  warnings=$((warnings + 1))
else
  say "PASS  No local .env file found."
fi

if [ "$git_available" -eq 1 ]; then
  remote_urls="$(git config --get-regexp '^remote\..*\.url$' 2>/dev/null || true)"
  if printf '%s\n' "$remote_urls" |
    grep -Eiq '(https?://[^/@[:space:]]+@|ghp_|github_pat_|[?&](access_token|token|api_key)=)'; then
    say "FAIL  A remote URL may contain embedded credentials (URL suppressed)."
    blockers=$((blockers + 1))
  else
    say "PASS  No embedded credential pattern found in configured remote URLs."
  fi

  tracked_env="$(git ls-files '.env' '.env.*' 2>/dev/null | grep -v '^.env.example$' || true)"
  if [ -n "$tracked_env" ]; then
    say "FAIL  A non-example .env file is tracked (path list suppressed)."
    blockers=$((blockers + 1))
  else
    say "PASS  No non-example .env file is tracked."
  fi

  tracked_raw="$(
    git ls-files \
      'data/raw/**' 'data/private/**' 'data/vendor/**' 'data/cache/**' 'data/cache_*/**' \
      '*.parquet' '*.feather' '*.h5' '*.hdf5' '*.pkl' '*.pickle' '*.joblib' 2>/dev/null
  )"
  if [ -n "$tracked_raw" ]; then
    say "FAIL  Raw/cache/serialized data candidates are tracked (path list suppressed)."
    blockers=$((blockers + 1))
  else
    say "PASS  No raw/cache/serialized data candidate is tracked."
  fi

  large_count=0
  while IFS= read -r tracked_file; do
    [ -f "$tracked_file" ] || continue
    file_size="$(stat -f '%z' "$tracked_file" 2>/dev/null || stat -c '%s' "$tracked_file" 2>/dev/null || printf '0')"
    if [ "$file_size" -gt 10485760 ] 2>/dev/null; then
      large_count=$((large_count + 1))
    fi
  done <<EOF
$(git ls-files)
EOF
  if [ "$large_count" -gt 0 ]; then
    say "WARN  $large_count tracked file(s) exceed 10 MiB; names suppressed pending manual review."
    warnings=$((warnings + 1))
  else
    say "PASS  No tracked file exceeds 10 MiB."
  fi

  commit_count="$(git rev-list --all --count 2>/dev/null || printf '0')"
  if [ "$commit_count" -gt 0 ] 2>/dev/null; then
    say "INFO  Git history exists ($commit_count commit(s)); run a dedicated history secret scanner before public release."
    warnings=$((warnings + 1))
  else
    say "INFO  Git repository has no commit history."
  fi
else
  say "SKIP  Tracked .env, raw-data, large-file, and history checks."
fi

say "Summary: blockers=$blockers warnings=$warnings"
if [ "$blockers" -gt 0 ]; then
  exit 1
fi
exit 0
