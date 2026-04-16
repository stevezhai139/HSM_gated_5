#!/usr/bin/env bash
# ==============================================================================
# install_advisors.sh — Phase 1, Tier 2 end-to-end advisor installation
# ------------------------------------------------------------------------------
# Installs the three pieces needed by hsm_{oltp,burst}_end_to_end.py:
#
#   1) HypoPG       — hypothetical-index extension (shared dependency)
#   2) Dexter       — Ankane/pgdexter, Ruby gem, log-driven advisor
#   3) index_advisor — Supabase PostgreSQL extension, SQL-driven advisor
#
# Target host: user's own machine (PostgreSQL 16, Ubuntu/Debian or macOS).
# This script is idempotent — safe to re-run.
#
# Exit codes:
#   0  everything installed and verified
#   1  missing prerequisite (psql, pg_config, make, ruby)
#   2  build/install failure
#
# Usage:
#   # The installer reuses the same .env pattern as the rest of HSM_gated:
#   cp .env.example .env && source .env
#   bash install_advisors.sh                       # build + install only
#
#   # To also enable extensions in a DB, pass HSM_DB (or rely on env):
#   HSM_DB=postgres bash install_advisors.sh       # Docker default DB
#   HSM_DB=$HSM_DOCKER_DB PGPORT=$HSM_DOCKER_PORT PGUSER=$HSM_DOCKER_USER \
#       PGPASSWORD=$HSM_DOCKER_PASSWORD bash install_advisors.sh
# ==============================================================================
set -euo pipefail

log()  { printf '[install] %s\n' "$*"; }
die()  { printf '[install][ERROR] %s\n' "$*" >&2; exit "${2:-2}"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ─── 0. Prerequisites ────────────────────────────────────────────────────────
log "Checking prerequisites…"
have pg_config || die "pg_config not found (install postgresql-server-dev-16)" 1
have psql      || die "psql not found (install postgresql-client-16)" 1
have make      || die "make not found" 1
have git       || die "git not found" 1
have ruby      || die "ruby not found (need >= 3.1 for pgdexter 0.6.x transitive google-protobuf)" 1

# Ruby >= 3.1 is required because pgdexter 0.6.x → pg_query 6.x → google-protobuf 4.x,
# which drops Ruby 2.6 / 2.7 / 3.0. macOS system Ruby is 2.6.10 — DO NOT use it.
RUBY_VER="$(ruby -e 'print RUBY_VERSION')"
RUBY_MAJOR="$(echo "${RUBY_VER}" | cut -d. -f1)"
RUBY_MINOR="$(echo "${RUBY_VER}" | cut -d. -f2)"
if (( RUBY_MAJOR < 3 )) || (( RUBY_MAJOR == 3 && RUBY_MINOR < 1 )); then
  die "Ruby ${RUBY_VER} is too old for pgdexter (need >= 3.1). On macOS: brew install ruby && export PATH=\"/opt/homebrew/opt/ruby/bin:\$PATH\"" 1
fi
log "Ruby ${RUBY_VER} ($(which ruby))"

PG_MAJOR="$(pg_config --version | awk '{print $2}' | cut -d. -f1)"
log "pg_config reports PostgreSQL ${PG_MAJOR}"
[[ "${PG_MAJOR}" -ge 14 ]] || die "PostgreSQL 14+ required; found ${PG_MAJOR}" 1

WORK="${TMPDIR:-/tmp}/hsm_tier2_build"
mkdir -p "${WORK}"
log "Build directory: ${WORK}"

# ─── 1. HypoPG ───────────────────────────────────────────────────────────────
if [[ -d "$(pg_config --sharedir)/extension" ]] && \
   ls "$(pg_config --sharedir)/extension" | grep -q '^hypopg'; then
  log "HypoPG already installed (sharedir)"
else
  log "Building HypoPG from source…"
  cd "${WORK}"
  [[ -d hypopg ]] || git clone --depth 1 https://github.com/HypoPG/hypopg.git
  cd hypopg
  make -s
  sudo make -s install
  log "HypoPG installed."
fi

# ─── 2. Supabase index_advisor ───────────────────────────────────────────────
if ls "$(pg_config --sharedir)/extension" | grep -q '^index_advisor'; then
  log "Supabase index_advisor already installed"
else
  log "Building Supabase index_advisor from source…"
  cd "${WORK}"
  [[ -d index_advisor ]] || git clone --depth 1 https://github.com/supabase/index_advisor.git
  cd index_advisor
  # index_advisor is a SQL-only extension (no C code to build).
  sudo make -s install
  log "Supabase index_advisor installed."
fi

# ─── 3. Dexter (Ruby gem) ────────────────────────────────────────────────────
if have dexter; then
  log "Dexter already installed ($(dexter --version 2>/dev/null || echo unknown))"
else
  log "Installing Dexter Ruby gem…"
  # `sudo` only if non-writable gem dir. Users of rbenv/rvm can drop sudo.
  if gem environment gemdir | xargs -I{} test -w {} 2>/dev/null; then
    gem install pgdexter
  else
    sudo gem install pgdexter
  fi
  log "Dexter installed: $(which dexter)"
fi

# ─── 4. Enable extensions in the target DB (optional) ────────────────────────
if [[ -n "${HSM_DB:-}" ]]; then
  log "Enabling extensions in database ${HSM_DB}…"
  psql -v ON_ERROR_STOP=1 -d "${HSM_DB}" <<'SQL'
CREATE EXTENSION IF NOT EXISTS hypopg;
CREATE EXTENSION IF NOT EXISTS index_advisor;
SQL
  log "Extensions enabled."
else
  log "HSM_DB not set — skipping CREATE EXTENSION step."
  log "  Run manually:  psql -d <yourdb> -c 'CREATE EXTENSION hypopg; CREATE EXTENSION index_advisor;'"
fi

# ─── 5. Verify ───────────────────────────────────────────────────────────────
log "Verifying installations…"
have dexter || die "dexter CLI missing after install" 2
ls "$(pg_config --sharedir)/extension" | grep -qE '^hypopg' || die "hypopg.control missing" 2
ls "$(pg_config --sharedir)/extension" | grep -qE '^index_advisor' || die "index_advisor.control missing" 2

log "All three components installed successfully."
log ""
log "Next steps:"
log "  1. cp .env.example .env  &&  source .env      (if not already done)"
log "  2. python scripts/tier2/smoke_test_advisors.py            # docker default (5433)"
log "     python scripts/tier2/smoke_test_advisors.py --mode native   # if using 5432"
