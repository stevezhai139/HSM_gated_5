# syntax=docker/dockerfile:1
# =============================================================================
# HSM_gated — PostgreSQL 16 + HypoPG + Supabase index_advisor
# -----------------------------------------------------------------------------
# Extends the official postgres:16 image with the two extensions required by
# the Tier-2 end-to-end advisor experiments (Dexter and Supabase index_advisor
# both depend on HypoPG):
#
#   * HypoPG              — hypothetical-index what-if engine
#   * Supabase index_advisor — SQL-driven advisor built on top of HypoPG
#
# All other validation experiments still work unchanged — this image is a
# superset of postgres:16.
#
# Memory cap, conf tuning, ports, volumes are defined in docker-compose.yml.
# =============================================================================
FROM postgres:16

# -----------------------------------------------------------------------------
# Build stage (combined into a single RUN so the intermediate layer is small
# after cleanup).  Uses the default branch of each upstream repo (REL1_STABLE
# for HypoPG, main for index_advisor) — this mirrors the host installer in
# scripts/tier2/install_advisors.sh, so both paths produce the same versions.
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates git build-essential \
        postgresql-server-dev-16; \
    \
    # HypoPG (default branch REL1_STABLE; currently ships as extension v1.4.2)
    git clone --depth 1 \
        https://github.com/HypoPG/hypopg.git /tmp/hypopg; \
    cd /tmp/hypopg && make && make install; \
    \
    # Supabase index_advisor — SQL-only extension; `make install` is enough
    git clone --depth 1 \
        https://github.com/supabase/index_advisor.git /tmp/index_advisor; \
    cd /tmp/index_advisor && make install; \
    \
    # Clean up build tooling to keep the runtime image reasonably small
    apt-get purge -y --auto-remove \
        git build-essential postgresql-server-dev-16; \
    rm -rf /tmp/hypopg /tmp/index_advisor /var/lib/apt/lists/*

# Sanity-check the extension control files are in place (fails the build
# early if either extension is missing).
RUN set -eux; \
    test -f /usr/share/postgresql/16/extension/hypopg.control; \
    test -f /usr/share/postgresql/16/extension/index_advisor.control

# Preserve the entrypoint from the base image.
