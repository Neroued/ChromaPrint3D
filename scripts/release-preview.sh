#!/usr/bin/env bash
#
# Usage: ./scripts/release-preview.sh <version> [rc_number]
#   e.g. ./scripts/release-preview.sh 1.2.12        # auto-detects rc number
#   e.g. ./scripts/release-preview.sh 1.2.12 3      # explicit rc.3
#   e.g. ./scripts/release-preview.sh 1.3.0 1 "- 新功能 A 预览" "- Preview of feature A"
#
# Creates a preview branch + PR. After the PR is merged, the
# release-tag.yml workflow auto-creates the git tag, which triggers
# the full release pipeline (release.yml) with pre-release flags.
#
# Steps:
#   1. Validate version format and prerequisites
#   2. Determine rc number (auto-increment or explicit)
#   3. Create preview/v<version>-rc.<N> branch from origin/master
#   4. Update version in CMakeLists.txt, web/frontend/package.json, web/electron/package.json
#   5. Generate web/frontend/public/version-manifest.json
#   6. Commit, push branch, and open PR via gh

set -euo pipefail

# ---------- Helpers ----------

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# ---------- Args ----------

VERSION="${1:-}"
RC_NUMBER="${2:-}"
CHANGELOG="${3:-}"
CHANGELOG_EN="${4:-}"
[[ -z "$VERSION" ]] && die "Usage: $0 <version> [rc_number] [changelog] [changelog_en]  (e.g. 1.2.12)"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Version must be MAJOR.MINOR.PATCH (got: $VERSION)"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------- Pre-checks ----------

command -v gh >/dev/null 2>&1 || die "gh CLI is required. Install: https://cli.github.com/"

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree is dirty. Commit or stash changes first."
fi

# ---------- Auto-detect rc number ----------

if [[ -z "$RC_NUMBER" ]]; then
    git fetch origin --tags
    LAST_RC="$(git tag --sort=-v:refname -l "v${VERSION}-rc.*" | head -1)"
    if [[ -n "$LAST_RC" ]]; then
        PREV_N="${LAST_RC##*-rc.}"
        RC_NUMBER=$(( PREV_N + 1 ))
        info "Auto-incremented from $LAST_RC -> rc.$RC_NUMBER"
    else
        RC_NUMBER=1
        info "No previous rc tag found for v${VERSION}, starting at rc.1"
    fi
fi

[[ "$RC_NUMBER" =~ ^[1-9][0-9]*$ ]] || die "rc_number must be a positive integer (got: $RC_NUMBER)"

PRERELEASE="rc.${RC_NUMBER}"
FULL_VERSION="${VERSION}-${PRERELEASE}"
TAG="v${FULL_VERSION}"
PREVIEW_BRANCH="preview/${TAG}"

# ---------- Check for conflicts ----------

if git rev-parse "$TAG" >/dev/null 2>&1; then
    die "Tag $TAG already exists."
fi

if git show-ref --verify --quiet "refs/heads/$PREVIEW_BRANCH" 2>/dev/null; then
    die "Branch $PREVIEW_BRANCH already exists locally."
fi

info "Preview version: $FULL_VERSION"

# ---------- 1. Create preview branch from latest master ----------

info "Fetching latest origin/master..."
git fetch origin master

info "Creating branch $PREVIEW_BRANCH from origin/master..."
git checkout -b "$PREVIEW_BRANCH" origin/master

# ---------- 2. Update versions ----------

info "Updating version to $FULL_VERSION..."

sed -i "s/project(ChromaPrint3D VERSION [0-9]\+\.[0-9]\+\.[0-9]\+/project(ChromaPrint3D VERSION ${VERSION}/" \
    "$ROOT/CMakeLists.txt"

if grep -q 'set(CHROMAPRINT3D_VERSION_PRERELEASE' "$ROOT/CMakeLists.txt"; then
    sed -i "s/set(CHROMAPRINT3D_VERSION_PRERELEASE \"[^\"]*\"/set(CHROMAPRINT3D_VERSION_PRERELEASE \"${PRERELEASE}\"/" \
        "$ROOT/CMakeLists.txt"
else
    die "CHROMAPRINT3D_VERSION_PRERELEASE not found in CMakeLists.txt"
fi

sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${FULL_VERSION}\"/" \
    "$ROOT/web/frontend/package.json"

sed -i "s/\"version\": \"[^\"]*\"/\"version\": \"${FULL_VERSION}\"/" \
    "$ROOT/web/electron/package.json"

grep -q "VERSION ${VERSION}" "$ROOT/CMakeLists.txt" || die "Failed to update CMakeLists.txt"
grep -q "\"${PRERELEASE}\"" "$ROOT/CMakeLists.txt" || die "Failed to update CHROMAPRINT3D_VERSION_PRERELEASE"
grep -q "\"version\": \"${FULL_VERSION}\"" "$ROOT/web/frontend/package.json" || die "Failed to update frontend package.json"
grep -q "\"version\": \"${FULL_VERSION}\"" "$ROOT/web/electron/package.json" || die "Failed to update electron package.json"

# ---------- 2b. Generate version manifest ----------

info "Generating version-manifest.json..."
mkdir -p "$ROOT/web/frontend/public"
if [[ -n "$CHANGELOG_EN" ]]; then
cat > "$ROOT/web/frontend/public/version-manifest.json" <<MANIFEST
{
  "version": "${FULL_VERSION}",
  "download_url": "https://github.com/neroued/ChromaPrint3D/releases/tag/${TAG}",
  "changelog": "${CHANGELOG}",
  "changelog_en": "${CHANGELOG_EN}",
  "published_at": "$(date -u +%Y-%m-%d)"
}
MANIFEST
else
cat > "$ROOT/web/frontend/public/version-manifest.json" <<MANIFEST
{
  "version": "${FULL_VERSION}",
  "download_url": "https://github.com/neroued/ChromaPrint3D/releases/tag/${TAG}",
  "changelog": "${CHANGELOG}",
  "published_at": "$(date -u +%Y-%m-%d)"
}
MANIFEST
fi

# ---------- 3. Commit, push, create PR ----------

info "Committing preview version bump..."
git add CMakeLists.txt web/frontend/package.json web/electron/package.json web/frontend/public/version-manifest.json
git commit -m "preview: ${TAG}"

info "Pushing branch $PREVIEW_BRANCH..."
git push -u origin "$PREVIEW_BRANCH"

info "Creating pull request..."
PR_URL=$(gh pr create \
    --base master \
    --title "preview: ${TAG}" \
    --body "## Preview ${TAG}

Bumps version to \`${FULL_VERSION}\` (pre-release) in:
- \`CMakeLists.txt\` (VERSION + PRERELEASE)
- \`web/frontend/package.json\`
- \`web/electron/package.json\`
- \`web/frontend/public/version-manifest.json\`

After this PR is merged, the \`release-tag\` workflow will automatically
create tag \`${TAG}\` and trigger the release pipeline with **pre-release** flags.

### Docker tags produced
- \`neroued/chromaprint3d:preview\` / \`neroued/chromaprint3d:${FULL_VERSION}\`
- \`neroued/chromaprint3d:preview-api\` / \`neroued/chromaprint3d:${FULL_VERSION}-api\`

### Promote to stable
When testing is complete, run:
\`\`\`
./scripts/release.sh ${VERSION}
\`\`\`")

info "PR created: $PR_URL"
info ""
info "Next steps:"
info "  1. Wait for CI to pass on the PR"
info "  2. Review and merge the PR"
info "  3. Tag ${TAG} will be created automatically, triggering the pre-release pipeline"
info "  4. Deploy to preview environment for testing"
info "  5. When ready, promote to stable: ./scripts/release.sh ${VERSION}"
