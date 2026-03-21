#!/usr/bin/env bash
#
# Usage: ./scripts/release.sh <new_version> [changelog] [changelog_en]
#   e.g. ./scripts/release.sh 1.2.6
#   e.g. ./scripts/release.sh 1.3.0 "- 新功能 A\n- 修复 B" "- New feature A\n- Bug fix B"
#
# Creates a release branch + PR. After the PR is merged, the
# release-tag.yml workflow auto-creates the git tag, which triggers
# the full release pipeline (release.yml).
#
# Steps:
#   1. Validate version format and prerequisites
#   2. Create release/v<version> branch from origin/master
#   3. Update version in CMakeLists.txt, web/frontend/package.json, web/electron/package.json
#   4. Generate web/frontend/public/version-manifest.json
#   5. Commit, push branch, and open PR via gh

set -euo pipefail

# ---------- Helpers ----------

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# ---------- Args ----------

VERSION="${1:-}"
CHANGELOG="${2:-}"
CHANGELOG_EN="${3:-}"
[[ -z "$VERSION" ]] && die "Usage: $0 <version> [changelog] [changelog_en]  (e.g. 1.2.0)"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "Version must be MAJOR.MINOR.PATCH (got: $VERSION)"

TAG="v${VERSION}"
RELEASE_BRANCH="release/${TAG}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------- Pre-checks ----------

command -v gh >/dev/null 2>&1 || die "gh CLI is required. Install: https://cli.github.com/"

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Working tree is dirty. Commit or stash changes first."
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    die "Tag $TAG already exists."
fi

if git show-ref --verify --quiet "refs/heads/$RELEASE_BRANCH" 2>/dev/null; then
    die "Branch $RELEASE_BRANCH already exists locally."
fi

# ---------- Version continuity check ----------

PREV_TAG="$(git tag --sort=-v:refname -l 'v[0-9]*' | head -1)"
if [[ -n "$PREV_TAG" ]]; then
    PREV="${PREV_TAG#v}"
    IFS='.' read -r PM Pm Pp <<< "$PREV"
    IFS='.' read -r NM Nm Np <<< "$VERSION"

    is_successor=false
    if   (( NM == PM + 1 && Nm == 0 && Np == 0 )); then is_successor=true
    elif (( NM == PM && Nm == Pm + 1 && Np == 0 )); then is_successor=true
    elif (( NM == PM && Nm == Pm && Np == Pp + 1 )); then is_successor=true
    fi

    if [[ "$is_successor" == false ]]; then
        echo "WARNING: $PREV -> $VERSION is not a sequential bump."
        read -rp "Continue anyway? [y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || die "Aborted."
    fi

    info "Version: $PREV -> $VERSION"
else
    info "No previous tag found, this will be the first release."
fi

# ---------- 1. Create release branch from latest master ----------

info "Fetching latest origin/master..."
git fetch origin master

info "Creating branch $RELEASE_BRANCH from origin/master..."
git checkout -b "$RELEASE_BRANCH" origin/master

# ---------- 2. Update versions ----------

info "Updating version to $VERSION..."

sed -i "s/project(ChromaPrint3D VERSION [0-9]\+\.[0-9]\+\.[0-9]\+/project(ChromaPrint3D VERSION ${VERSION}/" \
    "$ROOT/CMakeLists.txt"

sed -i "s/\"version\": \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/\"version\": \"${VERSION}\"/" \
    "$ROOT/web/frontend/package.json"

sed -i "s/\"version\": \"[0-9]\+\.[0-9]\+\.[0-9]\+\"/\"version\": \"${VERSION}\"/" \
    "$ROOT/web/electron/package.json"

grep -q "VERSION ${VERSION}" "$ROOT/CMakeLists.txt" || die "Failed to update CMakeLists.txt"
grep -q "\"version\": \"${VERSION}\"" "$ROOT/web/frontend/package.json" || die "Failed to update package.json"
grep -q "\"version\": \"${VERSION}\"" "$ROOT/web/electron/package.json" || die "Failed to update electron package.json"

# ---------- 2b. Generate version manifest ----------

info "Generating version-manifest.json..."
mkdir -p "$ROOT/web/frontend/public"
if [[ -n "$CHANGELOG_EN" ]]; then
cat > "$ROOT/web/frontend/public/version-manifest.json" <<MANIFEST
{
  "version": "${VERSION}",
  "download_url": "https://github.com/neroued/ChromaPrint3D/releases/tag/v${VERSION}",
  "changelog": "${CHANGELOG}",
  "changelog_en": "${CHANGELOG_EN}",
  "published_at": "$(date -u +%Y-%m-%d)"
}
MANIFEST
else
cat > "$ROOT/web/frontend/public/version-manifest.json" <<MANIFEST
{
  "version": "${VERSION}",
  "download_url": "https://github.com/neroued/ChromaPrint3D/releases/tag/v${VERSION}",
  "changelog": "${CHANGELOG}",
  "published_at": "$(date -u +%Y-%m-%d)"
}
MANIFEST
fi

# ---------- 3. Commit, push, create PR ----------

info "Committing version bump..."
git add CMakeLists.txt web/frontend/package.json web/electron/package.json web/frontend/public/version-manifest.json
git commit -m "release: v${VERSION}"

info "Pushing branch $RELEASE_BRANCH..."
git push -u origin "$RELEASE_BRANCH"

info "Creating pull request..."
PR_URL=$(gh pr create \
    --base master \
    --title "release: v${VERSION}" \
    --body "## Release v${VERSION}

Bumps version to \`${VERSION}\` in:
- \`CMakeLists.txt\`
- \`web/frontend/package.json\`
- \`web/electron/package.json\`
- \`web/frontend/public/version-manifest.json\`

After this PR is merged, the \`release-tag\` workflow will automatically
create tag \`${TAG}\` and trigger the full release pipeline.")

info "PR created: $PR_URL"
info ""
info "Next steps:"
info "  1. Wait for CI to pass on the PR"
info "  2. Review and merge the PR"
info "  3. Tag ${TAG} will be created automatically, triggering the release pipeline"
