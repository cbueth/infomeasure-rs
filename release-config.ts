// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

export default {
  changeTypes: [
    {
      title: '💥 Breaking changes',
      labels: ['breaking', 'Compat/Breaking'],
      bump: 'major',
      weight: 3,
    },
    {
      title: '🔒 Security',
      labels: ['security', 'Kind/Security'],
      bump: 'patch',
      weight: 2,
    },
    {
      title: '✨ Features',
      labels: ['feature', 'Kind/Feature'],
      bump: 'minor',
      weight: 1,
    },
    {
      title: '📈 Enhancement',
      labels: ['enhancement', 'refactor', 'Kind/Enhancement'],
      bump: 'minor',
    },
    {
      title: '🐛 Bug Fixes',
      labels: ['bug', 'Kind/Bug'],
      bump: 'patch',
    },
    {
      title: '⚡ Performance',
      labels: ['perf', 'Kind/Performance'],
      bump: 'patch',
    },
    {
      title: '📚 Documentation',
      labels: ['docs', 'documentation', 'Kind/Documentation'],
      bump: 'patch',
    },
    {
      title: '📦️ Dependency',
      labels: ['dependency', 'dependencies', 'Kind/Dependency'],
      bump: 'patch',
      weight: -1,
    },
    {
      title: 'Misc',
      labels: ['misc', 'Kind/Testing'],
      bump: 'patch',
      default: true,
      weight: -2,
    },
  ],
  skipLabels: ['skip-release', 'skip-changelog', 'regression'],
  skipCommitsWithoutPullRequest: true,
  commentOnReleasedPullRequests: true,
  // Update version files during release preparation so the release PR carries the bump:
  // Cargo.toml, CITATION.cff, plus the user-facing version references in README.md
  // ("Now available!" banner and dependency snippets) and src/lib.rs (docs.rs banner).
  beforePrepare: async ({ exec, nextVersion }) => {
    const today = new Date().toISOString().split('T')[0];
    await exec(`sed -i "s/^version:.*/version: ${nextVersion}/" CITATION.cff`);
    await exec(`sed -i "s/^date-released:.*/date-released: ${today}/" CITATION.cff`);
    await exec(`sed -i "1,/^version = .*/s/^version = .*/version = \\"${nextVersion}\\"/" Cargo.toml`);
    // Bump the "Now available!" banner in README and the crate docs (docs.rs).
    await exec(`sed -i "s/\\*\\*v[0-9][^ ]*/\\*\\*v${nextVersion}/g" README.md src/lib.rs`);
    // Bump the plain dependency snippet: infomeasure = "0.3.0"
    await exec(`sed -i "s/infomeasure = \\"[0-9][^\\"]*\\"/infomeasure = \\"${nextVersion}\\"/g" README.md`);
    // Bump the feature-flavored snippets: infomeasure = { version = "0.3.0", ... }
    await exec(`sed -i "s/infomeasure = { version = \\"[0-9][^\\"]*\\"/infomeasure = { version = \\"${nextVersion}\\"/g" README.md`);
  },
};
