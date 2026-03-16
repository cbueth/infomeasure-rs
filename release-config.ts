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
  commentOnReleasedPullRequests: false,
  // Update CITATION.cff during release preparation
  beforePrepare: async ({ exec, nextVersion }) => {
    const today = new Date().toISOString().split('T')[0];
    await exec(`sed -i "s/^version:.*/version: ${nextVersion}/" CITATION.cff`);
    await exec(`sed -i "s/^date-released:.*/date-released: ${today}/" CITATION.cff`);
  },
};
