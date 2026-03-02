export default {
  // Skip releases for PRs without labels
  skipCommitsWithoutPullRequest: true,
  // Don't comment on released PRs
  commentOnReleasedPullRequests: false,
  // Update CITATION.cff during release preparation
  beforePrepare: async ({ exec, nextVersion }) => {
    const today = new Date().toISOString().split('T')[0];
    await exec(`sed -i "s/^version:.*/version: ${nextVersion}/" CITATION.cff`);
    await exec(`sed -i "s/^date-released:.*/date-released: ${today}/" CITATION.cff`);
  },
};
