# Social Graph Builder

<!-- Badges -->
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) -->
<!-- [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/) -->
<!-- [![Tests](https://img.shields.io/github/actions/workflow/status/unblocked-aie/engineering-social-graph/tests.yml?label=tests)](https://github.com/unblocked-aie/engineering-social-graph/actions) -->

Build a developer social graph from GitHub PR history. Identifies domain experts, detects team clusters, and generates interactive visualizations of how your engineering organization collaborates -- all from data already in your GitHub repos.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>

## Features

- **Social Graph** -- Builds a weighted directed graph of who reviews whose PRs, with temporal decay so recent activity matters most
- **Team Detection** -- Automatically clusters contributors into teams using community detection on mutual review patterns
- **Expertise Mapping** -- Identifies domain experts for every code area using git history (commits, lines changed, file breadth, recency)
- **Interactive Visualization** -- Generates a single HTML file with force-directed graph, heat map, peer tables, team cards, and area expertise views
- **Bus Factor Analysis** -- Flags single-owner code areas and knowledge silos

## Quick Start

### 1. Build the CLI

```bash
git clone https://github.com/unblocked-aie/engineering-social-graph.git
cd engineering-social-graph
pip install poetry
poetry install --with dev
poetry run pyinstaller social-graph.spec

# Move the binary onto your PATH
mv dist/social-graph /usr/local/bin/    # macOS/Linux
```

### 2. Run it

```bash
cd /path/to/your/repo
social-graph run

# Or split the pipeline:
social-graph build            # fetch PRs, compute graph, detect teams
social-graph viz              # render visualization (instant, no API calls)

# To include all contributors in smaller repositories without high PR volume:
social-graph run --min-pr-rate 0.0
```

That's it. `social-graph run` fetches PR data from GitHub, builds the collaboration graph, detects teams, scores expertise, and opens an interactive visualization in your browser.

### Running from source (development)

If you're working on the tool itself, you can skip the binary build and run directly via Poetry:

```bash
cd engineering-social-graph
poetry install
poetry run social-graph run --repo-path /path/to/your/repo
```

### Requirements

- A local clone of a GitHub repository (org and repo are auto-detected from the git remote)
- A GitHub token (`gh auth token` or `GITHUB_TOKEN` environment variable)
- Python 3.11+ (only needed if building from source)
- Optional: `ANTHROPIC_API_KEY` for LLM-generated team labels (falls back to code area names)

## How It Works

The social graph algorithm models developer collaboration as a weighted directed graph, where edge weights represent how strongly one developer is connected to another through PR review activity. The algorithm draws on established techniques from network science and signal processing.

### Step 1: PR Data Collection

Uses GitHub's GraphQL API to fetch PRs with nested reviews, review comments, and conversation comments in a single query per page. One query fetches 50 PRs with all their interactions, compared to 150+ REST calls for the same data.

### Step 2: Interaction Scoring with Logarithmic Dampening

For each PR, reviewers are scored using a combination of interaction type weighting and logarithmic dampening:

**Interaction type weights:**

| Type | Weight | Rationale |
|------|--------|-----------|
| PR Review (approve/request changes) | 1.0 | Strongest signal of engagement |
| Review Comment (inline on diff) | 0.7 | Targeted technical feedback |
| Issue Comment (general discussion) | 0.4 | Lighter-weight participation |

**Logarithmic dampening** prevents prolific commenters from dominating the graph. Raw interaction counts are compressed using:

```
score(n) = (1 + log10(n)) * avg_weight(interactions)
```

This means 10 comments count as 2x (not 10x) a single comment. The approach follows the general principle of sublinear scaling in information retrieval, where the marginal value of additional signals diminishes -- analogous to TF-IDF term frequency dampening in text search.

Scores are normalized within each PR so the top reviewer scores 1.0.

### Step 3: Gaussian Time Decay

Recent reviews matter more than old ones. Each PR's contribution is weighted by a Gaussian decay function:

```
decay(t) = exp(-t^2 / C^2)
```

Where `t` is the age in days and `C = 112 days` (16 weeks) is the decay constant. This produces a smooth bell-curve falloff:

| Age | Decay Weight |
|-----|-------------|
| 0 days (today) | 1.000 |
| 56 days (2 months) | 0.779 |
| 112 days (4 months) | 0.368 |
| 224 days (8 months) | 0.018 |
| 336 days (12 months) | 0.000 |

The lookback window is 336 days (3 standard deviations), beyond which the decay is effectively zero. Gaussian decay is preferred over exponential decay because it provides a flatter plateau for recent activity before dropping off -- developers remain strongly connected to collaborators from the past few months, not just last week. This approach is grounded in the general concept of temporal weighting in time-series analysis, where Gaussian kernels are widely used for smoothing and relevance scoring [see e.g., time-decay models in collaborative filtering and recommender systems].

### Step 4: Accumulation and Normalization

For each author:
1. Collect all closed/merged PRs within the 336-day lookback window
2. For each PR, multiply the per-reviewer normalized score by the Gaussian decay factor
3. Sum across all PRs per reviewer
4. Normalize so the author's strongest connection = 1.0
5. Discard edges below 0.1 weight threshold

### Step 5: Team Detection via Community Clustering

Teams are detected using the **Louvain community detection algorithm** [Blondel et al., 2008], applied to an undirected graph derived from the directed social edges.

> V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast unfolding of communities in large networks," *Journal of Statistical Mechanics: Theory and Experiment*, 2008. [arXiv:0803.0476](https://arxiv.org/abs/0803.0476)

The key insight is a custom edge weighting scheme that distinguishes mutual reviewers (likely teammates) from one-directional reviewers (likely cross-team):

**Mutual review bonus** -- When A reviews B's PRs AND B reviews A's PRs, the undirected edge weight uses a geometric mean with a 3x multiplier:

```
w_mutual(A, B) = sqrt(w_AB * w_BA) * 3.0
```

The geometric mean is critical: two developers who equally review each other (0.5 / 0.5 = geometric mean 0.5) score higher than an imbalanced pair (1.0 / 0.1 = geometric mean 0.316), even though their raw sums are similar. This naturally separates same-team collaborators from cross-team reviewers.

**One-way penalty** -- Unidirectional relationships (e.g., a tech lead reviewing a junior's PRs but not vice versa) receive half weight:

```
w_oneway(A, B) = (w_AB + w_BA) * 0.5
```

Louvain is run with **resolution 2.0** to produce finer-grained clusters that better match real organizational teams rather than merging related teams into super-clusters.

### Step 6: Code Expertise Scoring

For each directory in the repository, contributors are scored using a weighted composite of four factors derived from `git log --numstat`:

```
expertise(author, area) = 0.4 * lines_changed_decayed
                        + 0.3 * commit_count_decayed
                        + 0.2 * file_breadth
                        + 0.1 * recency
```

| Factor | Weight | Signal |
|--------|--------|--------|
| Lines changed (time-decayed) | 40% | Volume of work, recency-weighted |
| Commit count (time-decayed) | 30% | Frequency of engagement |
| File breadth | 20% | Proportion of files in the area they have touched |
| Recency | 10% | Inverse of days since last commit |

All factors are normalized per-area so scores are comparable. **Bus factor** is computed as the number of contributors scoring at least 20% of the top contributor's score.

### Step 7: Visualization Layout

The interactive graph uses **Sigma.js** with the **ForceAtlas2** layout algorithm [Jacomy et al., 2014], a force-directed algorithm designed specifically for network spatialization and community visualization.

> M. Jacomy, T. Venturini, S. Heymann, and M. Bastian, "ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software," *PLOS ONE*, 9(6), 2014. [DOI: 10.1371/journal.pone.0098679](https://doi.org/10.1371/journal.pone.0098679)

Nodes are sized by **lines of code committed** (additions + deletions), using a square root scale normalized to the largest contributor. The square root provides visible proportionality without the most active contributor overwhelming the layout.

### Step 8: Team Labeling (Optional)

If an Anthropic API key is provided, Claude generates concise team names from each team's most distinctive code areas. The prompt includes member names, top 8 code areas ranked by the team's relative expertise, and a distinctiveness score. Falls back to directory-name-based labels otherwise.

## Visualization

The `viz` command generates a single HTML file with five tabs:

### Interactive Graph
- Sigma.js with ForceAtlas2 layout and community-aware clustering
- Nodes are GitHub avatars sized by contribution volume
- Team bounding circles with labels
- Hover highlights connected nodes; click shows review relationship details in sidebar
- Edge thickness and brightness proportional to collaboration strength

### Heat Map Grid
- Reviewer x Author matrix with color-coded cells

### Peer Tables
- Per-reviewer breakdown showing who they review and how strongly
- Progress bars for weight visualization

### Teams
- Auto-detected team cards with member avatars, contribution stats, and collaboration details
- AI-generated team labels (or code-area-based fallback)

### Experts
- Code area cards ranked by total activity
- Per-area expert rankings with composite scores
- Bus factor badges: "Single owner" (red), "At risk" (orange), "Shared" (blue), "Well covered" (green)

## CLI Reference

```
social-graph run              Build + visualize in one command (the default workflow)
social-graph build            Fetch PRs, compute graph, detect teams, store in DB
social-graph viz              Render from pre-built data (instant, no API calls)
social-graph status           Show current config and build summary
social-graph auth             Store API keys in the system keychain
social-graph label-teams      Re-generate AI team labels from cached data
social-graph experts          Find domain experts (subcommands: path, repo, areas, domain, domains)
social-graph export           Export graph as JSON or GraphML
social-graph reset            Delete all cached data and start fresh
```

### Build Options

```
--repo-path PATH       Path to local git clone (default: current directory)
--org TEXT             GitHub organization (auto-detected from git remote)
--repo TEXT            Repository name, can specify multiple (auto-detected)
--since-prs DATE       Fetch PRs updated since this date (e.g. '2025-01-01')
--since-git TEXT        Git log lookback period (default: '6 months ago')
--limit INT            Max PRs to fetch
--skip-fetch           Rebuild graph from cached PRs only (no API calls)
--fresh                Ignore cache timestamps, re-fetch everything
--min-pr-rate FLOAT    Minimum avg PRs per month for an author/reviewer (default: 7.0)
```

### Authentication

```bash
# Store keys in the system keychain (macOS Keychain / Windows Credential Manager)
social-graph auth --github ghp_xxxx
social-graph auth --anthropic sk-ant-xxxx

# Or use environment variables
export GITHUB_TOKEN=ghp_xxxx            # also falls back to `gh auth token`
export ANTHROPIC_API_KEY=sk-ant-xxxx    # optional, for AI team labels
```

## Data Storage

All computed data is stored in `~/.social-graph/cache.db` (SQLite):

| Table | Contents |
|-------|----------|
| `pull_requests` | Cached PR metadata from GitHub |
| `interactions` | PR reviews, review comments, issue comments |
| `social_edges` | Computed author-to-reviewer weighted edges |
| `lines_by_user` | Lines of code per contributor (from git) |
| `code_areas` | Detected code areas with bus factor |
| `area_experts` | Per-area expert rankings |
| `communities` | Team membership assignments |
| `team_labels` | Generated team names |
| `build_meta` | Build parameters and timestamps |

## Architecture

```
src/social_graph/
  cli.py              CLI entry point (Click)
  builder.py          Build pipeline orchestrator
  config.py           Settings, token resolution
  models.py           Data classes (User, PullRequest, SocialEdge, etc.)
  decay.py            Gaussian time decay function
  scoring.py          Interaction weights, log dampening, normalization
  fetcher/
    github.py         GitHub GraphQL client
    cache.py          SQLite storage layer
  graph/
    social.py         Social graph construction algorithm
    expertise.py      File/repo expertise from GitHub API
    code_expertise.py Area expertise detection from git log
  output/
    grid.py           HTML generation (all visualization tabs)
    visualization.py  Sigma.js graph data, community detection
    table.py          Rich terminal tables
    json_export.py    JSON/GraphML export
```

## Development

```bash
poetry install --with dev
poetry run pytest -v
poetry run ruff check src/
```

### Building the Binary

```bash
poetry install --with dev
pyinstaller social-graph.spec
# Binary is at dist/social-graph (macOS/Linux) or dist/social-graph.exe (Windows)
```

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository and create a feature branch from `main`
2. **Install** dev dependencies: `poetry install --with dev`
3. **Make your changes** and add tests for new functionality
4. **Run checks** before submitting:
   ```bash
   poetry run pytest -v
   poetry run ruff check src/
   ```
5. **Open a pull request** with a clear description of the change and its motivation

### Guidelines

- Keep PRs focused -- one feature or fix per PR
- Follow the existing code style (enforced by Ruff, line length 120)
- Add tests for new scoring logic or graph algorithms
- Update the README if you change CLI commands or algorithm behavior

## References

The following papers directly underpin algorithms used in this project:

- V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, "Fast unfolding of communities in large networks," *J. Stat. Mech.*, 2008. [arXiv:0803.0476](https://arxiv.org/abs/0803.0476) — The Louvain algorithm, used directly for team detection via `networkx.community.louvain_communities()`.
- G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing & Management*, 24(5), pp. 513-523, 1988. [DOI:10.1016/0306-4573(88)90021-0](https://doi.org/10.1016/0306-4573(88)90021-0) — Sublinear TF scaling (`1 + log(n)`), the basis for our interaction count dampening.
- M. Jacomy, T. Venturini, S. Heymann, and M. Bastian, "ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software," *PLOS ONE*, 9(6), e98679, 2014. [DOI:10.1371/journal.pone.0098679](https://doi.org/10.1371/journal.pone.0098679) — The force-directed layout algorithm used in the interactive visualization via Sigma.js.

## License

MIT -- see [LICENSE](LICENSE) for details.
