# Yoke Agent Skills

Guided, interview-style procedures that an AI coding agent follows to add
training capability to Yoke. Each skill is a folder with a `SKILL.md` (YAML
frontmatter `name` + `description`, then the procedure) and a `reference/`
folder of Yoke-specific notes it loads as needed.

| Skill | Produces |
| --- | --- |
| `yoke-new-dataset` | A `Dataset` class in `src/yoke/datasets/` + tests |
| `yoke-new-training-step` | Train/eval datasteps + epoch function in `src/yoke/utils/training/` + tests |
| `yoke-slurm-template` | `training_slurm.tmpl` (+ `evaluation_slurm.tmpl`) for the current cluster; saves a machine profile |
| `yoke-new-harness` | An `applications/harnesses/<name>/` directory on `HarnessTrainer`; calls the other three when needed |

Typical flow:

```
yoke-new-dataset ──► (no matching contract?) yoke-new-training-step ──┐
        │                                                             ▼
        └──────────────────────────────────────────────► yoke-new-harness
                                                                      │ SLURM?
                                                                      ▼
                                                           yoke-slurm-template
```

Skills hand results to each other through short **cards**
(`shared/cards.md`) written into a `dev_plans/` notes file.

## Layout

```
skills/
  shared/                 # used by all skills
    interview_conventions.md
    batch_contracts.md    # dataset tuple -> datastep -> epoch function
    cards.md
  yoke-new-dataset/
  yoke-new-training-step/
  yoke-slurm-template/
    machines/             # saved per-cluster profiles (created by the skill)
  yoke-new-harness/
```

This folder is the single, agent-neutral copy. Agents find the skills
through symlinks:

- Claude Code: `.claude/skills/<skill>` → `../../skills/<skill>`.
- Other agents (Codex, opencode): not linked yet. Agents that read
  `AGENTS.md` are pointed here from its "Agent skills" section.

Skills refer to each other's files by **repo-root paths**
(`skills/shared/...`) so they work through symlinks.

## Ground rules baked into every skill

- Ask, don't guess; record decisions in `dev_plans/`; get approval before
  writing code.
- Read-only inspection of data and machines.
- Never submit jobs, train, add dependencies, or commit.

## Keeping the skills accurate

`tests/skills/test_skill_files.py` runs in CI and checks:

- frontmatter;
- symlinks;
- that every repo path a skill mentions exists;
- that every `yoke.` dotted name imports;
- that each row of `shared/batch_contracts.md` matches the tuple its
  datastep unpacks;
- that saved machine profiles are dated and keep the allocation warning.

When you rename or change code that the skills mention, update them in the
same change.
