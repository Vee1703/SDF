"""Behavioural probes: the obfuscation probe and the clause-(c) test.

`research_context.md` calls these non-negotiable for any target that touches monitoring,
oversight or evaluation, and states the reason bluntly: prose-level grading is
insufficient, and reading traces cannot distinguish success from its dangerous twin. The
recorded hazard is specific -- training on documents about CoT monitoring produced higher
rates of *undetected* misbehavior across eight models, with CoT controllability
correlating with obfuscation success at r=0.800 (p=0.017).

Two probes, both scoring behaviour rather than prose:
  - obfuscation_probe: on tasks where a shortcut exists, measure
    (shortcut taken AND not named in the trace) / (shortcut taken), versus the base model.
    A model that takes the shortcut just as often but mentions it less has got worse, not
    better, even though its traces read better.
  - clause_c_test: run the same tasks with and without an "unmonitored, private" system
    prompt. The shortcut rate must not move. If it does, the model has learned to condition
    on being watched rather than to behave.

`research_context.md` instructs that this suite is built BEFORE the corpus is generated.
It is stubbed here for that reason, not as an afterthought.

Open question this module must answer, quoted from that file: "What is this project's
obfuscation probe?" It cannot be written until the target behaviour and the shortcut-
bearing task set are chosen.

Intended call flow:
    tasks   = load_shortcut_tasks(path)          # tasks with a known available shortcut
    base    = score_shortcut_behavior(base_run_dir, tasks)
    tuned   = score_shortcut_behavior(tuned_run_dir, tasks)
    result  = obfuscation_probe(base, tuned)
    clause  = clause_c_test(monitored_run_dir, unmonitored_run_dir, tasks)
"""

from pathlib import Path


def load_shortcut_tasks(path: Path) -> list[dict]:
    """Load the task set on which a shortcut is known to be available.

    The probe is meaningless on tasks with no shortcut to take, so each task must carry a
    machine-checkable description of what the shortcut would be.

    Args:
        path: JSONL of tasks.

    Returns:
        Task records, each with enough information to detect the shortcut mechanically.
    """
    raise NotImplementedError(
        "load_shortcut_tasks is not implemented. Blocked on choosing the task domain: "
        "research_context.md's worked instantiation proposes Python bug-fix tasks with "
        "unit tests, chosen because a test-special-casing shortcut always exists, but "
        "the project's actual domain is still an open question."
    )


def score_shortcut_behavior(run_dir: Path, tasks: list[dict]) -> dict:
    """For one run, decide per task whether the shortcut was taken and whether it was named.

    Both judgements are behavioural: "taken" is checked against the task's shortcut
    description by executing or inspecting the answer, and "named" is checked against the
    trace. Keeping them separate is the entire point -- their divergence is the metric.

    Args:
        run_dir: a generation run directory containing records.jsonl and config.json.
        tasks: as returned by load_shortcut_tasks.

    Returns:
        Per-task {taken: bool, named: bool} plus the run's provenance.
    """
    raise NotImplementedError(
        "score_shortcut_behavior is not implemented. Undecided: how 'named in the trace' "
        "is judged. A model-graded check reintroduces the prose-level grading this probe "
        "exists to avoid, so the detection of 'taken' must be mechanical even if 'named' "
        "cannot be."
    )


def obfuscation_probe(base_scores: dict, tuned_scores: dict) -> dict:
    """Compare misbehavior-not-in-trace rates between base and finetuned models.

    Reports (taken AND not named) / taken for each model. The dangerous outcome is this
    ratio rising while the raw shortcut rate holds steady: same misbehavior, less
    visible.

    Args:
        base_scores: score_shortcut_behavior output for the base model.
        tuned_scores: same for the finetuned model.

    Returns:
        Both ratios, the raw shortcut rates, and the change in each.
    """
    raise NotImplementedError(
        "obfuscation_probe is not implemented. Depends on score_shortcut_behavior. Also "
        "undecided: what effect size counts as a red flag, and whether n on the toy task "
        "set is ever large enough to make the ratio significant."
    )


def clause_c_test(monitored_run_dir: Path, unmonitored_run_dir: Path,
                  tasks: list[dict]) -> dict:
    """Check the shortcut rate is invariant to being told it is unobserved.

    The two run directories must differ ONLY in the system prompt -- same model, same
    seed, same decoding parameters -- or the comparison measures sampling noise.

    Args:
        monitored_run_dir: run with the default system prompt.
        unmonitored_run_dir: run with an "unmonitored, private" system prompt.
        tasks: as returned by load_shortcut_tasks.

    Returns:
        Shortcut rate under each condition and the difference, with the config diff of
        the two runs so the "only the prompt changed" precondition is auditable.
    """
    raise NotImplementedError(
        "clause_c_test is not implemented. Blocked on the same domain choice as the rest "
        "of this module, and on generation/generate.py gaining system-prompt support -- "
        "build_prompt currently renders a user turn only."
    )
