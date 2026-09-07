# Shadow Branching Protocol Specification

- **Status:** Draft, implementation-aligned specification
- **Specification revision:** 0.3.0
- **Reference implementation:** `aar-ext-shadow-branching` 0.3.0
- **Author:** Florian Fischer
- **License:** MIT

## 1. Purpose

The Shadow Branching Protocol gives a coding-agent harness a Git-native,
time-traversable workspace. It provides:

- per-session isolation from the user's base branch;
- automatic checkpoints after tool calls that change the repository;
- rollback by logical checkpoint rather than raw Git commit count;
- exploratory forks that preserve abandoned or alternative approaches;
- switching between timelines without carrying stale model context forward; and
- explicit squash merge-back to the original base branch.

This document specifies behavior for independent implementations in Aar,
Claude Code, Codex, or other agent harnesses. It is not a system prompt and
does not require slash commands specifically. A harness may expose the
operations as slash commands, tools, API calls, UI actions, or another control
surface, provided their semantics are preserved.

The Aar extension is the reference implementation. Where this specification
describes an optional architecture or a stronger recommendation than the
reference extension currently implements, that distinction is explicit.

## 2. Normative language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD NOT**,
and **MAY** are normative.

An implementation claiming full conformance MUST implement the workspace
operations and the session-context synchronization requirements in this
document. An implementation that provides only Git checkpoints and branches,
but cannot restore the agent's conversation or event context, SHOULD describe
itself as a **workspace-only implementation**, not a fully conforming Shadow
Branching implementation.

## 3. Scope and non-goals

The protocol controls versioned workspace state and the agent context associated
with that state. It is not:

- a process sandbox or permission system;
- a secret scanner;
- a replacement for backups or remote Git hosting;
- a guarantee that untracked ignored files can be restored;
- a merge-conflict resolver; or
- by itself, a concurrent multi-session solution in a single working tree.

A standard single-worktree implementation changes the repository's checked-out
branch for the whole working directory. Concurrent sessions require separate
Git worktrees or another isolation mechanism; see Section 15.

## 4. Host requirements

A conforming host MUST provide:

1. a stable session identifier;
2. a way to run Git commands and inspect their exit status;
3. a callback after each tool invocation, including shell commands;
4. a way to detect whether the working tree changed;
5. durable session context, or a way to reconstruct the model-visible event
   history for a selected timeline; and
6. a way to replace or restart the live model context after a timeline-changing
   operation.

The reference storage format is an append-only session JSONL file located in
the repository and committed on the shadow timeline. Other formats are valid.
For example, a harness using SQLite MAY materialize branch-local context
snapshots and restore the live conversation from them.

If the host cannot replace live context in place, it MAY satisfy requirement 6
by terminating the current agent loop and starting a new loop from the restored
context before the next model invocation.

## 5. Terminology

### 5.1 Base branch

The branch checked out when a fresh session starts, such as `main` or a feature
branch. The protocol records this branch in an anchor commit. Agent work MUST
NOT be committed directly to the base branch before explicit finalization.

### 5.2 Canonical shadow branch

The session's primary branch:

```text
shadow/session-<sid>
```

`<sid>` MUST be stable for the session and valid inside a Git ref. A harness
whose native session identifiers are not ref-safe MUST encode them with a
stable, collision-resistant mapping.

### 5.3 Preserved branch

A timeline preserved by the Branch operation. For the canonical timeline, the
usual form is:

```text
shadow/session-<sid>-branch-<K>
```

The reference implementation can also branch from a preserved branch. It
appends another suffix to the current timeline name, for example:

```text
shadow/session-<sid>-branch-1-branch-2
```

Generated branch numbers MUST be monotonic within the session namespace and
MUST be derived from refs on disk, not solely from process memory.

### 5.4 Session namespace

A branch belongs to session `<sid>` only when it is either:

```text
shadow/session-<sid>
```

or starts with the boundary-preserving prefix:

```text
shadow/session-<sid>-
```

Implementations MUST use an exact boundary. Substring or unbounded prefix
matching can confuse identifiers such as `s1` and `s10`.

### 5.5 Anchor commit

The first **session-specific** commit on a fresh shadow timeline. It is an empty
commit with the exact subject:

```text
shadow-init: base=<base-branch>
```

The anchor is not the root commit of the repository: the shadow branch inherits
the base branch's existing history.

### 5.6 Checkpoint commit

A commit created after a tool result when changes stageable by `git add -A` are
present:

```text
shadow-auto: <tool> turn-<N>
```

Only these commits count as logical checkpoints.

### 5.7 Meta commit

A housekeeping commit that persists session context or clears pending changes
before a history operation:

```text
shadow-meta: <label>
```

Meta commits MUST NOT count as logical checkpoints.

### 5.8 Timeline-changing operation

Any operation that can make the current files or branch history refer to an
earlier or different timeline. In this protocol these are Undo, Branch, and
Switch. Finalize also changes branches but ends active shadow execution.

## 6. Commit taxonomy and reserved subjects

| Subject | Meaning | Logical checkpoint |
|---|---|---:|
| `shadow-init: base=<branch>` | Records the merge-back target | No |
| `shadow-auto: <tool> turn-<N>` | Captures changes observed after a tool call | Yes |
| `shadow-meta: <label>` | Persists context or pending housekeeping changes | No |

Implementations MUST classify commits by their subject prefix, not by their
position relative to `HEAD`. Raw expressions such as `HEAD~N` are not valid
logical-checkpoint calculations because meta commits may be interleaved.

Tool names and labels SHOULD be sanitized to one line. Portable implementations
SHOULD retain full commit hashes; the Aar reference stores short hashes and uses
them for both state and display.

These prefixes are reserved. Repositories using them for unrelated commits on
the same history can make reconstruction ambiguous.

## 7. Protocol state and sources of truth

A host SHOULD maintain the following live state:

```text
session_id       stable host session identifier
base_branch      branch recorded by the anchor
current_branch   checked-out shadow or preserved branch
branch_counter   greatest generated branch suffix observed
turn_counter     number of logical checkpoints on the active timeline
checkpoints      ordered records: (logical turn, commit hash, tool name)
anchor_hash      hash of the selected shadow-init commit
enabled          whether Git checkpointing is active
mode             lifecycle profile, normally "git", "fallback", or "done"

optional checkpoint presentation fields may include changed-file count and a
sensitive-path flag
```

The reference extension mirrors this state to
`session.metadata["shadow_branching"]` using the concrete field names
`session_id`, `original_branch`, `shadow_branch`, `branch_counter`,
`turn_counter`, `checkpoints`, `base_anchor`, `enabled`, and `mode`. Its
checkpoint records use short hashes and may also include `files` and `flagged`.
Metadata is a cache and audit record, not the sole recovery source. A restarted implementation MUST be able to
reconstruct branch, base, and checkpoint state from Git refs and commit
subjects. It MAY additionally restore the last active preserved branch from
durable host metadata; the Aar reference implementation resumes the canonical
shadow branch after a process restart.

After a successful state transition, the implementation SHOULD persist updated
protocol metadata before the next model invocation.

## 8. Session–branch consistency invariant

Workspace state and model context form one logical timeline. Changing only the
Git branch is insufficient: the model may otherwise remember files or tool
results that do not exist on the checked-out branch.

A conforming implementation MUST maintain this invariant:

> Before the model is invoked after a timeline-changing operation, its live
> event history MUST be restored from the session snapshot reachable on the
> currently checked-out timeline. It MUST NOT contain events that exist only on
> the discarded or previously checked-out timeline.

During normal forward execution, the live event list MAY contain an unsaved,
append-only tail generated on the current timeline. Immediately after a reload,
the live history SHOULD exactly equal the restored snapshot. If no snapshot is
reachable at the selected boundary, the host MUST clear the live history or
restart with an empty history rather than retain stale events.

A session snapshot can legitimately predate the selected checkpoint. Restoring
that older prefix is conservative: it may forget valid context, but it does not
inject knowledge from a future or different timeline.

### 8.1 Required reload behavior

After Undo, Branch, or Switch, the implementation MUST:

1. load the session snapshot visible at the new `HEAD`;
2. replace the live events and step counter, rather than append to them;
3. clear events and reset the step counter if no snapshot exists;
4. restore ordinary host metadata from the snapshot as appropriate; and
5. preserve or overwrite the shadow-protocol metadata with the newly computed
   live protocol state.

Replacement SHOULD occur in place when other host components retain references
to the session object. Restarting the agent loop from the restored snapshot is
also conforming.

### 8.2 Re-entry guard

Some harnesses emit a session-start event on every user turn. Once protocol
state is active for a session, repeated initialization for that same session
MUST NOT check out the canonical branch or reconstruct state again. It SHOULD
only synchronize metadata and return.

Without this guard, the next user turn can silently undo a prior Switch.

## 9. Logical checkpoint boundaries

Let `Boundary(m, branch)` be the commit representing a timeline with exactly
`m` logical checkpoints retained.

A conforming implementation MUST compute it from first-parent history in
oldest-to-newest order:

```text
seen = 0
previous = none
last = none

for each commit (hash, subject) on branch, oldest first:
    if subject starts with "shadow-auto:":
        seen += 1
        if seen > m:
            return previous
    previous = hash
    last = hash

return last
```

The selected history SHOULD begin at the session anchor. The reserved-prefix
rule in Section 6 is REQUIRED if an implementation scans older base history as
the Aar reference implementation does.

This algorithm deliberately retains meta commits after the `m`th checkpoint
and before the next checkpoint. Those commits may contain the context snapshot
that belongs to the retained timeline.

## 10. Session initialization

Initialization MUST be idempotent.

### 10.1 Preflight

The implementation MUST:

1. determine whether the working directory is inside a Git work tree;
2. verify that Git commands can create commits, including an available author
   identity or an implementation-supplied identity;
3. ensure the repository has at least one commit; and
4. determine the current branch.

For an empty repository, an implementation MAY create `main` and an empty
`Initial commit`, matching the reference implementation.

If the directory is not a Git repository, the Git profile MUST report itself as
disabled. The Aar extension creates `.shadow_backups/` as a marker but does not
create fallback snapshots. An implementation MAY define a separate filesystem
snapshot profile, but creating an empty directory alone is not checkpointing
and MUST NOT be presented as equivalent protection.

### 10.2 Cross-session start guard

If `HEAD` is on a shadow branch belonging to a different session, a new session
MUST NOT branch from that stale timeline silently. It MUST either:

- switch to the base branch recorded in that timeline's anchor; or
- stop and require an explicit base selection.

The reference implementation automatically checks out the recorded base and
logs a warning. If that checkout fails, it warns and continues from the current
`HEAD`; stricter implementations SHOULD fail closed instead.

### 10.3 Resume

If the canonical branch for the same session already exists, the implementation
MUST:

1. check it out, unless durable host state explicitly selects another branch in
   the same session namespace;
2. read the oldest reachable `shadow-init:` subject and recover the base branch;
3. reconstruct checkpoints from `shadow-auto:` commits;
4. derive the greatest branch number from refs on disk; and
5. synchronize protocol metadata.

Prior branches from other sessions MAY be listed for information but MUST NOT be
resumed solely because they exist. Session identity selects the branch.

### 10.4 Fresh session

For a fresh session, the implementation MUST:

1. capture the current branch as the base;
2. create `shadow/session-<sid>` from the current `HEAD`;
3. create an empty `shadow-init: base=<base>` anchor commit;
4. record its hash; and
5. initialize an empty checkpoint list with turn counter zero.

The implementation MUST check every Git exit status. It MUST NOT report the
session active if branch creation failed. It SHOULD also fail closed if anchor
creation fails, because resume and Finalize otherwise lose their durable source
of truth. The Aar reference implementation currently logs an anchor failure and
continues with its in-memory base as a fallback.

## 11. Forward execution and persistence

### 11.1 Checkpoint after a tool result

After every tool result, the implementation MUST inspect the working tree. If
there are no changes stageable by `git add -A`, it MUST NOT create a checkpoint.

If changes exist, it MUST:

1. inspect the changed paths;
2. warn when names look sensitive, at minimum for `.env`, `.key`,
   `credentials`, `id_rsa`, and `secret` patterns;
3. stage all changes accepted by `git add -A`; this includes tracked changes
   and deletions even when a later ignore rule matches the path, but excludes
   untracked ignored files;
4. verify that the index contains staged changes;
5. create `shadow-auto: <tool> turn-<N>` where `N` is the next logical turn;
6. update state only after the commit succeeds; and
7. emit or log a checkpoint record containing turn, hash, and tool name.

Staging all changes is intentional: shell tools can create side effects outside
the paths declared in their arguments. The checkpoint therefore captures all
pending stageable repository changes observed at the hook, not necessarily only
the exact delta caused by that tool. Implementations MUST document this
attribution limitation.

A sensitive-path warning is not a security boundary. The Aar reference
implementation warns and proceeds without confirmation. Stricter hosts MAY
require approval or deny staging.

Untracked ignored paths are outside the protocol's recovery guarantee. A tool
call that changes only untracked ignored files creates no checkpoint. Changes
or deletions to already tracked files remain stageable even if an ignore rule
now matches them.

### 11.2 Meta commits and session saves

The branch-local session snapshot must be committed often enough to restore a
safe context prefix after a timeline operation. Implementations SHOULD commit a
snapshot immediately after the host saves it:

```text
shadow-meta: session-saved
```

The Aar host saves JSONL after its agent-loop lifecycle events have completed,
so the reference extension installs an idempotent post-save hook around
`SessionStore.save()`. It also performs defensive sweeps:

- before a new turn: `shadow-meta: turn sync`;
- at session end: `shadow-meta: session sync`; and
- before branch operations: `shadow-meta: pre-branch sync`,
  `shadow-meta: pre-switch sync`, or `shadow-meta: pre-done sync`.

Other hosts need not use these exact hook names, but MUST ensure pending context
and workspace state are committed before a checkout, branch rename, reset, or
final merge. A failed sweep MUST leave the operation stopped if the working
tree remains dirty.

A generic pending-change sweep uses this behavior:

```text
if work tree clean: return no-op
git add -A
if index clean: return no-op
git commit -m "shadow-meta: <label>"
```

Because this stages all pending changes accepted by `git add -A`, it may include
more than session bookkeeping. Hosts SHOULD avoid unrelated user edits in an active
shadow worktree and SHOULD surface changed paths when a sweep fails or captures
unexpected content.

## 12. Timeline operations

The operation names below are normative concepts. Slash-command spellings are
the reference user interface.

### 12.1 Undo — `/undo [N] [--force]`

Undo removes the latest `N` logical checkpoints from the active timeline.
`/revert` MAY be provided as an alias.

The implementation MUST:

1. default `N` to 1 and reject values beyond the checkpoint count;
2. refuse a dirty working tree unless force was explicitly requested;
3. compute `m = checkpoint_count - N`;
4. resolve `target = Boundary(m, current_branch)`;
5. run `git reset --hard <target>`;
6. when force was requested for a dirty tree, remove untracked, non-ignored
   files with semantics equivalent to `git clean -fd`;
7. truncate the checkpoint list and set the logical turn count to `m`;
8. reload session context from the restored timeline; and
9. persist updated protocol metadata.

`--force` is destructive. Ignored files are preserved by `git clean -fd`, but
tracked and untracked non-ignored work can be lost.

### 12.2 Branch — `/branch [N]`

Branch preserves the current attempt and creates a fresh timeline, optionally
rewound by `N` logical checkpoints.

Before modifying refs, the implementation MUST commit pending state and MUST
stop if the tree remains dirty.

It MUST then:

1. validate `N`; omitted or zero means branch from current `HEAD`;
2. for `N > 0`, resolve
   `fork_point = Boundary(checkpoint_count - N, current_branch)`;
3. derive `K` as one greater than the greatest generated branch suffix found in
   the session's refs;
4. preserve the current branch by renaming it to
   `<current-branch>-branch-<K>`;
5. recreate `<current-branch>` at `fork_point` and check it out;
6. roll back the rename if creating the fresh branch fails;
7. for `N > 0`, truncate logical checkpoint state to the fork point;
8. reload session context from the snapshot reachable at the fork point; and
9. persist updated protocol metadata.

When invoked on the canonical branch, this produces the common pair:

```text
preserved: shadow/session-<sid>-branch-<K>
active:    shadow/session-<sid>
```

When invoked while a preserved branch is active, the reference implementation
uses nested names as described in Section 5.3. A simpler implementation MAY
restrict Branch to the canonical timeline, but it MUST reject unsupported use
explicitly rather than overwrite an existing ref.

Checkpoint numbers may be reused on divergent timelines. Commit hashes and
branch names disambiguate them.

### 12.3 Switch — `/switch <target>`

Switch changes to another existing timeline in the same session namespace.

The implementation MUST:

1. resolve any UI shorthand to a full branch name;
2. reject branches outside the exact session namespace;
3. commit pending state and stop if the tree remains dirty;
4. verify the target exists;
5. check out the target;
6. reconstruct base, checkpoint count, and checkpoint records from its history;
7. reload session context from the target timeline; and
8. persist updated protocol metadata.

The reference command accepts:

| Input | Resolution |
|---|---|
| no argument | Show current branch and available targets; do not switch |
| `main`, `active`, or `shadow` | Canonical `shadow/session-<sid>` |
| `3` or `branch-3` | `shadow/session-<sid>-branch-3` |
| full branch name | Use verbatim after namespace validation |

Numeric shorthands identify only direct canonical branches. Nested preserved
branches require their full name.

### 12.4 List branches — `/branches`

The implementation SHOULD provide a way to list all refs in the current
session namespace and mark the active branch. The output format is not
normative.

### 12.5 Finalize — `/done [message] [--yes]`

Finalize squash-merges only the active timeline into the base branch.

The implementation MUST:

1. commit pending state and stop if the tree remains dirty;
2. detect preserved branches;
3. if preserved branches exist, require explicit confirmation such as `--yes`;
4. recover the base branch from the oldest reachable anchor subject;
5. check out the base branch;
6. run `git merge --squash <active-shadow-branch>`;
7. inspect unresolved paths with semantics equivalent to
   `git diff --name-only --diff-filter=U`;
8. if conflicts exist, stop without committing and report the paths;
9. otherwise create one final commit using the supplied or host-generated
   message; and
10. mark shadow execution inactive or end the session before another tool or
    persistence hook can run on the now-checked-out base branch.

The reference default message is:

```text
Aar session <sid>: <N> checkpoint(s)
```

Conflict handling does not restore the pre-merge state. The reference
implementation leaves the base branch checked out with unresolved merge state
for manual resolution. Implementations MUST describe whether they leave this
state intact or abort it; they MUST NOT claim success.

Finalize MUST NOT silently merge, delete, or combine preserved alternatives.
The Aar reference implementation leaves all shadow and preserved branches on
disk after success so recovery remains possible. Cleanup is a separate,
explicit operation.

The Aar extension sets `enabled=false` and `mode="done"` after a successful
`/done`, removes the session from its active save-hook registry, and preserves
that terminal marker in session metadata. Repeated initialization and cold
resume of the same merged session remain disarmed.

## 13. Failure handling and crash recovery

Every Git command MUST be checked. Implementations SHOULD apply a finite timeout
and return actionable errors for missing Git, lock contention, hooks, identity
failure, and large-tree staging timeouts.

State changes MUST follow successful Git changes, not precede them. In
particular, a failed checkpoint commit MUST NOT increment the logical turn, and
a failed Branch checkout MUST attempt to restore the original ref name.

On restart, implementations SHOULD prefer recovery from durable facts in this
order:

1. existing refs and current `HEAD`;
2. the anchor and reserved commit subjects;
3. branch-local session snapshots; and
4. cached protocol metadata.

Cached counters MUST NOT cause branch-name reuse when higher numbered refs
already exist.

Git hooks can be interactive or mutate files. Automated implementations SHOULD
use non-interactive Git configuration and MAY bypass commit hooks with
`--no-verify`, as the Aar reference extension does. Hosts that require hooks
MUST account for their side effects and failure modes.

## 14. Security and data-retention considerations

- `git add -A` captures all pending stageable changes, including unrelated
  edits. It excludes untracked ignored paths but still captures tracked paths
  that match ignore rules. Run the protocol in a dedicated working tree when possible.
- Sensitive-file warnings are heuristic. They do not prevent secrets from
  entering `.git/objects/`, reflogs, backups, or later pushes.
- Implementations MUST NOT push shadow refs automatically unless the user has
  explicitly enabled that behavior.
- Deleting a shadow branch does not immediately erase its objects. Sensitive
  data may remain recoverable until reflog expiration and garbage collection.
- Undo with force can destroy uncommitted work.
- A malicious or malformed session identifier must not be interpolated into a
  ref or shell command without validation and safe argument passing.
- Git commands SHOULD be invoked as argument arrays, not shell-interpolated
  command strings.

## 15. Optional worktree architecture for concurrency

Worktrees are not required for core protocol conformance and are not currently
implemented by `aar-ext-shadow-branching`. They are the recommended deployment
model for concurrent sessions.

A worktree-based host creates one working directory and index per session while
sharing the repository object store:

```text
main worktree       HEAD -> user's base branch
session worktree A  HEAD -> shadow/session-s1
session worktree B  HEAD -> shadow/session-s2
```

A conforming worktree profile SHOULD:

1. create or select the session shadow ref before adding the worktree;
2. run all tool calls and checkpoints inside that session's worktree;
3. keep the user's main worktree unchanged during agent execution;
4. avoid checking out one branch in multiple worktrees; and
5. perform Finalize from the main worktree or a temporary merge worktree,
   because Git will not allow the base branch to be checked out concurrently in
   the session worktree.

Worktrees avoid Git checkout contention and cross-session file contamination,
but each worktree still has a full checked-out file tree and its own index.
"Shared object store" does not mean zero filesystem cost.

## 16. Conformance checklist

A full implementation can use this checklist:

- [ ] Session IDs map safely and stably to Git refs.
- [ ] Fresh sessions create a canonical shadow branch and anchor.
- [ ] Base branches remain untouched until explicit Finalize.
- [ ] Tool results create `shadow-auto` commits only when staged changes exist.
- [ ] All changes accepted by `git add -A` are included and attribution limits are documented.
- [ ] Session snapshots are persisted as non-checkpoint `shadow-meta` commits or an equivalent timeline-bound mechanism.
- [ ] Undo and Branch count `shadow-auto` commits, not raw parents.
- [ ] Undo, Branch, and Switch restore model-visible context before the next model call.
- [ ] Missing snapshots clear context rather than retaining stale events.
- [ ] Repeated session-start events do not undo an active Switch.
- [ ] Cross-session namespace checks use exact boundaries.
- [ ] Branch numbers are derived from durable refs.
- [ ] Dirty-tree and Git-failure paths stop safely.
- [ ] Finalize squash-merges only the active timeline and reports conflicts without claiming success.
- [ ] Successful Finalize persists a terminal state and disarms all checkpoint and save hooks.
- [ ] Preserved branches are never silently discarded.
- [ ] Concurrent sessions either use separate worktrees or are explicitly unsupported.

## 17. Reference Aar integration

The Aar extension maps the protocol to these host surfaces:

| Protocol event or operation | Aar integration |
|---|---|
| Initialize / re-entry guard | `session_start` hook |
| Defensive pending-state sweep | `before_turn` and `session_end` hooks |
| Checkpoint | `tool_result` hook |
| Immediate context persistence | idempotent `SessionStore.save()` post-save hook |
| Undo | `/undo`, `/revert` |
| Branch | `/branch` |
| Switch | `/switch` |
| List | `/branches` |
| Finalize | `/done` |
| Live protocol cache | `session.metadata["shadow_branching"]` |
| Interactive tree and actions | Optional `UIPanel` for fixed TUI plus ACP stdio and HTTP |

Version 0.3.0 adds a guarded panel integration. Its snapshot presents the base,
active and preserved branches, newest-first checkpoints, pending changes, and
an inactive post-Finalize state. Panel actions wrap the same Undo, Branch,
Switch, and Finalize implementations; read-only Diff and Refresh plus explicit
preserved-branch deletion are presentation-layer additions. Checkpoints created
in the current process cache changed-file counts and sensitive-path flags; the
panel backfills those details from `git show` after resume.

### 17.1 Known Aar conformance gaps

The current Aar extension demonstrates the protocol but still has failure paths
that do not satisfy every **MUST** in this specification:

- a failed session reload logs a warning but Undo, Branch, or Switch still
  reports success, so the live context may remain stale;
- reload updates ordinary metadata in place instead of replacing it, so keys
  present only on the source timeline can survive; the missing-snapshot path
  clears events and step count but leaves ordinary metadata untouched;
- resume initialization reconstructs the canonical Git timeline but does not
  itself reload an already-live session object;
- a failed cross-session checkout logs a warning and initializes from the stale
  `HEAD` instead of failing closed;
- an anchor-commit failure is logged but does not disable the session;
- native session identifiers are interpolated into Git ref names without a
  ref-safety encoding or validation layer;
- several secondary recovery and probe commands are not checked, including the
  empty-repository bootstrap, forced-clean result, rename rollback result, and
  conflict-probe status;
- if `/done` reaches the base branch but then stops on a merge conflict or final
  commit failure, the extension remains enabled; a later automatic session save
  can therefore create a `shadow-meta:` commit on the base branch; and
- `/done` message parsing removes flags and integer-only tokens rather than
  preserving every supplied message token verbatim.

Portable implementations SHOULD follow the normative behavior above rather
than reproduce these gaps.

The implementation and its mostly real-Git end-to-end tests live in:

- `aar_ext_shadow_branching/__init__.py`
- `tests/test_shadow_branching.py`
- `tests/test_shadow_branching_panel.py`

This section describes the reference profile, not additional requirements on
other harnesses.
