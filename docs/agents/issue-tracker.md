# Issue tracker — GitHub

Issues for this repo live in **GitHub Issues**.

## Reading

```bash
gh issue list
gh issue view <number>
```

## Writing

```bash
gh issue create --title "..." --body "..."
gh issue close <number>
gh issue edit <number> --label "needs-triage"
```

## Commands

- `to-issues` — converts a plan/PRD into issues
- `triage` — moves issues through the triage workflow
- `to-prd` — creates a PRD from conversation
- `qa` — logs bugs to the issue tracker

## Required labels

The `triage` skill expects these five labels to exist. Create them once:

```bash
gh label create needs-triage --description "Maintainer needs to evaluate"
gh label create needs-info --description "Waiting on reporter"
gh label create ready-for-agent --description "Fully specified, AFK-ready"
gh label create ready-for-human --description "Needs human implementation"
gh label create wontfix --description "Will not be actioned"
```