# Triage label vocabulary

Five canonical triage roles used by the `triage` skill:

| Role | Label | Description |
|------|-------|-----------|
| needs-triage | needs-triage | Maintainer needs to evaluate |
| needs-info | needs-info | Waiting on reporter |
| ready-for-agent | ready-for-agent | Fully specified, AFK-ready |
| ready-for-human | ready-for-human | Needs human implementation |
| wontfix | wontfix | Will not be actioned |

## Customization

No overrides configured. To change a label name, edit `docs/agents/triage-labels.md` and update the five role rows above. The skill checks this file before applying any labels.