---
name: Your Root Cause Name Here
severity: warning
tags: [gpu, performance, example]
detection_skill: top_kernels
---

## Symptom

Describe what the profile looks like when this problem occurs.
What does the user see in the timeline or skill output?

## Why It Happens

Explain the mechanism — why does this happen technically?

## How to Detect

Which nsys-ai skill or query can detect this?

```bash
nsys-ai skill run <skill_name> profile.sqlite
```

What should the user look for in the output?

## How to Fix

Provide actionable fix steps.

## Real-World Example

(Optional) Describe a real-world case where this was encountered.
