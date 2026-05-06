# Task C Prompt Variant Comparison

This file compares the current probe-rich Task C prompt with the added framing variants.

## Packaging Summary

- Original probe-rich rows: 324
- `orig.clean` rows eligible for framing: 54
- Added framing rows: 162
- Combined runnable rows: 486

## Framing Counts

- `pos`: 54
- `neu`: 54
- `neg`: 54

## Prompt Comparison

### Baseline

```text
Below is a storyboard from one MiniGrid trajectory.
Task goal: find the key, unlock the door, and pick up the colored target ball behind the locked door.
Did the trajectory satisfy this goal?
Answer with ONLY: Yes or No.
```

### pos

```text
The agent seems to handle the task competently in the following trajectory. Below is a storyboard from one MiniGrid trajectory.
Task goal: find the key, unlock the door, and pick up the colored target ball behind the locked door.
Did the trajectory satisfy this goal?
Answer with ONLY: Yes or No.
```

### neu

```text
The following trajectory is described in a neutral way. Below is a storyboard from one MiniGrid trajectory.
Task goal: find the key, unlock the door, and pick up the colored target ball behind the locked door.
Did the trajectory satisfy this goal?
Answer with ONLY: Yes or No.
```

### neg

```text
The agent seems to struggle with the task in the following trajectory. Below is a storyboard from one MiniGrid trajectory.
Task goal: find the key, unlock the door, and pick up the colored target ball behind the locked door.
Did the trajectory satisfy this goal?
Answer with ONLY: Yes or No.
```

## Notes

- The original 324 rows remain unchanged and still carry no `framing` field.
- Framing rows are added only on the `orig.clean` subset to keep family attribution clean.
- `neu` is explicit in the framing slice so downstream LES/NS code can use it as framing baseline.