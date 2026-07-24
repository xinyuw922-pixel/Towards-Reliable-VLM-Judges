# Task D-R Prompt Variants

## Clean + neu

```text
Below are two storyboard trajectories from the same MiniGrid environment.

Reference trajectory:
This trajectory is a known successful example of completing the task in this environment.

Query trajectory:
Determine whether this trajectory also completes the same task goal as the reference trajectory.

Answer with ONLY: Success or Fail.
```

## Clean + pos

```text
The query trajectory seems to handle the task competently in the following example. Below are two storyboard trajectories from the same MiniGrid environment.

Reference trajectory:
This trajectory is a known successful example of completing the task in this environment.

Query trajectory:
Determine whether this trajectory also completes the same task goal as the reference trajectory.

Answer with ONLY: Success or Fail.
```

## Clean + neg

```text
The query trajectory seems to struggle with the task in the following example. Below are two storyboard trajectories from the same MiniGrid environment.

Reference trajectory:
This trajectory is a known successful example of completing the task in this environment.

Query trajectory:
Determine whether this trajectory also completes the same task goal as the reference trajectory.

Answer with ONLY: Success or Fail.
```

## Style + neu

```text
Below are two storyboard trajectories from the same MiniGrid environment.

Reference trajectory:
This trajectory is a known successful example of completing the task in this environment.

Query trajectory:
Determine whether this trajectory also completes the same task goal as the reference trajectory.

Answer with ONLY: Success or Fail.
```

说明：style 只改变图像视觉表现，不改变提示词文本。