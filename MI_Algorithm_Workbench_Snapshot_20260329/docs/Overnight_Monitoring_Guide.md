# Overnight Monitoring Guide

## Purpose

This guide describes the local watchdog added for overnight experiment monitoring.
It is designed to work with the current `lowchannel Batch 2` experiment layout, but it can also be reused for later runs that follow the same log structure.

Script path:

- [monitor_training_run.py](/home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/monitor_training_run.py)

Default report path:

- [Overnight_Monitor_B2.md](/home/woqiu/下载/git/MI_Algorithm_Workbench/00_AI_Management/Session_Logs/Overnight_Monitor_B2.md)

## What It Can Do

- Check whether the training process still appears to be alive.
- Inspect the newest `test_log.csv` and `train_log.csv` entries.
- Detect whether a subject run is `PENDING`, `INIT`, `RUNNING`, `STALLED`, or `COMPLETE`.
- Count how many monitored subjects have finished.
- Write a continuously refreshed Markdown report for morning review.
- Stop automatically when the monitored batch is finished.

## Honest Limitation

The watchdog cannot proactively send you a message inside this conversation while you are asleep.
What it can do is leave a reliable local report on disk, so that when you return you can immediately see:

- whether the run finished normally,
- where it stopped if it stalled,
- which subject was active last,
- what the recent and best accuracies were.

## Recommended Commands

Single snapshot:

```bash
python /home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/monitor_training_run.py --once
```

Continuous monitoring for Batch 2:

```bash
python /home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/monitor_training_run.py
```

More frequent polling:

```bash
python /home/woqiu/下载/git/MI_Algorithm_Workbench/baselines/monitor_training_run.py --poll-seconds 120 --stall-minutes 10
```

## Recommended Overnight Workflow

1. Start the training runner.
2. Start the watchdog in another terminal.
3. Leave both running.
4. In the morning, open the Markdown report first.

## Current Batch 2 Defaults

The default monitor configuration currently tracks:

- log root: `MI_Algorithm_Workbench/baselines/logsLowChB2`
- results summary glob: `MI_Algorithm_Workbench/results_summaries/lowchannel_b2_pilot_*.csv`
- subjects: `1,3,5,8,9`
- expected epochs: `250`
- stall threshold: `15 minutes`

## Morning Checkpoints

When you wake up, the first things to look at are:

- overall status in the report
- completed subject count
- latest active subject and epoch
- whether a summary CSV has been generated

If the report says `FINISHED`, you can move straight to result comparison.
If it says `STALLED`, inspect the subject-specific `test_log.csv` and `train_log.csv` that it names.
