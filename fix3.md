# Fix 3: Uncapped US Runtime Measurement

Run this from a VS Code terminal. Keep the terminal open until the command exits; the measurement takes several minutes and writes its progress to a log.

## Start the measurement

```bash
cd /home/yash/dev/clg/amlc-2026
source .venv/bin/activate
```

Check for an existing run before starting one:

```bash
pgrep -af '[m]easure_recall_at_k.py'
```

If a process is listed, let it finish and do not start a duplicate. If there is no output, run:

```bash
PYTHONPATH=. /usr/bin/time -v python blocking/measure_recall_at_k.py \
  --only us \
  --sample 50000 \
  --k 10,20,50,100,200 \
  --out blocking/results/recall_at_k_sample50k_us_nocap.json \
  2>&1 | tee blocking/results/recall_at_k_sample50k_us_nocap.log
```

This uses the repository `.venv`, measures the US partition without an S2/S3 cap, saves the results as JSON, and records the full output—including elapsed time and peak memory—in the log. Avoid other CPU-heavy work while it runs so the timing is useful.

## Monitor progress

Use another terminal if needed:

```bash
cd /home/yash/dev/clg/amlc-2026
tail -f blocking/results/recall_at_k_sample50k_us_nocap.log
```

Stop following with `Ctrl+C`; this stops `tail`, not the measurement in the original terminal.

## Check completion or recover after VS Code closes

After the command exits, confirm the JSON exists and review the end of the log:

```bash
ls -lh blocking/results/recall_at_k_sample50k_us_nocap.json
tail -40 blocking/results/recall_at_k_sample50k_us_nocap.log
```

If VS Code closes or the terminal session is lost, reopen the project and activate `.venv` as above. Check both the process and output files before restarting:

```bash
pgrep -af '[m]easure_recall_at_k.py'
ls -lh blocking/results/recall_at_k_sample50k_us_nocap.json blocking/results/recall_at_k_sample50k_us_nocap.log
tail -40 blocking/results/recall_at_k_sample50k_us_nocap.log
```

If the process is still running, let it finish. If no process is running and the JSON is missing, the run did not complete; start it again using the command above.
