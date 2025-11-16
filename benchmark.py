import time
import subprocess
import threading
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy import stats
import psutil
import os
import signal

# ============================================================
#  Temperature & Power Sampler (macOS powermetrics)
# ============================================================

class PowerSampler:
    def __init__(self, interval=100):
        """
        interval: sampling interval in ms for powermetrics
        """
        self.interval = interval
        self.proc = None
        self.running = False
        self.thread = None
        self.cpu_temps = []
        self.gpu_temps = []
        self.cpu_power = []
        self.gpu_power = []

    def _sample(self):
        # Launch powermetrics
        cmd = [
            "sudo", "powermetrics",
            "--samplers", "all",
            "--show-initial-usage",
            "-i", str(self.interval)
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.running = True

        # Parse lines
        for line in self.proc.stderr:
            if not self.running:
                break

            # CPU temperature
            m = re.search(r"CPU die temperature:\s+([\d\.]+)", line)
            if m:
                self.cpu_temps.append(float(m.group(1)))

            # GPU temperature
            m = re.search(r"GPU die temperature:\s+([\d\.]+)", line)
            if m:
                self.gpu_temps.append(float(m.group(1)))

            # CPU power
            m = re.search(r"CPU Power:\s+([\d\.]+)", line)
            if m:
                self.cpu_power.append(float(m.group(1)))

            # GPU power
            m = re.search(r"GPU Power:\s+([\d\.]+)", line)
            if m:
                self.gpu_power.append(float(m.group(1)))

    def start(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.proc:
            try:
                self.proc.send_signal(signal.SIGINT)
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2)

    def summary(self):
        def safe_avg(arr):
            return np.mean(arr) if arr else float('nan')

        return {
            "cpu_temp_avg": safe_avg(self.cpu_temps),
            "gpu_temp_avg": safe_avg(self.gpu_temps),
            "cpu_power_avg": safe_avg(self.cpu_power),
            "gpu_power_avg": safe_avg(self.gpu_power),
        }


# ============================================================
#  GPU Benchmark (PyTorch + MPS)
# ============================================================

def benchmark_mps(steps=50, seq_len=512, hidden=1024, layers=10, batch=4):

    print("\n██████ GPU (MPS) BENCHMARK ██████")

    device = torch.device("mps")
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(hidden)
            self.fc1 = nn.Linear(hidden, hidden * 4)
            self.fc2 = nn.Linear(hidden * 4, hidden)

        def forward(self, x):
            h = F.gelu(self.fc1(self.ln(x)))
            return x + self.fc2(h)

    model = nn.Sequential(*[Block() for _ in range(layers)]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(batch, seq_len, hidden, device=device)

    times = []

    for i in range(steps):
        t0 = time.perf_counter()
        opt.zero_grad()
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        opt.step()
        t1 = time.perf_counter()

        times.append(t1 - t0)

        if (i+1) % 10 == 0:
            print(f"Step {i+1}/{steps} — {times[-1]:.4f} sec")

    # Statistics using scipy.stats
    mean_time = np.mean(times)
    ci_low, ci_high = stats.t.interval(
        0.95, df=len(times)-1, loc=mean_time, scale=stats.sem(times)
    )

    print("\n--- GPU STATS ---")
    print(f"Mean step time:      {mean_time:.4f} sec")
    print(f"95% CI:              {ci_low:.4f} – {ci_high:.4f}")
    print(f"Throughput:          {1/mean_time:.2f} steps/sec")

    return mean_time, (ci_low, ci_high), times


# ============================================================
#  CPU Benchmark (NumPy Bootstrap)
# ============================================================

def benchmark_bootstrap(reps=20000, n=2000):

    print("\n██████ CPU BOOTSTRAP BENCHMARK ██████")

    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, size=n)
    means = []

    t0 = time.perf_counter()
    for i in range(reps):
        sample = rng.choice(data, size=n, replace=True)
        means.append(sample.mean())
    t1 = time.perf_counter()

    total_time = t1 - t0
    iter_sec = reps / total_time

    # Stats
    ci_low, ci_high = stats.t.interval(
        0.95, df=reps-1, loc=np.mean(means), scale=stats.sem(means)
    )

    print("\n--- CPU STATS ---")
    print(f"Total time:          {total_time:.2f} sec")
    print(f"Iterations/sec:      {iter_sec:.1f}")
    print(f"Bootstrap mean CI:   {ci_low:.4f} – {ci_high:.4f}")

    return total_time, iter_sec


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("Starting Apple Silicon M4 Benchmark...\n")
    print("Note: powermetrics sampling requires sudo privileges.")

    sampler = PowerSampler()

    print("Starting temp/power sampling in background...")
    sampler.start()
    time.sleep(1)

    gpu_mean, gpu_ci, gpu_samples = benchmark_mps()
    cpu_total, cpu_ips = benchmark_bootstrap()

    print("Stopping sampler...")
    sampler.stop()
    summary = sampler.summary()

    print("\n██████ THERMAL / POWER SUMMARY ██████")
    print(f"CPU Temp Avg:    {summary['cpu_temp_avg']:.1f} °C")
    print(f"GPU Temp Avg:    {summary['gpu_temp_avg']:.1f} °C")
    print(f"CPU Power Avg:   {summary['cpu_power_avg']:.1f} W")
    print(f"GPU Power Avg:   {summary['gpu_power_avg']:.1f} W")

    print("\n==================== FINAL SUMMARY ====================")
    print(f"GPU Mean Step Time:     {gpu_mean:.4f} sec")
    print(f"GPU 95% CI:             {gpu_ci[0]:.4f} – {gpu_ci[1]:.4f}")
    print(f"CPU Bootstrap Time:     {cpu_total:.2f} sec")
    print(f"CPU Iter/sec:           {cpu_ips:.1f}")
    print("=======================================================\n")
