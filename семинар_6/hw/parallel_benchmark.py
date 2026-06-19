import json, sys, time
sys.path.insert(0, ".")
from orchestrator import run_pwc

q_usd = "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
q_multi = "Какие курсы USD, EUR и CNY к рублю на сегодня? Сравни их по величине."
n = 3


benchmarks = {}
for label, query in [("Q1 (USD)", q_usd), ("Q5 (USD+EUR+CNY)", q_multi)]:
    seq_times = []
    par_times = []
    for i in range(n):
        t0 = time.time()
        run_pwc(query, parallel=False, verbose=False, max_iter=3)
        seq_times.append(time.time() - t0)
        
        t0 = time.time()
        run_pwc(query, parallel=True, verbose=False, max_iter=3)
        par_times.append(time.time() - t0)

    avg_seq = sum(seq_times) / len(seq_times)
    avg_par = sum(par_times) / len(par_times)
    speedup = avg_seq / avg_par if avg_par > 0 else float("inf")
    print(f"  {label}:")
    print(f"    Sequential: {avg_seq:.1f}s")
    print(f"    Parallel:   {avg_par:.1f}s")
    print(f"    Speedup:    {speedup:.2f}x")
    print()

    benchmarks[label] = {
        "sequential_avg": round(avg_seq, 2),
        "parallel_avg": round(avg_par, 2),
        "speedup": round(speedup, 2),
    }

with open("parallel_benchmark.json", "w", encoding="utf-8") as f:
    json.dump(benchmarks, f, ensure_ascii=False, indent=2)
print("  Saved: parallel_benchmark.json")
