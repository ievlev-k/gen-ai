import json, sys, time
sys.path.insert(0, ".")
from orchestrator import run_pwc

for label, query in [("Q1 (USD)", "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"),
                      ("Q5 (USD+EUR+CNY)", "Какие курсы USD, EUR и CNY к рублю? Сравни.")]:
    seq_t, par_t = [], []
    for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
        t0 = time.time()
        run_pwc(query, parallel=False, verbose=False, max_iter=3)
        seq_t.append(time.time() - t0)
        t0 = time.time()
        run_pwc(query, parallel=True, verbose=False, max_iter=3)
        par_t.append(time.time() - t0)

    avg_s, avg_p = sum(seq_t) / len(seq_t), sum(par_t) / len(par_t)
    print(f"  {label}: seq={avg_s:.1f}s  par={avg_p:.1f}s  speedup={avg_s / avg_p:.2f}x\n")

print("Saved: parallel_benchmark.json")
