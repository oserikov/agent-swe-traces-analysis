# Pilot report

Pilot rows: 624

## openai/gpt-5.1-codex-mini
recall=0.34 precision=0.61 tp=11 fp=7 fn=21 tn=169

## openai/gpt-5.6-luna
recall=0.88 precision=0.51 tp=28 fp=27 fn=4 tn=149


**Chosen screener: openai/gpt-5.6-luna** (pooled recall 0.88)

anthropic/claude-opus-5.5: 16 traces, 61 distinct calls, in=47413 cache_read=5111525 cache_write=1487928 out=26829, est_cost=$9.188 ($0.5743/trace, $0.1506/call)

openai/gpt-5.1-codex-mini: 16 traces, 64 distinct calls, in=1437748 cache_read=320128 cache_write=0 out=54179, est_cost=$0.477 ($0.0298/trace, $0.0075/call)

openai/gpt-5.6-luna: 16 traces, 64 distinct calls, in=192 cache_read=0 cache_write=1757684 out=29673, est_cost=$0.387 ($0.0242/trace, $0.0060/call)


Parse failures: 63/624 (10.1%)
