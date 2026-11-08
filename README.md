# mm-engine

Motor de market making estatístico para um único instrumento: feed ITCH 5.0 → livro L3 → features de microestrutura → filtro de Kalman estendido + GBDT de toxicidade → cotação Avellaneda-Stoikov. Hot path sem alocação de heap, sem `HashMap`, sem `unwrap`/`expect` em código de produção.

Decisões não óbvias, alternativas avaliadas e medições: [`docs/design-decisions.md`](docs/design-decisions.md).

## Workspace

| Crate | Responsabilidade |
|---|---|
| `core-types` | `Price`/`Qty` em ponto fixo `i64 × 10⁸`, `OrderId`, `Side`, `MarketEvent` (64 B), `Quote` (64 B), `DepthSnapshot`, `FeatureVector` (64 B), `CachePadded<T>` (`#[repr(C, align(64))]`). |
| `orderbook-l3` | Livro L3 prioridade preço-tempo com motor de matching. Arena de ordens, índice open-addressing, ladder densa de níveis com bitmap hierárquico. |
| `microstructure-simd` | OFI multinível, micro-price ponderado por profundidade, imbalance, VPIN, volatilidade realizada, fluxo agressor. Kernels em `wide::f64x8`. |
| `ml-alpha` | EKF de 3 estados (fair value, drift, log-variância); avaliador de floresta oblivious em layout SoA vetorizado; modelo estático gerado; trainer offline (feature `trainer`). |
| `strategy-mm` | Avellaneda-Stoikov com overlay de toxicidade, limites de inventário, clamp post-only. |
| `engine-runtime` | Codec ITCH 5.0, ring SPSC, `StrategyEngine` (Thread 2), pipeline de 3 threads com CPU pinning, telemetria, simulador agent-based, benchmarks, binários `mm-engine` e `train-toxicity`. |

Rust ≥ 1.89 (MSRV; edition 2024). Dependências externas de produção: `thiserror`, `wide`, `core_affinity`. `unsafe` restrito a `engine-runtime/src/ring.rs` (slots do SPSC) e `clock.rs` (`RDTSC`); demais crates usam `#![forbid(unsafe_code)]`.

## Topologia

```
  ITCH 5.0 (Arc<[u8]>, frames com prefixo u16 BE)
        |
+-------v------------------------------------------------------------+
| T1 md-ingest            core lógico 2                              |
|   RDTSC -> FrameCursor::next_event -> decode_message               |
|   -> MarketEvent { seq, exch_ts, recv_tsc, body }  (64 B, 1 linha)  |
+-------+------------------------------------------------------------+
        | SPSC A  2^16 slots × 64 B   head/tail em CachePadded distintos
        | push_spin (backpressure)    cursor remoto cacheado por lado
+-------v------------------------------------------------------------+
| T2 strategy             core lógico 4      StrategyEngine::on_event |
|   OrderBook::apply ----FillSink----> FeatureEngine::on_trade        |
|        |                             (VPIN, fluxo, paper fills)     |
|   OrderBook::snapshot (8 níveis/lado, SoA)                          |
|   FeatureEngine::update (f64x8) -> FairValueEkf::observe            |
|   predict_toxicity (GBDT f64x8/u64x8) -> AvellanedaStoikov::quote   |
|   -> Quote (64 B)                                                   |
+-------+------------------------------------------------------------+
        | SPSC B  2^16 slots × 128 B  push não-bloqueante (descarte contado)
+-------v------------------------------------------------------------+
| T3 telemetry            core lógico 6                              |
|   histogramas log-lineares (3 776 buckets, erro < 1.6 %), CSV       |
+--------------------------------------------------------------------+
```

Memória da Thread 2, alocada uma vez em `StrategyEngine::new` (configuração padrão):

```
OrderBook
  OrderArena     Box<[OrderNode]>    131 072 × 32 B = 4 MiB    free-list intrusiva LIFO
  OrderIndex     Box<[Slot]>         262 144 × 16 B = 4 MiB    linear probing, carga <= 0.5
  bids, asks     Box<[PriceLevel]>    65 536 × 24 B = 1.5 MiB/lado
                 LevelBitmap         l0: 1 024 u64, l1: 16 u64, l2: 1 u64
FeatureEngine    4 × f64x8 + Vpin<50> ([i64; 50])               inline
FairValueEkf     [f64; 3], [[f64; 3]; 3]                        inline
TOXICITY_FOREST  static: 6 blocos × 5 níveis × (u64x8 + f64x8) + 48 × 32 folhas ≈ 16 KiB
```

`tests/zero_alloc.rs` instala um `GlobalAlloc` contador por thread e processa 250 000 mensagens pelo caminho completo (decode → SPSC → `on_event`): zero alocações; `reset()` também não aloca.

## Livro L3

- Preço → nível: `idx = (p − p_min) / tick` sobre ladder densa de até 262 144 ticks por lado (padrão 65 536). Preço fora da ladder ou fora do grid é rejeitado antes de qualquer mutação.
- Nível: lista FIFO duplamente ligada intrusiva (`prev`/`next: u32` no `OrderNode`), agregados `total`/`count`.
- Melhor preço: bitmap de 3 níveis (64³ bits). `min`, `max`, `next_above`, `next_below` em no máximo 3 `lzcnt`/`tzcnt`, independente da distância ao próximo nível ocupado.
- `OrderId → slot`: tabela open-addressing com hash de Fibonacci e deleção por backward shift (sem tombstones).
- Operações: `submit` (GTC/IOC com matching), `cancel`, `reduce` (ITCH `X`, preserva prioridade), `execute` (ITCH `E`), `replace` (ITCH `U`, perde prioridade), `apply(EventBody)`. Toda validação precede a mutação; em `Err` o livro não muda.
- Complexidade: add/cancel/reduce/execute O(1) esperado; matching O(1) por fill + O(1) por nível esvaziado; snapshot O(8).

## Features

Preços em ticks, quantidades em unidades, decaimento por tempo de bolsa `A ← A·e^{−Δt/τ} + x` (τ_fluxo = 0.25 s, τ_vol = 5 s). Pesos de nível `w_m = e^{−0.4 m}`.

$$e^b_m = \mathbb{1}\{P^b_m \ge P^{b\prime}_m\}\,q^b_m - \mathbb{1}\{P^b_m \le P^{b\prime}_m\}\,q^{b\prime}_m,\qquad e^a_m = \mathbb{1}\{P^a_m \le P^{a\prime}_m\}\,q^a_m - \mathbb{1}\{P^a_m \ge P^{a\prime}_m\}\,q^{a\prime}_m$$

$$\mathrm{OFI}_m = e^b_m - e^a_m$$

$$D_b = \sum_m w_m q^b_m,\quad V_b = \frac{\sum_m w_m q^b_m P^b_m}{D_b},\quad \text{micro} = \frac{V_a D_b + V_b D_a}{D_a + D_b}\ \text{(clamp em } [P^b_1, P^a_1]\text{)},\quad I = \frac{D_b - D_a}{D_b + D_a}$$

$$\mathrm{VPIN} = \frac{1}{N V}\sum_{\tau=n-N+1}^{n} \left|V^B_\tau - V^S_\tau\right|,\qquad N = 50,\ V = 25\ \text{lotes}$$

Níveis ausentes carregam sentinelas finitas (`i64::MIN`/`i64::MAX`, quantidade 0), o que mantém todos os kernels sem branch nas 8 lanes. VPIN usa lado agressor do próprio feed L3 e soma incremental exata em `i64`. Somas horizontais usam uma árvore de redução fixa, de modo que as features são bit a bit idênticas com e sem `target-cpu=native`.

| Slot | Feature | Sinal |
|---|---|---|
| 0 | OFI nível 1 decaído / profundidade média L1 | sim |
| 1 | OFI multinível ponderado / profundidade ponderada | sim |
| 2 | imbalance de profundidade `I` | sim |
| 3 | `micro − mid` (ticks) | sim |
| 4 | spread (ticks) | não |
| 5 | VPIN | não |
| 6 | volatilidade realizada do mid (ticks/√s) | não |
| 7 | fluxo agressor sinalizado / volume total | sim |

## EKF

Estado `x = [m, μ, h]` (fair value, drift, `h = ln σ²`), amostrado a cada 100 ms de tempo de bolsa:

$$m' = m + \mu\Delta t,\quad \mu' = e^{-\Delta t/\tau_\mu}\mu,\quad h' = \bar h + e^{-\Delta t/\tau_h}(h - \bar h),\quad Q = \mathrm{diag}(e^h\Delta t,\ q_\mu\Delta t,\ q_h\Delta t)$$

$$z = \begin{bmatrix} \text{micro} \\ r^2 \end{bmatrix},\quad h(x) = \begin{bmatrix} m \\ e^h\Delta t + \mu^2\Delta t^2 + 2R \end{bmatrix},\quad H = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2\mu\Delta t^2 & e^h\Delta t \end{bmatrix}$$

Covariância na forma de Joseph, re-simetrizada a cada passo; inovação de `r²` truncada em `[−ŷ, 8ŷ]`. O intervalo de 100 ms vem da curva de assinatura de volatilidade: em 10 ms o termo `2R` domina `σ²Δt` e o filtro subestima σ em até 10×; em 100 ms o σ estimado (4.45 ticks/√s) coincide com o σ realizado no horizonte de seleção adversa `h` e é insensível a `R` (4.26–4.49 para `R` entre 0.005 e 0.1). Tabelas em [`docs/design-decisions.md`](docs/design-decisions.md#3-intervalo-de-amostragem-do-ekf-pela-curva-de-assinatura).

## Modelo de toxicidade

Rótulo: para cada agressão (primeiro fill de cada burst), `y = 1` se o mid 100 ms depois está além do preço do touch executado, i.e. markout de 100 ms negativo para o lado passivo. Features do estado anterior à agressão, orientadas ao agressor (features com sinal multiplicadas pelo sinal do agressor). Na inferência a mesma floresta é avaliada nas duas orientações: compradora → risco da ask, vendedora → risco da bid.

Floresta oblivious, 48 árvores, profundidade 5, perda logística, folhas por passo de Newton, histograma de 32 quantis por feature. Avaliação: árvores agrupadas 8 por bloco; por (bloco, nível) uma busca das 8 features (um `vpermq` com AVX-512, 8 loads de L1 sem AVX-512) e um compare geram 8 bits de decisão. Nenhum branch dependente de dado.

| Conjunto | Origem | Amostras | Taxa base | AUC | Log-loss |
|---|---|---|---|---|---|
| treino | seed `0xA11CE`, 4 M eventos, 190 s | 285 130 | 0.3652 | 0.6975 | 0.5996 |
| holdout | seed `0xB0B`, 1 M eventos, 48 s | 70 971 | 0.3538 | 0.6925 | 0.5998 (taxa base: 0.6500) |

Regeneração de `crates/ml-alpha/src/toxicity_model.rs`:

```
cargo run --release -p engine-runtime --features trainer --bin train-toxicity
```

O arquivo gerado é idêntico byte a byte com e sem `target-cpu=native` em x86_64-pc-windows-msvc. Em outras plataformas a libm (`exp`, `ln`, `sin`, `cos`) altera os últimos ULPs dos limiares; as métricas não mudam na quarta casa.

## Estratégia

Avellaneda-Stoikov (2008). Utilidade CARA sobre a riqueza terminal, intensidade de execução `λ(δ) = A e^{−kδ}`:

$$\max_{\delta^a,\,\delta^b}\ \mathbb{E}\!\left[-e^{-\gamma\,(X_T + q_T S_T)}\right],\qquad dS_t = \sigma\, dW_t$$

Preço de reserva e spread ótimo (preços em ticks, `q` em lotes):

$$r(s, q, t) = s - q\,\gamma\,\sigma^2\,(T - t)$$

$$\delta^a + \delta^b = \gamma\,\sigma^2\,(T - t) + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right)$$

Overlay de seleção adversa com toxicidades `p^b, p^a` do GBDT, taxa base `p̄` (já embutida na calibração de `k`) e `ν = σ√h`, `h = 100 ms`:

$$r' = r + \mu\,T_\mu + \zeta\,(p^a - p^b)\,\nu$$

$$\delta^b = \max\!\Big(\tfrac{1}{2}(\delta^a + \delta^b) + \eta\,(p^b - \bar p)^+\,\nu,\ \delta_{\min}\Big),\qquad \delta^a = \max\!\Big(\tfrac{1}{2}(\delta^a + \delta^b) + \eta\,(p^a - \bar p)^+\,\nu,\ \delta_{\min}\Big)$$

$$P^b = \min\!\big(\lfloor r' - \delta^b \rfloor,\ P^a_1 - 1\big),\qquad P^a = \max\!\big(\lceil r' + \delta^a \rceil,\ P^b_1 + 1\big)$$

Um lado é retirado quando `p ≥ 0.9` ou quando `|q|` atinge 20 lotes no sentido de acumular. `s` = micro-price; `σ` e `μ` do EKF após 100 passos de aquecimento (antes disso, volatilidade realizada). σ é medido na escala de `h` (100 ms); o termo de inventário usa `T − t = 1 s`, onde a variância por unidade de tempo é ~1.4× maior, e essa diferença está absorvida na calibração de `γ`.

| Parâmetro | Valor |
|---|---|
| `γ` | 0.018 / tick |
| `k` | 3.0 / tick |
| `T − t` | 1 s (horizonte rolante) |
| `η`, `ζ` | 1.0, 0.5 |
| `p̄` | 0.3652 (`BASE_RATE` do modelo) |
| `T_μ` | 0.1 s |
| `δ_min` | 0.5 tick |
| `σ_min` | 0.5 tick/√s |

Fills de paper (para dar dinâmica ao inventário): estimativa de posição FIFO — ao entrar num nível, fila à frente = volume exibido; prints exibidos no preço da cotação consomem a fila antes de preencher; prints através do preço preenchem direto; prints ocultos só preenchem através. Em 5 M mensagens (238 s simulados): 5 670 fills, inventário p1/p50/p99 = −2/0/+2 lotes, nunca no limite, cotação sempre two-sided, spread cotado médio 2.21 ticks (mercado: 1.24).

## Verificação

| Crate | Testes | Conteúdo |
|---|---|---|
| `core-types` | 6 | aritmética de ponto fixo, tick, `Display` exato, layout `CachePadded` |
| `orderbook-l3` | 17 | bitmap × `BTreeSet` (proptest), índice × `HashMap` (proptest), arena, 9 cenários, 2 proptests de invariantes |
| `microstructure-simd` | 12 | OFI SIMD × definição escalar, VPIN × referência unidade-a-unidade (proptest), sinais de features |
| `ml-alpha` | 9 (+3 `trainer`) | recuperação de σ e drift pelo EKF, salto de preço, floresta SIMD × escalar no modelo publicado (proptest, inclusive NaN/inf), AUC do trainer |
| `strategy-mm` | 8 | forma fechada A-S, limites de inventário, baseline de toxicidade, proptest: cotação não cruzada, post-only, monotonicidade em `q` e `p` |
| `engine-runtime` | 15 + 4 | layouts ITCH 5.0 byte a byte, decode sem pânico (proptest), SPSC 2 M itens entre threads, histograma, replay, pipeline, zero alocação |

Proptest principal (`orderbook-l3/tests/proptest_invariants.rs`, 512 casos × até 250 operações aleatórias de submit/cancel/reduce/execute/replace): após cada operação verifica `audit()` (links, agregados, bitmap, índice), `best_bid < best_ask`, sequência de fills idêntica a um livro de referência `BTreeMap<_, VecDeque<_>>`, profundidade idêntica nível a nível e conservação de volume:

$$\sum Q_{\text{submetido}} = 2\sum Q_{\text{match}} + \sum Q_{\text{exec}} + \sum Q_{\text{cancel}} + \sum Q_{\text{IOC}} + Q_{\text{resting}}$$

`replay_consistency.rs`: 300 000 eventos do simulador codificados em ITCH, decodificados e reaplicados; o livro reconstruído é idêntico ao livro da bolsa simulada (ordens, volume e todos os níveis), sem rejeições, com toda cotação abaixo/acima do touch oposto.

```
cargo test --workspace --all-features                                  # 74 testes
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

A suíte passa nos dois caminhos de código SIMD (build base e `target-cpu=native` com AVX-512). CI (`.github/workflows/ci.yml`): fmt, clippy, testes e build dos benchmarks em Linux e Windows com Rust 1.96.0; `cargo check` com Rust 1.89 (MSRV).

## Benchmarks

Ambiente: AMD Ryzen 9 9950X3D (Zen 5, 16C/32T, AVX-512), 31 GB, Windows 11 Pro 10.0.26300, rustc 1.96.0, perfil `bench` = release + LTO fat + `codegen-units = 1`. Thread do benchmark fixada no core lógico 4. Sem isolamento de núcleos (Windows).

Duas configurações de build: **nativo** (`RUSTFLAGS="-C target-cpu=native"`, AVX-512) e **base** (sem `RUSTFLAGS`, x86-64 SSE2, o que `cargo bench` produz por padrão).

`cargo bench -p engine-runtime --bench latency` (Criterion 0.8, 100 amostras; estimativa de inclinação e IC 95 %, ns/op):

| Benchmark | Nativo | IC 95 % | Base | IC 95 % | Escopo |
|---|---:|---|---:|---|---|
| `e2e/itch_bytes_to_quote` | **162.83** | 161.70 – 164.10 | **314.63** | 312.34 – 317.08 | bytes ITCH → `Quote`, thread única, média sobre 1 M mensagens do simulador (decode, livro, snapshot, features, EKF, GBDT, A-S, fill model) |
| `itch/decode_add` | 4.86 | 4.82 – 4.89 | 4.86 | 4.83 – 4.90 | frame `A` de 36 B |
| `itch/decode_stream` | 9.33 | 9.19 – 9.50 | 9.04 | 8.98 – 9.12 | mix real de mensagens |
| `orderbook/add_cancel_at_touch` | 17.35 | 16.60 – 18.28 | 16.27 | 16.07 – 16.51 | `submit` GTC + `cancel` no melhor bid de um livro populado |
| `orderbook/sweep_5_levels` | 111.68 | 109.85 – 113.70 | 107.89 | 106.33 – 109.59 | IOC consumindo 20 ordens em 5 níveis |
| `orderbook/snapshot_8_levels` | 28.52 | 28.39 – 28.70 | 27.01 | 26.79 – 27.26 | 8 níveis por lado |
| `models/features_update_simd` | 24.45 | 23.99 – 25.01 | 36.82 | 36.46 – 37.26 | OFI, micro-price, VPIN, vol, fluxo |
| `models/gbdt_toxicity_both_sides` | 50.55 | 50.14 – 51.03 | 183.95 | 182.00 – 186.25 | 48 árvores × 2 orientações + 2 sigmoides |
| `models/ekf_step` | 45.82 | 45.58 – 46.10 | 107.12 | 105.53 – 109.06 | predição + atualização completas |
| `models/avellaneda_stoikov_quote` | 8.84 | 8.73 – 8.97 | 11.86 | 11.67 – 12.07 | |
| `spsc/push_pop_same_thread` | 6.29 | 6.26 – 6.32 | 6.46 | 6.44 – 6.50 | `MarketEvent` de 64 B |

O EKF executa um passo a cada 100 ms de tempo de bolsa; nas demais mensagens retorna após uma comparação. Livro e decode são código escalar e as diferenças entre colunas nesses benchmarks estão dentro da variação entre execuções; a diferença real está no GBDT (permute em registrador × loads), nas features (vetores de 512 × 128 bits) e no EKF (FMA em hardware × `fma()` emulado).

Pipeline de 3 threads, build nativo (`mm-engine --events 5000000 --rate 1000000`, T1/T2/T3 nos cores lógicos 2/4/6, latências via `RDTSC` calibrado em 4.292 GHz):

| Modo | Throughput | Métrica (ns) | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| sem pacing | 5.96 M msg/s | `on_event` | 151 | 171 | 282 | 1 240 | 923 015 |
| pacing 1 M msg/s | 1.00 M msg/s | tick → quote | 225 | 286 | 506 | 3 600 | 1 071 843 |
| | | `on_event` | 171 | 231 | 402 | 510 | 38 413 |
| | | salto SPSC | 44 | 65 | 85 | 3 312 | 1 072 023 |
| pacing 2 M msg/s | 2.00 M msg/s | tick → quote | 215 | 274 | 494 | 468 992 | 2 104 104 |

tick → quote = `RDTSC` na T1 antes do decode até a cotação pronta na T2. No modo sem pacing essa métrica é tempo de fila (ring cheio: 2^16 × ~160 ns ≈ 10 ms) e não é reportada. Caudas de p99.9 em diante são preempção do scheduler do Windows: a 2 M msg/s, uma pausa do tamanho do máximo observado (~2 ms) acumula ~4 000 mensagens, que drenam a ~4 M msg/s líquidos; uma ou duas pausas assim bastam para que mais de 0.1 % das mensagens passe centenas de microssegundos na fila. Em 5 M amostras, 18 saltos SPSC tiveram `RDTSC` do consumidor menor que o do produtor (offset de TSC entre núcleos maior que o salto); foram registrados como 0 e contados em `tsc_skew_clamped`.

Histórico de otimização (mesmo benchmark ponta-a-ponta):

| Mudança | Build | GBDT (ns) | e2e (ns) |
|---|---|---:|---:|
| avaliador escalar árvore a árvore | nativo | 175 | 289 |
| layout SoA em blocos de 8, `vpermq` + compare vetorial | nativo | 54.7 | 162 |
| orientação/saneamento de features em `f64x8` | nativo | 50.1 | 156 |
| sigmoide via `f64x2::exp` de `wide` (revertido: +4.6 %) | nativo | 57.4 | 170 |
| sem AVX-512: permute emulado pela `wide` | base | 10 354 | 10 654 |
| sem AVX-512: gather por 8 loads de L1 | base | 184 | 315 |

## Execução

```
cargo run --release -p engine-runtime --bin mm-engine -- \
    --events 5000000 --mode both --rate 1000000 --cores 2,4,6 [--quotes out/quotes.csv]
```

`--mode throughput|latency|both`, `--no-pin` desativa pinning, `--seed` muda o mercado simulado. O CSV de cotações é escrito pela T3 no run com pacing.

Build nativo (números da coluna "Nativo"), em target-dir separado para não invalidar o cache do build base:

```
# bash
RUSTFLAGS="-C target-cpu=native" cargo bench --target-dir target/native -p engine-runtime --bench latency
# PowerShell
$env:RUSTFLAGS = "-C target-cpu=native"; cargo bench --target-dir target/native -p engine-runtime --bench latency
```

Binários nativos exigem a mesma ISA da máquina de build.

## Limitações

- Instância única por instrumento; ladder densa sem re-centralização (±327 unidades de preço com tick 0.01 na configuração padrão).
- Modelo de toxicidade, curva de volatilidade e calibração de A-S feitos sobre o simulador; em produção, retreinar sobre ITCH real e recalibrar `k`, `γ`, `p̄` e o intervalo do EKF.
- Fill model de paper não reflete impacto da própria ordem nem cancelamentos à frente além do teto pelo volume exibido.
- Sem gateway de ordens, risk checks pré-trade ou reconciliação; a saída é a cotação.
- `CachePadded` usa 64 B. Em CPUs Intel com prefetch de linha adjacente, 128 B evita false sharing residual.
- Latências de cauda medidas em Windows sem `isolcpus`/`nohz_full`; o RDTSC entre núcleos assume TSC invariante sincronizado.

## Licença

Licenciado sob [MIT](LICENSE-MIT), à escolha.
