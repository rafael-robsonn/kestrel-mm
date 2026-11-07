# Decisões de design

Registro das decisões não óbvias do motor, com o problema que cada uma resolve, as alternativas avaliadas e a medição que sustenta a escolha. Hardware das medições: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), Windows 11, rustc 1.96.0, build release.

1. [Bitmap hierárquico de ocupação dos níveis de preço](#1-bitmap-hierárquico-de-ocupação-dos-níveis-de-preço)
2. [Floresta oblivious em layout SoA por blocos de 8 árvores](#2-floresta-oblivious-em-layout-soa-por-blocos-de-8-árvores)
3. [Intervalo de amostragem do EKF pela curva de assinatura](#3-intervalo-de-amostragem-do-ekf-pela-curva-de-assinatura)
4. [Reprodutibilidade numérica entre builds](#4-reprodutibilidade-numérica-entre-builds)

---

## 1. Bitmap hierárquico de ocupação dos níveis de preço

**Código:** `crates/orderbook-l3/src/bitmap.rs`, `crates/orderbook-l3/src/book.rs`

### Problema

O livro usa uma ladder densa: o nível de um preço é `idx = (p − p_min) / tick`, acesso O(1) sem hash. Isso resolve "onde fica o nível do preço P", mas não "qual é o próximo nível ocupado". Essa segunda pergunta aparece em dois pontos do hot path:

- quando o melhor nível esvazia (cancel, execução ou sweep no topo), é preciso achar o novo melhor preço;
- o snapshot de profundidade caminha de nível ocupado em nível ocupado (8 por lado, a cada evento).

Perto do touch os níveis costumam ser contíguos. Longe dele, e logo após sweeps, o livro tem buracos de centenas ou milhares de ticks. É justamente aí que o tempo de reação importa.

### Decisão

Três níveis de bitmaps de 64 bits sobre os índices da ladder:

```
l2: 1 palavra            bit k  = l1[k] != 0
l1: 64 palavras (máx.)   bit w  = l0[w] != 0
l0: 4 096 palavras (máx.) bit i = nível i ocupado
```

`min`, `max`, `next_above` e `next_below` fazem no máximo três buscas de bit (`lzcnt`/`tzcnt`) com máscara, independente da distância até o próximo nível. Capacidade: 64³ = 262 144 níveis. Com a configuração padrão (65 536 níveis) o bitmap ocupa 1 024 + 16 + 1 palavras, cerca de 8.3 KiB.

### Medição

Consulta "próximo nível ocupado abaixo do topo", com o único nível ocupado a `gap` ticks de distância. Ladder de 65 536 níveis, cache quente, duas execuções (ns por consulta):

| gap (ticks) | bitmap hierárquico | bitmap de 1 nível | varredura da ladder (24 B/nível) |
|---:|---:|---:|---:|
| 1 | 0.6 | 0.6 | 0.4 |
| 8 | 0.6 | 0.6 | 1.2 |
| 64 | 1.2 | 0.8 – 0.9 | 9.6 |
| 512 | 1.2 | 2.5 | 78 |
| 4 096 | 1.3 – 1.4 | 16 – 17 | 627 – 637 |
| 32 768 | 1.3 – 1.4 | 157 – 158 | 5 370 – 7 305 |

No caso comum (gap ≤ 8) as três abordagens empatam. A escolha não se apoia no caso médio, e sim no **pior caso limitado**: a varredura linear cresce cerca de 0.2 ns por nível e chega a microssegundos exatamente nos eventos de sweep, e o bitmap de um nível cresce com `gap / 64`. Com cache frio, que é o cenário real depois de um sweep, a diferença aumenta.

### Alternativas descartadas

| Alternativa | Por que não |
|---|---|
| `BTreeMap<Price, Level>` | O(log n) por acesso, aloca a cada nível novo (proibido no hot path), navegação por ponteiros. |
| Vetor ordenado de níveis (melhor preço no fim) | Comum em produção quando a profundidade é pequena. Localizar um preço é busca binária O(log n) e inserir no meio é `memmove` O(n). No ITCH, Adds e Deletes chegam em qualquer profundidade. |
| `HashMap` preço → nível + heap de melhores preços | Hash por mensagem e manutenção de heap O(log n); `HashMap` é proibido pela especificação. |
| Ladder sem índice auxiliar | É a coluna "varredura" da tabela. |

### Custos aceitos

- A faixa de preço é fixa e pré-alocada: 65 536 níveis × 24 B = 1.5 MiB por lado, ±327 unidades de preço com tick de 0.01. Não há re-centralização da ladder.
- `set` atualiza os três níveis (3 OR). `clear` só sobe de nível quando uma palavra zera.

### Verificação

Proptest contra `BTreeSet` (`min`, `max`, `next_above`, `next_below`, `contains` sob inserções e remoções aleatórias). O proptest do livro inteiro audita a consistência entre bitmap e níveis após cada operação.

---

## 2. Floresta oblivious em layout SoA por blocos de 8 árvores

**Código:** `crates/ml-alpha/src/gbdt.rs`, gerador em `crates/ml-alpha/src/train.rs`

### Problema

O modelo de toxicidade roda a cada evento, duas vezes: uma com as features orientadas ao agressor comprador (risco da ask) e outra ao vendedor (risco da bid). Na primeira versão, com avaliação escalar árvore a árvore, o GBDT custava **175 ns de um hot path de 289 ns**.

### Decisão, parte 1: árvores oblivious

Numa árvore oblivious, todos os nós de um mesmo nível usam o mesmo par (feature, limiar). O índice da folha é o vetor de bits das D comparações:

```
leaf(x) = Σ_d [ x[f_d] > θ_d ] · 2^d
```

As D comparações são independentes entre si: o nível d não depende do resultado do nível d − 1. Não há ponteiros nem branch dependente de dado. Numa árvore comum, mesmo em array e sem branch (`idx = 2·idx + 1 + [x > θ]`), cada nível depende do load do nível anterior, uma cadeia de D loads por árvore. Com ponteiros, features ruidosas produzem misprediction perto de 50%.

Custo: uma árvore oblivious é menos expressiva, porque o nível inteiro compartilha o split. Compensa-se com mais árvores. Não houve comparação contra árvores comuns com o mesmo orçamento. No simulador, as variações testadas (48 × 5, 64 × 4, 48 × 6) deram AUC de holdout entre 0.691 e 0.693, o que indica que a capacidade do modelo não é o gargalo.

### Decisão, parte 2: layout SoA em blocos de 8 árvores

```
AoS (uma árvore por vez)              SoA (bloco k = árvores 8k … 8k+7, nível d)
árvore t:                             features[k][d]   = u64x8 [ f(8k,d) … f(8k+7,d) ]
  features[t]   = [f0 f1 f2 f3 f4]    thresholds[k][d] = f64x8 [ θ(8k,d) … θ(8k+7,d) ]
  thresholds[t] = [θ0 θ1 θ2 θ3 θ4]    leaves[k][lane]  = [w0 … w31]
  leaves[t]     = [w0 … w31]
```

O vetor de features tem exatamente 8 `f64`, um registrador de 512 bits. Buscar "a feature que cada uma das 8 árvores testa neste nível" vira um único permute de lanes (`vpermq`), em vez de oito loads. Uma comparação vetorial produz os oito bits de decisão do nível. Para 48 árvores de profundidade 5 são 6 blocos × 5 níveis = 30 passos de (permute, compare, and, or), mais 48 loads de folha. As duas orientações compartilham os loads de parâmetros (`margin_pair`).

### Medição (build nativo, ponta a ponta = bytes ITCH → cotação)

| Versão | GBDT (ns) | ponta a ponta (ns) |
|---|---:|---:|
| escalar, AoS | 175 | 289 |
| SoA em blocos + `vpermq` | 54.7 | 162 |
| + orientação e saneamento das features em `f64x8` | 50.1 | 156 |
| + sigmoide via `f64x2::exp` da `wide` (revertida, +4.6%) | 57.4 | 170 |

A última linha ficou registrada porque a hipótese era razoável (um `exp` vetorial em vez de dois escalares) e a medição a refutou: o `exp` polinomial da `wide` é mais caro que duas chamadas da libm.

### Portabilidade

O permute com índices em runtime só é uma instrução com AVX-512. Sem AVX-512, a `wide` emula com um fallback escalar em camadas, de cerca de 170 ns por permute. No build x86-64 base isso deixou o GBDT em **10.35 µs** e o ponta a ponta em 10.65 µs. O avaliador agora escolhe em tempo de compilação (`cfg(target_feature = "avx512f")`):

- com AVX-512: `vpermq`;
- sem AVX-512: oito loads escalares do array de features, que está em L1.

Resultado no build base: GBDT em 184 ns e ponta a ponta em 315 ns. No build nativo: 50.6 ns e 163 ns.

### Modelo como dados estáticos

O trainer gera `toxicity_model.rs` com `static` arrays já no layout SoA. Não há parsing nem IO na inicialização, não há alocação, e a forma do modelo (`L = 2^D`) é verificada em tempo de compilação. Custo: trocar o modelo exige recompilar.

### Verificação

Proptest compara o caminho SIMD com a avaliação escalar árvore a árvore no modelo publicado, inclusive entradas NaN e infinitas, com tolerância de 1e-12. O teste roda nos dois caminhos (`vpermq` e loads), conforme as flags de compilação.

---

## 3. Intervalo de amostragem do EKF pela curva de assinatura

**Código:** `crates/ml-alpha/src/ekf.rs`, configuração em `crates/engine-runtime/src/engine.rs`

### Problema

O EKF estima a difusão σ do fair value a partir do micro-price amostrado a cada Δ. A equação de medida do retorno ao quadrado é:

```
E[r²] = σ²·Δ + μ²·Δ² + 2R        (R = variância do micro-price em torno do fair value)
```

σ alimenta dois termos da estratégia:

- o skew de inventário do Avellaneda-Stoikov, `q·γ·σ²·τ`, com τ = 1 s;
- a escala do overlay de toxicidade, `ν = σ·√h`, com h = 100 ms, o mesmo horizonte do rótulo do GBDT.

Sintoma observado com Δ = 10 ms: o EKF estimava σ = 0.35 ticks/√s, contra cerca de 4.5 realizado. O A-S caía no piso de σ, o skew de inventário ficava em 0.0125 tick por lote e o inventário passava 54% do tempo colado no limite de ±20 lotes.

### Diagnóstico: dois efeitos

1. **Identificabilidade.** Com Δ = 10 ms, σ²Δ fica entre 0.12 e 0.2 tick², a mesma ordem de 2R. O filtro não separa variância do preço de ruído de medida e atribui tudo ao ruído. Por isso o σ estimado despenca quando R aumenta (tabela abaixo).
2. **Curva de assinatura.** No mercado simulado, assim como em mercados reais com descoberta de preço gradual, os retornos de alta frequência são positivamente autocorrelacionados: o mid persegue o fair value em varreduras sucessivas. A variância realizada por unidade de tempo cresce com o horizonte até saturar.

### Medição

4 M eventos (190 s simulados, seed `0x5EED`). σ em ticks/√s.

Curva de assinatura (σ realizado com amostragem regular, último valor observado):

| Δ | 10 ms | 25 ms | 50 ms | 100 ms | 250 ms | 500 ms | 1 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| σ do mid | 3.42 | 3.52 | 3.98 | 4.69 | 5.43 | 5.86 | 5.57 |
| σ do micro-price | 2.51 | 2.91 | 3.56 | 4.43 | 5.37 | 5.89 | 5.58 |
| VR = σ²(Δ) / σ²(10 ms), mid | 1.00 | 1.06 | 1.36 | 1.89 | 2.52 | 2.94 | 2.66 |
| amostras | 18 996 | 7 598 | 3 799 | 1 899 | 759 | 379 | 189 |

O ponto de 1 s tem erro padrão de cerca de 5%. O fair value latente do simulador tem σ = 6 ticks/√s mais saltos (0.2/s de ±4 ticks), √(36 + 3.2) ≈ 6.3: é o limite de longo horizonte, e a curva se aproxima dele.

σ médio estimado pelo EKF (após 10 s de aquecimento), por intervalo de amostragem e por R:

| Δ \ R | 0.005 | 0.02 | 0.1 |
|---|---:|---:|---:|
| 10 ms | 2.25 | 1.43 | 0.23 |
| 25 ms | 2.82 | 2.60 | 0.87 |
| 50 ms | 3.54 | 3.45 | 2.94 |
| **100 ms** | **4.49** | **4.45** | **4.26** |
| 250 ms | 5.46 | 5.44 | 5.38 |

### Decisão

Δ = 100 ms, R = 0.02, h̄ = ln(4.5²) ≈ 3.0. Motivos:

- **Escala correta para o overlay.** `ν = σ·√h` com h = 100 ms precisa de σ medido em h. O EKF a 100 ms dá 4.45, contra 4.69 do mid e 4.43 do micro-price realizados a 100 ms.
- **Robustez a R**, o parâmetro menos conhecido do filtro. A 100 ms, σ varia entre 4.26 e 4.49 quando R varia 20× (de 0.005 a 0.1). A 10 ms, varia de 0.23 a 2.25.
- **Custo:** 10 passos por segundo. O filtro reage mais devagar a mudanças de regime de volatilidade, o que é irrelevante com τ_h = 60 s.

Histórico: a primeira correção usou Δ = 50 ms, escolhido olhando apenas o σ realizado a 95 ms. A curva completa mostrou que 50 ms ainda subestima a escala de h e ainda depende de R, e o intervalo passou para 100 ms.

### Consequência para o Avellaneda-Stoikov

O termo de inventário usa τ = 1 s, horizonte em que a curva dá σ ≈ 5.6, com VR(1 s) / VR(100 ms) ≈ 1.4. Em vez de manter dois σ, γ foi calibrado contra σ(100 ms):

- o A-S só usa o produto γσ²;
- `(1/γ)·ln(1 + γ/k) ≈ 1/k` é insensível a γ quando γ ≪ k.

Com γ = 0.018, o skew é γσ²τ ≈ 0.36 tick por lote, igual ao da calibração anterior (γ = 0.03 com σ = 3.45). O comportamento se manteve: em 5 M mensagens, inventário com p1/p99 de ±2 lotes, nunca no limite, e 5 670 fills.

### Limitações

- A curva foi medida no simulador. Com dados reais a medição precisa ser refeita, porque o formato depende do ativo e do horário.
- Alternativas mais completas: σ por horizonte via variance ratio ou realized kernels, ou um EKF com ruído de microestrutura autocorrelacionado (um estado a mais).

---

## 4. Reprodutibilidade numérica entre builds

**Código:** `crates/microstructure-simd/src/ladder.rs` (`hsum`), `crates/microstructure-simd/src/engine.rs`

### Problema

O modelo publicado é gerado a partir das features calculadas pelo próprio motor. Duas fontes faziam as features diferirem nos últimos ULPs entre builds com e sem `target-cpu=native`, o que mudava os limiares do modelo gerado:

- `f64::mul_add` e `f64x8::mul_add`: fundidos (um arredondamento) com FMA em hardware, multiplicação + soma (dois arredondamentos) sem ele. Sem FMA, o `mul_add` escalar ainda vira uma chamada de `fma()` emulada em software, cara no hot path.
- `reduce_add`: a soma horizontal associa as lanes em ordem diferente conforme o conjunto de instruções.

### Decisão

- As EWMAs das features usam `a * b + c` explícito. O arredondamento único do FMA não traz ganho mensurável nesses acumuladores.
- As somas horizontais usam uma árvore de redução fixa, `((a₀+a₄)+(a₂+a₆)) + ((a₁+a₅)+(a₃+a₇))`, montada com somas vetoriais por lane: `f64x8 → 2 × f64x4 → 2 × f64x2`. Entre as variantes de ordem fixa medidas lado a lado, foi cerca de 1.7 ns mais rápida no `update` de features que extrair as 8 lanes e somar escalarmente.
- O EKF mantém `mul_add`. Roda a 10 Hz, a covariância na forma de Joseph se beneficia do arredondamento único, e o EKF não participa do treino.

### Resultado

`train-toxicity` gera o mesmo `toxicity_model.rs`, byte a byte, com e sem `target-cpu=native`, verificado em x86_64-pc-windows-msvc. Em outra plataforma, a libm (`exp`, `ln`, `sin`, `cos` no simulador e no treino) muda os últimos ULPs e, com isso, o arquivo gerado. As métricas do modelo não mudam na quarta casa decimal.
