## LCB/BCB-Instruct Cross-Benchmark Evaluation on test split (using XGBoost baselines)

### Qwen3-Coder

Trained on LiveCodeBench, tested on BigCodeBench-Instruct.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last code token       | **0.648309** | **-0.400704** | **0.660910** | **0.552193** | 1    |
| first+last code token | 0.628632     | -0.224934     | 0.640532     | 0.549561     | 2    |
| last token            | 0.620964     | -0.537230     | 0.630690     | 0.462719     | 3    |
| first+last token      | 0.573961     | -0.189899     | 0.631420     | 0.464912     | 4    |
| first token           | 0.527632     | -0.125264     | 0.013133     | 0.538596     | 5    |
| first code token      | 0.500475     | -0.041575     | 0.483197     | 0.500877     | 6    |

Trained on BigCodeBench-Instruct, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss          | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------ | ------------ | ------------ | ---- |
| last code token       | **0.823983** | **0.195959** | **0.800183** | **0.768783** | 1    |
| first+last code token | 0.726469     | 0.078729     | 0.672764     | 0.659259     | 2    |
| first+last token      | 0.660408     | 0.009648     | 0.627016     | 0.608466     | 3    |
| last token            | 0.622642     | -0.010945    | 0.564978     | 0.582011     | 4    |
| first token           | 0.543572     | -0.069269    | 0.735008     | 0.581481     | 5    |
| first code token      | 0.514168     | -0.023301    | 0.648287     | 0.554497     | 6    |

### GPT-OSS

Trained on LiveCodeBench, tested on BigCodeBench-Instruct.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last token            | **0.596994** | -0.759983     | 0.633072     | 0.463845     | 1    |
| first+last token      | 0.579973     | -0.694104     | **0.633102** | 0.463404     | 2    |
| first+last code token | 0.579668     | -0.282876     | 0.608084     | 0.508377     | 3    |
| last code token       | 0.560857     | -0.880882     | 0.629426     | 0.464727     | 4    |
| first code token      | 0.559198     | **-0.199883** | 0.600590     | **0.522046** | 5    |
| first token           | 0.549722     | -0.459830     | 0.632911     | 0.462963     | 6    |

Trained on BigCodeBench-Instruct, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last code token       | **0.734240** | **-0.392674** | 0.733426     | 0.638365     | 1    |
| first+last code token | 0.712481     | -0.706272     | 0.612371     | 0.527044     | 2    |
| last token            | 0.689348     | -0.471268     | 0.707981     | 0.608805     | 3    |
| first+last token      | 0.675053     | -0.403271     | **0.803761** | **0.698113** | 4    |
| first code token      | 0.611264     | -0.841178     | 0.100741     | 0.236478     | 5    |
| first token           | 0.486233     | -0.700292     | 0.505094     | 0.419497     | 6    |

### Nvidia Nemotron

Trained on LiveCodeBench, tested on BigCodeBench-Instruct.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| first+last token      | **0.583756** | -0.739149     | 0.611758     | 0.441410     | 1    |
| last token            | 0.557510     | -1.188312     | 0.612646     | 0.444053     | 2    |
| first+last code token | 0.549105     | -0.858902     | 0.612496     | 0.445374     | 3    |
| last code token       | 0.539821     | -1.097255     | **0.613013** | **0.447137** | 4    |
| first code token      | 0.510379     | **-0.717742** | 0.612508     | 0.443172     | 5    |
| first token           | 0.500475     | -0.881698     | 0.611196     | 0.440088     | 6    |

Trained on BigCodeBench-Instruct, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last token            | **0.777089** | -1.421912     | 0.503563     | 0.421053     | 1    |
| last code token       | 0.692757     | -2.527210     | 0.098160     | 0.185596     | 2    |
| first+last code token | 0.683663     | -1.986916     | 0.164706     | 0.213296     | 3    |
| first code token      | 0.628982     | -2.060167     | 0.188825     | 0.222530     | 4    |
| first+last token      | 0.519809     | -1.273296     | **0.626785** | **0.493075** | 5    |
| first token           | 0.447814     | **-1.156974** | 0.607923     | 0.469991     | 6    |

### Olmo Instruct

Trained on LiveCodeBench, tested on BigCodeBench-Instruct.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| first+last code token | **0.614892** | -0.390349     | 0.512090     | 0.492819     | 1    |
| last code token       | 0.596904     | -0.551340     | **0.514555** | 0.446140     | 2    |
| first code token      | 0.581355     | -0.260917     | 0.486117     | 0.493268     | 3    |
| first+last token      | 0.566836     | -0.063593     | 0.360028     | 0.594704     | 4    |
| last token            | 0.558959     | -0.176050     | 0.469639     | 0.498205     | 5    |
| first token           | 0.486353     | **-0.047423** | 0.010554     | **0.663375** | 6    |

Trained on BigCodeBench-Instruct, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last code token       | **0.740994** | -0.646883     | 0.021505     | 0.431250     | 1    |
| first+last token      | 0.725338     | -0.289516     | 0.034115     | 0.433750     | 2    |
| last token            | 0.722801     | -0.473877     | 0.088751     | 0.448125     | 3    |
| first+last code token | 0.694644     | -0.874288     | 0.004338     | 0.426250     | 4    |
| first token           | 0.575537     | **-0.172302** | **0.167969** | **0.467500** | 5    |
| first code token      | 0.477583     | -0.467960     | 0.002167     | 0.424375     | 6    |
