## LCB/BCB-Fusion Cross-Benchmark Evaluation on test split (using XGBoost baselines)

### Qwen3-Coder

Trained on LiveCodeBench, tested on BigCodeBench-Fusion.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last code token       | **0.650942** | -0.372366     | **0.758800** | **0.631718** | 1    |
| last token            | 0.636227     | -0.381507     | 0.752888     | 0.613656     | 2    |
| first+last code token | 0.635395     | -0.094262     | 0.741868     | 0.618943     | 3    |
| first+last token      | 0.583917     | -0.076976     | 0.755781     | 0.618502     | 4    |
| first token           | 0.547527     | -0.080018     | 0.678466     | 0.567841     | 5    |
| first code token      | 0.502341     | **-0.048092** | 0.571647     | 0.506167     | 6    |

Trained on BigCodeBench-Fusion, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss          | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------ | ------------ | ------------ | ---- |
| last code token       | **0.816433** | **0.290591** | **0.806708** | **0.768254** | 1    |
| last token            | 0.803613     | 0.204567     | 0.779531     | 0.696825     | 2    |
| first+last code token | 0.733510     | 0.121068     | 0.755504     | 0.665079     | 3    |
| first token           | 0.621722     | -0.163637    | 0.736984     | 0.585714     | 4    |
| first+last token      | 0.615652     | -0.000541    | 0.720301     | 0.567725     | 5    |
| first code token      | 0.488321     | -0.117053    | 0.724283     | 0.583069     | 6    |

### GPT-OSS

Trained on LiveCodeBench, tested on BigCodeBench-Fusion.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last token            | **0.687451** | -0.585559     | **0.735286** | 0.581754     | 1    |
| first token           | 0.673806     | -0.261678     | 0.735229     | 0.581313     | 2    |
| first+last token      | 0.664570     | -0.489989     | **0.735286** | 0.581754     | 3    |
| last code token       | 0.616959     | -0.643950     | 0.735048     | 0.582195     | 4    |
| first+last code token | 0.611437     | -0.234349     | 0.731920     | **0.593213** | 5    |
| first code token      | 0.558098     | **-0.135669** | 0.716381     | 0.591009     | 6    |

Trained on BigCodeBench-Fusion, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss          | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------ | ------------ | ------------ | ---- |
| first+last token      | **0.741523** | **0.077238** | **0.890416** | **0.809434** | 1    |
| last code token       | 0.715822     | -0.235853    | 0.820345     | 0.724528     | 2    |
| first token           | 0.708531     | -0.078997    | 0.862834     | 0.770440     | 3    |
| first+last code token | 0.700157     | -0.255217    | 0.873924     | 0.788050     | 4    |
| last token            | 0.686018     | -0.352136    | 0.799666     | 0.698113     | 5    |
| first code token      | 0.671602     | -0.223197    | 0.888014     | 0.803145     | 6    |

### Nvidia Nemotron

Trained on LiveCodeBench, tested on BigCodeBench-Fusion.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last token            | **0.623615** | -0.768669     | 0.711304     | 0.556545     | 1    |
| first+last code token | 0.605049     | -0.511082     | 0.711756     | 0.557881     | 2    |
| last code token       | 0.583649     | -0.688449     | **0.713246** | **0.561443** | 3    |
| first code token      | 0.560707     | **-0.378046** | 0.708839     | 0.551202     | 4    |
| first token           | 0.559447     | -0.570959     | 0.706966     | 0.546750     | 5    |
| first+last token      | 0.522629     | -0.481267     | 0.709659     | 0.552983     | 6    |

Trained on BigCodeBench-Fusion, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last token            | **0.842646** | **-0.703338** | **0.826829** | **0.737765** | 1    |
| last code token       | 0.831427     | -1.047041     | 0.673669     | 0.569714     | 2    |
| first+last code token | 0.797988     | -1.386882     | 0.593867     | 0.498615     | 3    |
| first+last token      | 0.737331     | -0.823118     | 0.749035     | 0.639889     | 4    |
| first code token      | 0.669188     | -1.666137     | 0.356832     | 0.317636     | 5    |
| first token           | 0.457516     | -0.873890     | 0.745783     | 0.610342     | 6    |

### Olmo Instruct

Trained on LiveCodeBench, tested on BigCodeBench-Fusion.

| Baseline Model        | roc_auc      | bss           | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------- | ------------ | ------------ | ---- |
| last token            | **0.671347** | -0.019660     | 0.661115     | 0.578568     | 1    |
| first+last code token | 0.669868     | -0.175409     | 0.657837     | **0.581270** | 2    |
| last code token       | 0.669819     | -0.125430     | **0.669951** | 0.577668     | 3    |
| first+last token      | 0.668785     | -0.144283     | 0.665713     | 0.579018     | 4    |
| first code token      | 0.636001     | **-0.016082** | 0.634944     | 0.557857     | 5    |
| first token           | 0.602878     | -0.120716     | 0.593390     | 0.523638     | 6    |

Trained on BigCodeBench-Fusion, tested on LiveCodeBench.

| Baseline Model        | roc_auc      | bss          | f1           | accuracy     | rank |
| --------------------- | ------------ | ------------ | ------------ | ------------ | ---- |
| last token            | **0.814912** | 0.090798     | 0.614296     | 0.659375     | 1    |
| first+last token      | 0.809496     | **0.243280** | **0.754366** | **0.736250** | 2    |
| last code token       | 0.790935     | 0.123288     | 0.643709     | 0.663750     | 3    |
| first+last code token | 0.769425     | -0.168245    | 0.288288     | 0.506250     | 4    |
| first token           | 0.694377     | 0.038604     | 0.607350     | 0.619375     | 5    |
| first code token      | 0.592164     | -0.303791    | 0.101291     | 0.434375     | 6    |
