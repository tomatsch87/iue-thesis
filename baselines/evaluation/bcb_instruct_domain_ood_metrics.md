# OOD generalization metrics

Epsilon: 0.02
Domains: 7

## Domain: computation
| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |
|---|---:|---:|---:|---:|
| Qwen3-Coder | 0.684119 | 0.656977 | -0.027142 | 0.027142 |
| GPT-OSS | 0.656871 | 0.621281 | -0.035590 | 0.035590 |
| Nvidia Nemotron | 0.641119 | 0.646553 | 0.005434 | 0.000000 |
| Olmo Instruct | 0.730893 | 0.702319 | -0.028574 | 0.028574 |

## Domain: cryptography
| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |
|---|---:|---:|---:|---:|
| Qwen3-Coder | 0.684119 | 0.635290 | -0.048829 | 0.048829 |
| GPT-OSS | 0.656871 | 0.628117 | -0.028754 | 0.028754 |
| Nvidia Nemotron | 0.641119 | 0.674607 | 0.033488 | 0.000000 |
| Olmo Instruct | 0.730893 | 0.714686 | -0.016207 | 0.016207 |

## Domain: general
| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |
|---|---:|---:|---:|---:|
| Qwen3-Coder | 0.684119 | 0.674291 | -0.009828 | 0.009828 |
| GPT-OSS | 0.656871 | 0.640230 | -0.016641 | 0.016641 |
| Nvidia Nemotron | 0.641119 | 0.652831 | 0.011712 | 0.000000 |
| Olmo Instruct | 0.730893 | 0.708952 | -0.021941 | 0.021941 |

## Domain: network
| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |
|---|---:|---:|---:|---:|
| Qwen3-Coder | 0.684119 | 0.729427 | 0.045308 | 0.000000 |
| GPT-OSS | 0.656871 | 0.670130 | 0.013259 | 0.000000 |
| Nvidia Nemotron | 0.641119 | 0.725914 | 0.084795 | 0.000000 |
| Olmo Instruct | 0.730893 | 0.727468 | -0.003425 | 0.003425 |

## Domain: system
| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |
|---|---:|---:|---:|---:|
| Qwen3-Coder | 0.684119 | 0.664331 | -0.019788 | 0.019788 |
| GPT-OSS | 0.656871 | 0.650238 | -0.006633 | 0.006633 |
| Nvidia Nemotron | 0.641119 | 0.656793 | 0.015674 | 0.000000 |
| Olmo Instruct | 0.730893 | 0.710680 | -0.020213 | 0.020213 |

## Domain: time
| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |
|---|---:|---:|---:|---:|
| Qwen3-Coder | 0.684119 | 0.640209 | -0.043910 | 0.043910 |
| GPT-OSS | 0.656871 | 0.610492 | -0.046379 | 0.046379 |
| Nvidia Nemotron | 0.641119 | 0.650781 | 0.009662 | 0.000000 |
| Olmo Instruct | 0.730893 | 0.756970 | 0.026077 | 0.000000 |

## Domain: visualization
| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |
|---|---:|---:|---:|---:|
| Qwen3-Coder | 0.684119 | 0.681409 | -0.002710 | 0.002710 |
| GPT-OSS | 0.656871 | 0.637895 | -0.018976 | 0.018976 |
| Nvidia Nemotron | 0.641119 | 0.632766 | -0.008353 | 0.008353 |
| Olmo Instruct | 0.730893 | 0.696717 | -0.034176 | 0.034176 |

## Summary (per model)
| Model | Mean shortfall | Std shortfall | 95% CI mean shortfall | Fraction shortfall < epsilon | Mean signed drop | Std signed drop |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Coder | 0.021744 | 0.019300 | [0.008883, 0.035281] | 0.571 | -0.015271 | 0.031530 |
| GPT-OSS | 0.021853 | 0.016240 | [0.010790, 0.033036] | 0.571 | -0.019959 | 0.019633 |
| Nvidia Nemotron | 0.001193 | 0.003157 | [0.000000, 0.003580] | 1.000 | 0.021773 | 0.030465 |
| Olmo Instruct | 0.017791 | 0.012472 | [0.009069, 0.026119] | 0.429 | -0.014066 | 0.020182 |

## Summary (all models pooled)
| Mean shortfall | Std shortfall | 95% CI mean shortfall | Fraction shortfall < epsilon | Mean signed drop | Std signed drop |
|---:|---:|---:|---:|---:|---:|
| 0.015645 | 0.015911 | [0.010070, 0.021630] | 0.643 | -0.006881 | 0.029869 |
