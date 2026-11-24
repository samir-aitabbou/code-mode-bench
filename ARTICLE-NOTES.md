# Notes pour l'article (Hashnode) — benchmark Code Mode vs MCP classique

Notes de travail en français pour rédiger l'article à partir de `REPORT.md` (en anglais,
destiné au dépôt). Chaque chiffre cité ici est sourcé (fichier CSV ou figure).

## Le message principal

Sur un petit GPU local (24 Go), le Code Mode n'est pas qu'une optimisation de tokens : **c'est
souvent la seule façon de finir la tâche**. Sur les tâches « lourdes en données » (T3 : agrégation
chaînée, T4 : boucle avec écritures, T5 : gros volume), le MCP classique échoue carrément à 0% de
réussite sur gpt-oss-20b (et sur T4 pour les deux modèles), par dépassement de contexte — pas
parce que le modèle « raisonne mal », mais parce que le contexte est la ressource qui manque en
premier sur ce matériel.

## Chiffres à citer (avec leur source)

- **280 essais réels**, 2 modèles, ~4,9h de temps GPU cumulé, ~7,87M tokens au total
  (`results/summary/trials.csv`, sommes directes).
- **H7 confirmée nettement** : les 20 échecs `context_overflow` observés sont **tous** en mode
  classique (10 par modèle), zéro en Code Mode (`results/summary/context_overflow_by_model.csv`,
  figure `results/figures/03_success_rate_by_task.png` et `06_failure_breakdown.png`).
- **T4 (boucle + écritures) : 0% de réussite classique pour les DEUX modèles**, 80-100% en Code
  Mode (`results/summary/by_cell.csv`). C'est la tâche la plus spectaculaire pour l'article :
  symétrique entre les deux modèles, donc pas un artefact d'un seul modèle.
- **gpt-oss-20b est « sauvé » par le Code Mode** sur T3/T4/T5 (0% → 80-100%), alors que
  **qwen3.5-9b réussit déjà en classique** sur ces tâches et gagne surtout en tokens, pas en
  réussite (`results/summary/codemode_effect.csv`). Bon exemple concret pour H5 (« ça dépend du
  modèle ») — à raconter comme deux histoires différentes, pas une moyenne.
- **Effet « tool bloat » (H6)** : passer de 12 à 60 outils fait grimper les tokens d'entrée du
  mode classique de **+55% à +114%** sur T3/T5 (`results/summary/tool_bloat_effect.csv`, figure
  `05_tool_count_effect.png`) — un vrai coût mesurable, pas juste une intuition.
- **Coût estimé total : ~2,44 $** (temps GPU x 0,50 $/h, un taux indicatif, pas un vrai prix
  cloud — bien le préciser dans l'article pour ne pas tromper le lecteur).

## Anecdotes tirées des traces (results/raw/traces/)

1. **Le modèle qui invente un nom d'outil corrompu.** gpt-oss-20b, de temps en temps, appelle un
   outil nommé `list_invoices<|channel|>commentary` au lieu de `list_invoices` — un jeton interne
   de son format de sortie (Harmony) qui fuit dans le nom de l'outil. Le serveur MCP répond
   poliment « outil inconnu », et le modèle retente parfois **exactement le même nom corrompu**
   plusieurs fois avant d'abandonner avec une liste vide. Bon exemple pour parler de la fragilité
   du tool-calling classique face à un format de sortie propriétaire mal maîtrisé par vLLM.
   (`gpt-oss-20b__classic__T3__tools12__rep0`)
2. **Le même bug, en pire.** Sur la variante à 60 outils, ce jeton corrompu casse carrément
   l'analyseur de vLLM côté serveur (erreur 500 franche) au lieu d'une simple erreur d'outil —
   comme si le bruit supplémentaire (48 outils inutiles) rendait le modèle plus susceptible de
   mal former sa réponse. Anecdote utile pour illustrer que le « tool bloat » n'est pas qu'un
   coût en tokens, il peut aussi dégrader la fiabilité. (`gpt-oss-20b__classic__T3__tools60__rep2`)
3. **Le mur de contexte, concrètement.** Pour T4, un seul appel `list_invoices(status=...)` pour
   un client avec beaucoup de factures renvoie à lui seul ~900 lignes / ~108 000 caractères —
   assez pour dépasser la fenêtre de 32 768 tokens **en un seul appel**, avant même que le modèle
   ait eu l'occasion de réfléchir. Bonne image pour expliquer visuellement pourquoi le Code Mode
   change la donne : les données brutes ne traversent jamais le contexte du modèle, seul le
   résultat filtré y entre.

## Limites à mentionner honnêtement

- **Échantillon petit** (10 répétitions par cellule, 5 pour la variante 60 outils) : les
  intervalles de confiance à 95% sont larges, surtout à 0% ou 100% observé — formuler les
  pourcentages extrêmes avec prudence (« au moins/au plus environ X% », pas un chiffre figé).
- **2 modèles sur les 4 prévus.** Gemma 4 et Qwen3.8-27B ont été abandonnés pour des conflits de
  dépendances durs sur cette VM (driver CUDA 12.8 trop ancien pour les versions de `transformers`
  qu'exigent leurs architectures) — à mentionner comme une limite matérielle du banc d'essai, pas
  un jugement sur ces modèles eux-mêmes.
- **Coût = temps GPU local, pas un prix cloud réel.** Le taux de 0,50 $/h est un repère indicatif
  de PLAN.md, pas une facture API.
- **Un bug de parseur spécifique à vLLM 0.19.0** (le jeton Harmony qui fuit) ne se reproduira
  peut-être pas avec une autre version de vLLM ou un autre serveur d'inférence — ne pas généraliser
  ce point précis à « gpt-oss-20b est peu fiable » en général.
- **Un seul profil matériel** (L40S-24Q, ~21,7 Go utilisables) : le mur de contexte observé est
  spécifique à ce budget VRAM ; un GPU avec plus de mémoire (donc un `max_model_len` plus grand)
  repousserait ce mur, pas ne l'éliminerait pas forcément pour autant sur de plus gros volumes.

## Figures disponibles (results/figures/, 1600px, style du blog)

1. `01_input_tokens_by_task.png` — tokens d'entrée par tâche, classic vs codemode, par modèle.
2. `02_gpu_time_per_trial.png` — temps GPU médian par essai.
3. `03_success_rate_by_task.png` — taux de réussite avec IC 95% (la plus parlante pour H7/H5).
4. `04_latency_by_task.png` — latence médiane par essai.
5. `05_tool_count_effect.png` — effet du nombre d'outils (12 vs 60), pour H6.
6. `06_failure_breakdown.png` — répartition des types d'échec par modèle/mode.
