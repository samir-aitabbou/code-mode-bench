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

- **300 essais réels** (280 pour les 6 tâches principales + 20 pour la sonde `T4p`, voir plus
  bas), 2 modèles, ~5,2h de temps GPU cumulé, ~9,1M tokens au total (`results/summary/trials.csv`,
  sommes directes).
- **H7 confirmée nettement** : les 20 échecs `context_overflow` du cœur de l'expérience (hors
  sonde `T4p`) sont **tous** en mode classique (10 par modèle), zéro en Code Mode, et tous sur T4
  (`results/summary/context_overflow_by_model.csv`, figure `03_success_rate_by_task.png` et
  `06_failure_breakdown.png`).
- **T4 (boucle + écritures) : 0% de réussite classique pour les DEUX modèles**, 80-100% en Code
  Mode (`results/summary/by_cell.csv`). C'est la tâche la plus spectaculaire pour l'article :
  symétrique entre les deux modèles, donc pas un artefact d'un seul modèle. **Attention à la
  formulation exacte de la cause** (voir anecdote 3, corrigée) : l'appel qui déborde porte sur
  *toutes* les factures en retard de *tous* les clients (filtre par statut seul, sans filtre
  client), pas sur « un client avec beaucoup de factures ».
- **Nouveau : même avec un outil `list_invoices` bien conçu (paginé), le mode classique échoue
  encore 0/20 sur T4** (`T4p`, ajouté après coup — voir `REPORT.md` §4.5 et `DEVIATIONS.md`). Ce
  n'est plus un dépassement en un seul appel : le modèle doit enchaîner ~5 pages pour couvrir tous
  les clients, et comme le mode classique renvoie toute la conversation à chaque tour, le total
  cumulé dépasse quand même la fenêtre de 32 768 tokens dans la plupart des essais — ou le modèle
  abandonne en cours de route avec une liste de clients incomplète (et parfois sans même avoir
  appelé les outils d'écriture). **Message plus fort pour l'article** : paginer l'outil ne suffit
  pas, c'est la conception même du mode classique (tout renvoyer, à chaque tour) qui bute sur le
  contexte, pas seulement un outil mal conçu.
- **Chiffres de tokens « propres » (H1)** : seules les paires (modèle, tâche) où les deux modes
  réussissent assez souvent pour que la comparaison ait un sens sont citables. Chez gpt-oss-20b,
  T3/T4/T5 classique échouent à 0% — ses chiffres de tokens sur ces tâches comparent un essai
  avorté à un essai réussi et **ne doivent pas être cités** (`codemode_effect.csv`, colonne
  `tokens_comparable`). Le chiffre honnête vient de **qwen3.5-9b, le seul cas où les deux modes
  réussissent T3 et T5** : **T3 : -51% de tokens d'entrée. T5 : -65% de tokens d'entrée**
  (`results/summary/codemode_effect.csv`). Prendre qwen comme cas de référence pour le discours
  « gain de tokens » ; gpt-oss sert plutôt d'exemple de fragilité du mode classique (voir
  anecdotes 1-2).
- **Effet « tool bloat » (H6), dans les DEUX modes** : passer de 12 à 60 outils fait grimper les
  tokens d'entrée de **+55% à +114%** en classique **et** de **-6% à +97% en Code Mode**
  (`results/summary/tool_bloat_effect.csv`, figure `05_tool_count_effect.png`) — parce que le
  fichier `tools.py` généré pour le Code Mode liste aussi les 60 outils avec leur documentation
  complète. Le Code Mode n'est pas à l'abri de ce coût ici, faute d'un mécanisme de recherche
  d'outils ; à nuancer dans l'article (ce n'est pas un problème propre au MCP classique).
- **Coût estimé total : ~2,62 $** (temps GPU x 0,50 $/h, un taux indicatif, pas un vrai prix
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
3. **Le mur de contexte, concrètement (explication corrigée).** Pour T4, l'outil `list_invoices`
   ne permet pas de filtrer par client pour cette requête : un seul appel
   `list_invoices(status='overdue')` renvoie *toutes* les factures en retard de *tous* les ~500
   clients d'un coup — ~900 lignes / ~108 000 caractères — assez pour dépasser la fenêtre de
   32 768 tokens **en un seul appel**, avant même que le modèle ait eu l'occasion de réfléchir.
   (Ce n'est donc pas « un client avec beaucoup de factures » qui pose problème, mais l'absence de
   filtre par client sur cet appel précis.) Bonne image pour expliquer visuellement pourquoi le
   Code Mode change la donne : les données brutes ne traversent jamais le contexte du modèle, seul
   le résultat filtré y entre.
4. **Paginer l'outil ne suffit pas (nouvelle anecdote, après la sonde `T4p`).** On a testé
   l'hypothèse « si l'outil était bien conçu (paginé), le mode classique réussirait T4 ». Résultat :
   toujours 0/20. Exemple concret (`gpt-oss-20b__classic__T4p__tools12__rep9`) : le modèle ne fait
   que 2 appels d'outil au total, puis répond avec une liste de 19 clients sur les 212 attendus —
   sans jamais avoir appelé les outils d'écriture (note client, email) pour aucun d'eux. Il
   abandonne la pagination en cours de route plutôt que de dépasser le contexte d'un coup. Bonne
   anecdote pour nuancer le message « il suffit de bien concevoir l'outil » : la conception du mode
   classique (renvoyer toute la conversation à chaque tour) reste le vrai goulot d'étranglement.

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

## Figures disponibles (results/figures/, style publication scientifique)

Depuis le 25/11, toutes les figures (`results/figures/`, un seul jeu cohérent) sont générées par
`src/cmbench/report/plots.py` : police serif STIX, palette Okabe-Ito (classic = gris `#9a9a9a`,
Code Mode = bleu `#0072B2`), IC 95% de Wilson sur les taux de réussite, médiane + IQR sur
tokens/latence/temps GPU, et des barres hachurées pour signaler une cellule à moins de 50% de
réussite (chiffre non comparable, point C). `T4p` apparaît désormais sur la figure 3.

1. `01_input_tokens_by_task.png` — tokens d'entrée par tâche, classic vs codemode, par modèle.
2. `02_gpu_time_per_trial.png` — temps GPU médian par essai.
3. `03_success_rate_by_task.png` — taux de réussite avec IC 95% (la plus parlante pour H7/H5),
   `T4p` inclus en 7e catégorie (classique uniquement).
4. `04_latency_by_task.png` — latence médiane par essai.
5. `05_tool_count_effect.png` — effet du nombre d'outils (12 vs 60), pour H6.
6. `06_failure_breakdown.png` — répartition des types d'échec par modèle/mode.
7. `07_setup_diagram.png` — schéma d'architecture du harness (bonus, pas dans la liste CLAUDE.md §6).
8. `08_context_limit.png` — taille des requêtes sur T4/T4p face à la limite de contexte (bonus).

Régénération : `uv run cmbench report` (1-6) puis `uv run python scripts/paper_figures.py` (1-8,
le script appelle les mêmes fonctions de `plots.py`, aucune duplication de code).
