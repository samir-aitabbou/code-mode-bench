# PROGRESS

> Tenu à jour à chaque étape, pour reprendre après une coupure (abonnement Pro limité en usage).
> Lire aussi `../CLAUDE.md` et `../PLAN.md` avant de reprendre.

## Fait

- **Étape 1 (vérifications VM)** : connexion SSH à la VM de calcul (alias configuré en local) OK.
  IP corrigée dans `CLAUDE.md` et `../CONTEXTE.md` (était fausse par erreur). GPU L40S-24Q, 1 MiB utilisé / 21 766 MiB libres (VRAM libre confirmée).
  Driver 570.124.06, CUDA 12.8. 16 CPU, 62 Gi RAM. Docker 29.4.3 présent (3 conteneurs tiers
  tournent déjà — `candide_blue_poc2-app/-db`, `elkhabir-db` — ne pas y toucher). Internet OK.
  317G disponibles sur `/`. `~/venv-vllm` existant avec vLLM 0.17.1 (ne pas modifier).
  `openai/gpt-oss-20b` déjà en cache HF sur la VM (72G de cache HF au total, autres modèles
  divers non liés au projet).

- **Vérification des 4 modèles sur HF + recipes.vllm.ai (2026-11-22)** : voir tableau dans
  `configs/models.yaml` (section `notes` de chaque modèle) pour le détail sourcé. Résumé :
  - `openai/gpt-oss-20b` (MXFP4 natif) : compatible vLLM ≥0.10.1, déjà en cache.
  - `Qwen/Qwen3.5-9B` → quant retenue `RedHatAI/Qwen3.5-9B-FP8-dynamic` (FP8, 14G) : compatible
    vLLM ≥0.17.0 (recette officielle), construit avec vllm==0.18.1.
  - `google/gemma-4-26B-A4B-it` → quant retenue `axteam/gemma-4-26B-A4B-AWQ-4bit` (AWQ 4-bit,
    17,6G) : **l'architecture MoE Gemma 4 nécessite vLLM ≥0.19.0**, donc incompatible avec
    `~/venv-vllm` (0.17.1). Décision de Samir (voir ci-dessous) : créer un venv séparé.
  - `Qwen/Qwen3.8-27B` → quant retenue `RedHatAI/Qwen3.8-27B-INT4` (AWQ-smoothed W4A16 GPTQ,
    compressed-tensors) : compatible vLLM ≥0.17.0 (recette officielle). Désaccord entre la
    recette officielle (`--tool-call-parser qwen3_coder`) et la fiche du modèle
    (`--tool-call-parser qwen3_xml`) — **à trancher empiriquement au test de fumée** (tester les
    deux, garder celui qui marche, noter le résultat dans PROGRESS.md et REPORT.md).
    `barrydeen/Qwen3.8-27B-AWQ-4bit` (AWQ vrai) écarté : 27,8G sur disque, trop gros pour 24G de
    VRAM avec le KV cache.

- **Décision de Samir (2026-11-22)** : un seul venv `~/code-mode-bench/.venv-vllm` avec une
  seule version de vLLM ≥0.19 pour les **4 modèles** (équité de comparaison), build CUDA 12.8
  (index `cu128`), sans toucher à `~/venv-vllm`.

- **Installation de vLLM sur la VM — plusieurs tentatives, voir le détail car c'est instructif :**
  1. `pip install vllm==0.32.0 --extra-index-url .../cu128` → a installé torch 2.13.0+**cu130**
     (pip a ignoré l'index cu128 car vllm 0.32.0 exige `torch==2.13.0` exactement, et ce torch
     n'a plus de build cu128 publié). `torch.cuda.is_available()` → **False** (driver limité à
     CUDA 12.8).
  2. `uv pip install vllm==0.32.0 --torch-backend=auto` → même résultat (le problème n'est pas
     la détection du backend, c'est l'absence de build cu128 pour torch 2.13.0).
  3. Recherche des versions de vLLM encore compatibles cu128 : vLLM ≥0.27.0 exige
     `torch==2.13.0` (pas de build cu128) ; 0.19.0–0.26.0 exigent `torch==2.10.0` ou `2.11.0`
     (build cu128 disponible) ; mais vLLM ≥0.25.0 exige en plus `torchcodec>=0.14`, dont le
     build cu128 ne va que jusqu'à 0.11.1 → bloquant aussi.
  4. **Retenu : vLLM 0.24.0** (dernière version sans la contrainte `torchcodec`, avec
     `torch==2.11.0`) + `uv pip install vllm==0.24.0 --torch-backend=cu128`. Résultat confirmé :
     `torch 2.11.0+cu128`, **`torch.cuda.is_available() == True`**, GPU détecté
     (`NVIDIA L40S-24Q`). ✅
  - Conséquence : on n'a pas exactement la version « testée » par la fiche HF du quant AWQ de
    Gemma 4 (0.32.0), mais on satisfait le plancher qu'elle indique (≥0.19.0). Un bug connu de
    chargement AWQ (vllm-project/vllm#40286, sur 0.19.1, corrigé par la PR #40708) n'est pas
    confirmé présent ou absent en 0.24.0 — **à vérifier empiriquement au smoke test de Gemma 4**
    (si l'erreur `KeyError` du bug apparaît, il faudra remonter de version et retrouver un
    conflit cu128, ou essayer un wheel CPU-offload / une autre quantification — en reparler à
    Samir si ça arrive).
  - `configs/models.yaml` mis à jour avec la version réelle (0.24.0) et cette explication.

- **Squelette du dépôt** : `git init` local dans `code-mode-bench/` (git config local, pas
  touché à la config git globale de Samir). Structure `src/cmbench/...`, `tests/`, `configs/`,
  `scripts/`, `docs/`, `results/`. `pyproject.toml` (hatch + uv, Python ≥3.11), `Makefile`,
  `.gitlab-ci.yml` (lint/types/tests, sans GPU ni API), `.gitignore`, `.gitattributes` (LF forcé),
  `.env.example` (URL vLLM local, clé factice `EMPTY`), `LICENSE` (MIT, © 2026 Samir Ait-Abbou),
  `README.md` (stub, sera complété à la fin). `configs/models.yaml` (4 modèles, infos sourcées)
  et `configs/experiment.yaml` (tâches, modes, répétitions, limites — 560 essais au total,
  conforme à PLAN.md §6).

- **Générateur de données** : `src/cmbench/data/constants.py` (vocabulaire partagé) et
  `src/cmbench/data/seed.py` (génération SQLite déterministe seed=42, volumes PLAN.md §3 :
  500 clients / 200 produits / 5000 commandes / 8000 factures / 1500 tickets / outbox vide).
  Garantit l'absence d'ambiguïté (T3 : écart >0,01 entre les 4 premiers clients VIP+South par
  impayés ; T4 : ≥5 clients avec >2 factures en retard ; T5 : écart >0,01 entre les 6 premières
  commandes Electronics d'octobre 2026) via `_verify_unambiguous`, avec un mécanisme de
  correction déterministe (`_nudge_ties`) si un seed futur produisait une égalité. `tests/test_seed.py`
  écrit (déterminisme, comptages, ids-ancres C0042/C0107/O1337, non-ambiguïté T3/T4/T5, reset).
  Pas encore exécuté (installation locale `uv sync --extra dev` en cours).

- **Serveur MCP** : `src/cmbench/mcp_server/tools.py` (logique pure, `CrmTools`, partagée entre
  les deux modes) + `server.py` (12 outils MCP) + `extra_tools.py` (48 outils « bruit » générés
  par template pour la variante 60 outils) + `adapter.py` (MCP `Tool` → `ToolSpec`). **Attention
  changement d'API important** : le paquet `mcp` installé est la version **2.3.0**, où `FastMCP`
  a été renommé `MCPServer` (`mcp.server.mcpserver`, pas `mcp.server.fastmcp`). Le code et les
  tests utilisent la nouvelle API. 24 tests passent (seed + tools + server).

- **Providers, modes et sandbox** (cœur du harness) :
  - `providers/base.py` : interface `Provider` (abstraite), `Usage`, `ToolSpec`, `ChatMessage`,
    `UsageMissingError` (essai invalide si pas de `usage`, jamais estimé — CLAUDE.md §3.4).
  - `providers/openai_compat.py` : client vLLM (clé factice `EMPTY`). Gère l'union de types
    `tool_calls` du SDK `openai` 3.x (function vs « custom » tool calls).
  - `sandbox/bridge.py` + `sandbox/runner.py` : bac à sable codemode. Le code du modèle tourne
    dans un sous-processus avec un module `tools` généré ; les appels `tools.xxx()` passent par
    un protocole JSON-RPC ligne par ligne (stdin/stdout) vers l'exécuteur réel du harness.
    Isolation réseau via `unshare -rn` sur Linux (testée au runtime, repli sans isolation +
    avertissement loggé si indisponible — **à vérifier sur la VM**), délai max, sortie tronquée
    à 8000 caractères. Testé de bout en bout (succès, erreur d'outil, timeout, troncature).
  - `modes/classic.py` + `modes/codemode.py` : les deux boucles d'agent, partageant le protocole
    `FINAL_ANSWER` (`modes/common.py`). `modes/typed_api.py` génère le module `tools.py` typé
    montré au modèle en codemode directement depuis les schémas JSON MCP (donc le prompt et
    l'exécution réelle ne peuvent pas diverger).
  - 16 nouveaux tests (40 au total), tous contre un `StubProvider` simulé (pas de GPU/API en
    CI) + le vrai sous-processus sandbox. Ruff et mypy propres.

- **Dépôt poussé sur la VM** (pas de `rsync` sur le PC Windows → utilisé `git` à la place,
  autorisé par CLAUDE.md §1) : `git init -b master` + `receive.denyCurrentBranch=updateInstead`
  dans `~/code-mode-bench` sur la VM, remote `vm` ajouté en local (alias VM + `:code-mode-bench`),
  `git push vm master`. `.venv-vllm` n'a pas été touché (gitignore `.venv-*/`). `uv` installé en
  user-space sur la VM via l'installeur officiel (`curl -LsSf https://astral.sh/uv/install.sh`,
  dans `~/.local/bin`, PATH non hérité en SSH non interactif → utiliser le chemin complet
  `~/.local/bin/uv`). `uv sync --extra dev` fait dans `~/code-mode-bench` (son propre `.venv`,
  séparé de `.venv-vllm`). **84 tests passent sur la VM (Linux)**.

- **CLI** (`src/cmbench/cli.py`, manquait malgré la référence dans `pyproject.toml`) :
  `data`, `run`, `smoke` (sort en erreur si échec, utilisé par `run_all.sh`), `bench`,
  `status`, `serve-cmd` (génère la commande `vllm serve` depuis `configs/models.yaml`, évite de
  parser du YAML en bash), `list-models`.

- **Isolation réseau du bac à sable — testée en conditions réelles sur la VM** :
  `unshare -rn` échoue (`write failed /proc/self/uid_map: Operation not permitted` — user
  namespaces non privilégiés désactivés sur cette VM). **Repli sur Docker confirmé** :
  l'utilisateur est dans le groupe `docker`, `docker run --network none python:3.12-slim`
  fonctionne sans sudo. `sandbox/runner.py` essaie `unshare -rn` puis `docker run --network
  none` puis repli sans isolation (avec avertissement loggé) si aucun des deux ne marche.
  Testé de bout en bout sur la VM (8 tests sandbox/codemode passent avec Docker actif).

- **Les 3 modèles manquants sont téléchargés** sur la VM (`scripts/download_model.py`, bande
  passante excellente : les 3 en moins d'une minute cumulée) : `RedHatAI/Qwen3.5-9B-FP8-dynamic`,
  `axteam/gemma-4-26B-A4B-AWQ-4bit`, `RedHatAI/Qwen3.8-27B-INT4`. Disque : 256G libres (sur 1008G),
  en baisse normale après les téléchargements.

- **Rebondissement sur la version de vLLM — `0.24.0` ne peut en réalité pas tourner ici :**
  le premier test de fumée (gpt-oss-20b) a planté au lancement de `vllm serve` avec
  `ImportError: libcudart.so.13: cannot open shared object file`. Investigation :
  - `~/venv-vllm` (0.17.1, existant, jamais testé en vrai jusqu'ici) s'importe sans problème —
    pas le même souci.
  - vLLM 0.19.0 installé **sans aucun forçage** (`pip install vllm==0.19.0`) choisit
    naturellement `torch==2.10.0+cu128` : import OK, `torch.cuda.is_available()==True`, **et un
    vrai produit matriciel sur GPU fonctionne**.
  - vLLM 0.24.0 installé **sans aucun forçage** choisit naturellement `torch==2.11.0+cu130` :
    `import vllm` réussit (torch cu130 embarque son propre `libcudart.so.13`), mais
    `torch.cuda.is_available()` est **False** (driver trop ancien pour CUDA 13). Le wheel
    vLLM 0.24.0 embarque une extension native (`vllm._C_stable_libtorch`) liée à CUDA 13,
    **indépendamment de la version de torch associée** — forcer torch en cu128 à la place
    casse l'import (`ImportError` direct, pas de `libcudart.so.13` nulle part) ; le laisser en
    cu130 permet l'import mais le GPU reste inutilisable (driver trop ancien). Dans les deux
    cas, 0.24.0 est **inutilisable sur ce driver**.
  - **Conclusion : `~/code-mode-bench/.venv-vllm` recréé avec vLLM 0.19.0 pur** (pas de
    `--torch-backend`, pas de bricolage), la version plancher annoncée par la fiche AWQ de
    Gemma 4. `configs/models.yaml` et ce fichier mis à jour. Risque résiduel à vérifier
    empiriquement au smoke test de Gemma 4 : le bug de chargement AWQ connu
    (vllm-project/vllm#40286) a été signalé sur 0.19.1 (un patch au-dessus de notre 0.19.0) —
    pas confirmé si 0.19.0 l'a aussi.
  - Les 3 modèles téléchargés précédemment restent valides (le téléchargement HF ne dépend pas
    de la version de vLLM).

- **Ajustements des flags `vllm serve` (universels, `cmbench serve-cmd`)** : `--max-num-seqs 16`
  (le harness n'envoie jamais de requêtes concurrentes ; le défaut 256 sur-réserve la VRAM pour
  un warm-up jamais utile — a causé un OOM CUDA au démarrage de gpt-oss-20b) et `--enforce-eager`
  (la capture de CUDA graphs optimise pour un débit concurrent dont on n'a pas besoin ici — a
  aussi causé un OOM, 28 MiB manquants sur 23,76 GiB).

- **Test de fumée gpt-oss-20b : ✅ OK** (classic + codemode, T1), après les deux fixes ci-dessus.

- **Test de fumée Qwen3.5-9B-FP8-dynamic : ✅ OK** après un fix tokenizer. Premier essai :
  `ValueError: Tokenizer class TokenizersBackend does not exist` même avec `--trust-remote-code`
  (ce n'est pas du code custom, juste un nom de classe que transformers<5 ne connaît pas — le
  flag ne peut rien y faire). Cause : le `tokenizer_config.json` du dépôt FP8 (`RedHatAI/...`)
  spécifie `tokenizer_class: TokenizersBackend`. Contournement : `--tokenizer Qwen/Qwen3.5-9B`
  (dépôt de base, `Qwen2Tokenizer` standard) tout en chargeant les poids quantifiés du dépôt
  RedHatAI — sans impact sur les poids. Noté dans `configs/models.yaml`.

- **Qwen3.8-27B-INT4 : tuning de `gpu_memory_utilization`.** 0.85 → KV cache 30 Mo trop court
  (0,64 Gio dispo vs 0,67 Gio requis). 0.90 → erreur vLLM indiquant seulement 20,84 Gio de
  mémoire réellement libre au démarrage sur les 23,76 Gio bruts : **ce profil de vGPU
  (L40S-24Q) réserve ~2,9 Gio que `nvidia-smi` n'affiche jamais comme "used"** par aucun
  processus — à retenir pour la suite, le vrai plafond utilisable n'est pas
  `total - nvidia-smi"used"`. Réglé à **0.87** (tient sous le vrai plafond avec marge pour le
  KV cache). Confirmé par démarrage réussi du serveur vLLM.

- **Décision de Samir (2026-11-22) : Gemma 4 26B-A4B est abandonné.** Le test de fumée a
  échoué avec une erreur pydantic définitive : `model type 'gemma4' but Transformers does not
  recognize this architecture`. Vérifié sur PyPI : transformers 4.57.6 est la toute dernière
  version 4.x avant le saut direct à 5.0.0 — l'architecture Gemma 4 MoE nécessite
  `transformers>=5.0`, que vLLM 0.19.0 (et toute version de vLLM compatible avec le driver
  CUDA 12.8 de cette VM) épingle à `<5`. **Ce n'est pas un problème de configuration mais un
  conflit de dépendances dur** : aucune combinaison pip-installable de vLLM + transformers ne
  peut à la fois reconnaître l'architecture Gemma 4 ET utiliser ce GPU (les versions de vLLM
  qui embarquent transformers≥5, ≥0.24.0, embarquent aussi une extension native liée à CUDA 13,
  inutilisable sur ce driver — voir l'historique plus haut). Présenté à Samir avec 3 options
  (abandonner / remplacer / tenter l'image Docker officielle vLLM) ; **Samir a choisi
  d'abandonner Gemma 4** et de continuer le benchmark avec les 3 modèles restants.
  `configs/models.yaml` et `configs/experiment.yaml` mis à jour (3 modèles, 420 essais au
  total au lieu de 560). À documenter dans `REPORT.md` §8 (Deviations from plan).

- **Bug du harness trouvé et corrigé : la limite de 600s par essai (`TRIAL_WALL_CLOCK_LIMIT_S`)
  ne fonctionnait pas.** `run_trial` entourait l'appel au fournisseur d'un
  `with ThreadPoolExecutor(...) as pool:` ; `future.result(timeout=600)` levait bien
  `TimeoutError`, mais la sortie du bloc `with` appelait `pool.shutdown(wait=True)`, qui
  **attendait quand même** que le thread abandonné (et sa requête HTTP en cours vers vLLM)
  se termine — rendant la limite inopérante en pratique. Repéré sur le test de fumée
  Qwen3.8-27B : le process a tourné 100+ minutes au lieu d'être coupé à 10 minutes.
  Corrigé (`src/cmbench/runner.py`) : plus de `with`, `pool.shutdown(wait=False)` sur chaque
  chemin de sortie. 84 tests passent. Deuxième correctif lié : un timeout HTTP côté client
  (`OpenAICompatProvider.request_timeout_s`) ajouté en filet de sécurité — mis d'abord à 900s
  (au-dessus de la limite d'essai, donc inutile), puis corrigé à **120s** après avoir observé
  qu'une requête abandonnée continuait de tourner sur vLLM en parallèle du prochain essai
  (`Running: 2 reqs`, KV cache grimpant à 86% en moins d'une minute — risque réel d'OOM sur
  une longue série). Les deux commits sont poussés sur la VM.

- **Découverte critique : Qwen3.8-27B-INT4 (RedHatAI) génère du charabia incohérent sous
  vLLM 0.19.0, quel que soit `--enforce-eager` ou non.** Le re-test du test de fumée (après
  les deux correctifs ci-dessus) échouait encore par timeout sur les deux essais (classic et
  codemode), même avec 120s par requête et 2 tentatives. Diagnostic en `curl` direct (hors
  harness, prompts triviaux type "2+2 ?", "capitale de la France ?", `max_tokens=30`,
  `temperature=0`) : le serveur répond **vite** (~1,3s) mais le champ `reasoning` est un
  mélange incohérent de caractères latins/chinois/arabes sans aucun sens, systématiquement,
  sur 5 générations indépendantes. Testé aussi en désactivant `--enforce-eager` (CUDA graphs
  réactivés, serveur de diagnostic séparé, port identique, modèle arrêté/relancé proprement) :
  **même résultat, charabia identique.** Donc : (a) ce n'est pas un problème de timeout/lenteur
  — le modèle ne produit jamais de token de fin cohérent, d'où les essais qui épuisent leur
  budget de tours/temps à chaque fois ; (b) ce n'est pas `--enforce-eager` qui casse
  l'architecture hybride (16 couches full-attention + 48 Gated DeltaNet) ; (c) c'est très
  probablement un bug de chargement/déquantification spécifique à ce schéma INT4
  AWQ-smoothed-W4A16 GPTQ (compressed-tensors) sous vLLM 0.19.0 — par analogie avec le bug AWQ
  connu déjà noté pour Gemma 4 (vllm-project/vllm#40286), mais sur un schéma de quantification
  différent, donc pas nécessairement le même bug exact. **Pas encore de piste de correction
  confirmée.** VRAM libérée après le diagnostic (serveur de diag + process `VLLM::EngineCore`
  orphelin tué manuellement — `kill` du process principal n'avait pas suffi, bien noter ça
  pour `stop_vllm.sh` : vérifier qu'il n'y a pas de sous-processus `VLLM::EngineCore` orphelin
  après un arrêt).
  **À trancher avec Samir avant d'aller plus loin — 2 modèles sur 4 potentiellement
  inutilisables sur cette VM.**

- **Investigation complémentaire (après accord de Samir pour chercher une quantification de
  remplacement) : le même charabia apparaît avec `--quantization compressed-tensors` forcé
  explicitement** (au lieu de la détection automatique) — élimine l'hypothèse d'une mauvaise
  détection de format. Sortie identique au token près (déterministe, `temperature=0`),
  confirmant que ce n'est pas un problème de flag.
  - Recherche HF de dépôts de remplacement : `avyukth/Qwen3.8-27B-AWQ-INT4` (18G, tient dans le
    budget) et `barrydeen/Qwen3.8-27B-AWQ-4bit` (27,8G, trop gros) utilisent **tous les deux**
    le même format `compressed-tensors` (même famille de chargeur que RedHatAI) — donc pas de
    piste évidente pour éviter le bug juste en changeant de dépôt.
  - **Hypothèse retenue, cohérente avec toutes les observations : même blocage que Gemma 4.**
    La fiche officielle de Qwen3.8-27B (notée dès le départ dans `configs/models.yaml`) exige
    `vLLM >= 0.17.0` **et** `transformers >= 5.8.0`. vLLM 0.19.0 (notre version, floor pour
    Gemma 4) épingle `transformers<5`. L'architecture hybride de Qwen3.8-27B (16 couches
    full-attention + 48 couches Gated DeltaNet, un type de couche récent) a probablement une
    implémentation native dans vLLM 0.19.0 qui dépend de comportements/API seulement présents
    dans transformers≥5.8 — contrairement à Gemma 4 (erreur immédiate, architecture non
    reconnue), ici le chargement réussit et le modèle tourne (vite, formes correctes) mais
    produit un calcul numériquement faux : pas de crash, juste un résultat incohérent. Cela
    expliquerait tous les faits observés (rapide, déterministe, indépendant de
    `--enforce-eager`, indépendant du flag `--quantization`).
    **Donc : changer de dépôt de quantification ne réglerait rien, puisque le problème serait
    dans l'implémentation de l'architecture elle-même, pas dans les poids.** Même mur que
    Gemma 4 (transformers≥5 vs CUDA 12.8 de cette VM), pour une raison différente (support
    d'architecture vs version de classe de modèle).

- **Décision de Samir (2026-11-23) : Qwen3.8-27B-INT4 est abandonné aussi.** Même mur que
  Gemma 4 (voir ci-dessus), confirmé après investigation complémentaire (flag de
  quantification forcé, 2 dépôts de remplacement vérifiés). **Le benchmark se fait avec 2
  modèles : gpt-oss-20b et Qwen3.5-9B-FP8-dynamic.** `configs/models.yaml` et
  `configs/experiment.yaml` mis à jour (280 essais au total au lieu de 560/420). Processus de
  diagnostic nettoyés, VRAM confirmée libre (1 MiB) à chaque étape.

## Prochaine étape

1. **Tableau de compatibilité final : 2 modèles sur 4 fonctionnels.**

   | Modèle | Statut | Détail |
   |---|---|---|
   | gpt-oss-20b | ✅ OK | Smoke test passé (classic + codemode, T1) |
   | Qwen3.5-9B-FP8-dynamic | ✅ OK | Smoke test passé, après contournement du tokenizer |
   | Gemma 4 26B-A4B | ❌ Abandonné | `transformers>=5.0` requis, incompatible avec vLLM utilisable sur ce GPU (CUDA 12.8 max) |
   | Qwen3.8-27B-INT4 | ❌ Abandonné | Même mur (`transformers>=5.8.0`) ; charge et tourne mais calcule faux (charabia déterministe), confirmé indépendant du quantization flag et de `--enforce-eager` |

2. **⏸ Présenter à Samir le pilote** (1 répétition de toutes les tâches, 2 modèles, 2 modes —
   voir CLAUDE.md §4 étape 8) avant de lancer la série complète (280 essais).
3. ✅ **Fait** : tests de fumée propres re-passés sur gpt-oss-20b et Qwen3.5-9B avec le runner
   corrigé — les deux `OK` (classic + codemode, T1), rapides (quelques secondes par essai,
   aucun souci de timeout). VRAM libérée après chaque arrêt.
4. ✅ **Fait : pilote exécuté** (1 répétition de T1-T6, 2 modèles, 2 modes = 24 essais). Voir
   les résultats et la projection ci-dessous.

## Résultats du pilote (24 essais, 1 rép. × 6 tâches × 2 modes × 2 modèles)

**gpt-oss-20b** — classic 4/6 OK, codemode 6/6 OK :
- classic : T1 OK, T2 OK, **T3 FAIL (tool_error, 144,7s)**, **T4 FAIL (context_overflow, 1,0s)**,
  **T5 FAIL (timeout, 600s — plafond atteint)**, T6 OK.
- codemode : les 6 tâches OK (T3 12,3s, T4 20,0s, T5 6,8s — bien plus rapide que classic quand
  ça marche).

**qwen3.5-9b** — classic 4/6 OK, codemode 6/6 OK :
- classic : T1 OK, T2 OK, T3 OK (142,0s), **T4 FAIL (context_overflow, 3,1s)**,
  **T5 FAIL (tool_error, 425,4s)**, T6 OK.
- codemode : les 6 tâches OK (T3 38,0s, T4 130,9s, T5 46,0s).

**Lecture** : les deux modèles réussissent presque toujours T1/T2 en classic (aucun signe de
bug du harness/correcteur, conforme au contrôle de cohérence de CLAUDE.md §4 étape 10). Le
Code Mode réussit systématiquement les tâches T3/T4/T5 (grosses volumétries : top-3/5 sur
des milliers de lignes), alors que le mode classic échoue régulièrement sur ces mêmes tâches
(dépassement de contexte, erreurs d'outils, ou timeout) — exactement l'effet que le
benchmark est censé mesurer (cohérent avec les hypothèses H1-H3 de PLAN.md). Bon signal que
le harness produit un résultat différenciant et pas juste du bruit.

**Projection de durée pour la série complète (280 essais, en extrapolant les latences du
pilote × nombre de répétitions prévu — approximation, surtout pour la variante 60 outils où
le pilote n'a pas de données directes)** :
- gpt-oss-20b : ~2,2h (série principale, 120 essais) + ~1,1h (variante 60 outils, 20 essais)
  ≈ **3,3h**
- qwen3.5-9b : ~2,3h (série principale) + ~0,9h (variante 60 outils) ≈ **3,2h**
- **Total estimé : ~6,5h** de calcul GPU (+ quelques minutes de changement de modèle/VRAM).
  Tient largement en une nuit avec `run_all.sh` dans `tmux`.

## Enquête d'équité demandée par Samir avant la série complète (2026-11-23)

Samir a demandé de vérifier 4 points avant de lancer la série, avec preuves. Résumé des
preuves et des correctifs (détails complets dans l'historique de session) :

**1. Erreurs d'outil en classic** : confirmé par le code et par trace que le message
d'erreur d'un outil est bien renvoyé au modèle, qui réessaie (pas d'arrêt à la première
erreur). Les échecs classic observés sur T3/T5 ne sont pas des erreurs d'outil mais des
pannes côté serveur/réseau — reclassées `provider_error` au lieu du `tool_error` trompeur
d'avant.

**2. Dépassement de contexte T4** : confirmé que `list_invoices(status=...)` existe et est
utilisé correctement par le modèle. Le dépassement vient du volume brut renvoyé par un seul
appel filtré (107 696 caractères, ~900 factures) et non d'un filtre manquant — conforme au
protocole voulu (classic doit faire entrer les lignes brutes dans son contexte, Code Mode
non).

**3. Timeout T5 gpt-oss** : diagnostic sans plafond de temps → le modèle ne boucle pas et ne
bute pas sur un résultat géant ; il passe ~6 tours consécutifs à ~100s/~11 500 tokens de
raisonnement caché sans réponse visible, puis produit un `FINAL_ANSWER` correct au tour 8,
~670s au total — juste au-dessus de l'ancien plafond de 600s. Le qwen3.5-9b T5 classic
« tool_error » était en fait un timeout réseau par requête (120s) alors que seulement 63,5s
avaient été utilisées sur le budget de l'essai.

**Correctif de biais** : les deux limites de temps (`max_trial_seconds`,
`request_timeout_s`) étaient trop serrées pour classic sur les tâches lourdes en données,
risquant de compter comme échecs des essais qui auraient réussi. Avec l'accord de Samir,
remontées à 900s / 400s (ce dernier ajusté deux fois : 120→180→400s, car qwen3.5-9b butait
encore à 180s avant de réussir à 400s — ~191s pour la requête la plus lente, confirmant une
génération lente mais progressive, pas un blocage). Les deux valeurs sont maintenant lues
depuis `configs/experiment.yaml` (`limits.max_trial_seconds`/`limits.request_timeout_s`) au
lieu d'être une constante Python figée — `max_trial_seconds` existait déjà dans le YAML mais
n'était lue par aucun code avant ce correctif.

**Bug trouvé en cours de route (pas spécifique à classic, corrigé)** : un essai abandonné
après timeout continuait de tourner en arrière-plan (jusqu'à 15 tours de plus),
consommant du GPU en même temps que l'essai suivant (observé : 2 requêtes concurrentes,
cache KV grimpant à 86%+). Corrigé par un thread démon + signal d'annulation vérifié en
haut de chaque tour (`classic.py`/`codemode.py`), qui borne le dépassement à une requête en
vol au lieu de 15. Testé (régression ajoutée dans `tests/test_runner.py`), poussé.

**4. 3e modèle proposé puis abandonné** : `Qwen/Qwen3-32B-AWQ` (dense, attention standard,
sans le risque d'architecture de Qwen3.8-27B) testé avec l'accord de Samir. Charge
correctement (19 Go sur disque) mais ses poids seuls consomment ~18,6 Go de nos ~20,8 Go
utilisables, ne laissant qu'environ 2,3 Go pour le cache KV — un `max_model_len` réaliste
d'environ 9000 tokens, bien en dessous des ~21 000-46 000 tokens de prompt déjà observés sur
T3/T4/T5. Abandonné (accord de Samir) : ce serait un biais de VRAM, pas une mesure valide de
ses capacités. La série reste à 2 modèles.

**Pilote refait pour T3/T4/T5 (tâches concernées) après les correctifs** :

```
gpt-oss-20b:   T3 classic FAIL(provider_error, bug connu du parseur gpt-oss, pas une régression)
               T3 codemode FAIL(provider_error, même bug, confirme qu'il n'est pas spécifique à classic)
               T4 classic FAIL(context_overflow, attendu) · T4 codemode FAIL(provider_error)
               T5 classic FAIL(wrong_answer — échec honnête, plus de faux timeout)
               T5 codemode OK
qwen3.5-9b:    T3 classic OK (142,1s) · T3 codemode OK (38,0s)
               T4 classic FAIL(context_overflow, attendu) · T4 codemode OK (130,7s)
               T5 classic OK (255,0s, 8 requêtes) · T5 codemode OK (45,8s)
```

## Second passage de Samir avant la série (2026-11-23/24) : 4 points supplémentaires

Samir a redemandé 4 vérifications avant le feu vert, avec un seul récap consolidé à la fin.

**1. Cause réelle du `provider_error` de gpt-oss-20b — trouvée et corrigée (pas un bug vLLM).**
Le `provider_error` n'était pas un bug du parseur Harmony de vLLM comme on le pensait : lu le
code source de `openai_tool_parser.py` dans vLLM — il attrape déjà sa propre
`JSONDecodeError` et renvoie le texte brut tel quel (`200 OK`), confirmé dans les logs du
serveur vLLM (le warning "Error decoding JSON tool call" est immédiatement suivi d'un `200
OK` pour la même requête). Le vrai bug était **côté client, dans notre propre
`src/cmbench/providers/openai_compat.py`** : `json.loads(tc.function.arguments or "{}")`
plantait sans être rattrapé quand vLLM renvoyait ce texte de repli non-JSON, et l'exception
remontait jusqu'en haut, étiquetée à tort `provider_error`, qui faisait échouer tout l'essai.
Corrigé en ajoutant un champ `ToolCall.parse_error` : au lieu de planter, le client renvoie
`arguments={}` + le message d'erreur, qui est ensuite présenté au modèle comme un résultat
d'outil normal (`ERROR: ...`), avec retry, exactement comme pour une erreur "outil inconnu"
déjà gérée. Vérifié empiriquement : en relançant T1-T4/T6 sur gpt-oss-20b après le correctif,
**zéro `provider_error`** (T3 classic échoue maintenant honnêtement en `wrong_answer`, T3
codemode et T4 codemode réussissent là où ils plantaient avant). Pas eu besoin des 30 minutes
allouées à la recherche infructueuse — un vrai correctif a été trouvé directement.

**2. Erreurs exactes de Gemma 4 et Qwen3.8-27B (pour validation de l'abandon par Samir) :**
- **Gemma 4 26B-A4B** : erreur pydantic définitive au test de fumée —
  `model type 'gemma4' but Transformers does not recognize this architecture`. Cause : son
  architecture MoE exige `transformers>=5.0`, que toute version de vLLM utilisable sur le
  driver CUDA 12.8 de cette VM épingle à `<5` (conflit de dépendances dur, pas un réglage).
- **Qwen3.8-27B-INT4** : pas de crash, mais calcul numériquement faux et déterministe —
  charge et répond vite (~1,3s) mais le champ `reasoning` est un charabia incohérent
  (mélange de caractères latins/chinois/arabes sans aucun sens), identique sur 5 générations
  indépendantes, identique avec/sans `--enforce-eager`, identique avec `--quantization
  compressed-tensors` forcé explicitement, et identique sur 2 dépôts de quantification de
  remplacement. Cause retenue : même mur que Gemma 4 mais côté architecture — la fiche
  officielle du modèle exige `transformers>=5.8.0`, absent de notre pile vLLM.
- Les deux décisions d'abandon restent recommandées : ce sont des murs de dépendances durs
  sur cette VM, pas des problèmes de réglage qu'on pourrait encore corriger.

**3. T1/T2/T6, 1 répétition, 2 modèles (re-vérifié à neuf après le correctif du point 1) :**
les **12 cellules réussissent** (6 par modèle) :
- gpt-oss-20b : classic T1/T2/T6 OK, codemode T1/T2/T6 OK.
- qwen3.5-9b : classic T1 OK (7,8s), T2 OK (7,0s), T6 OK (6,3s) ; codemode T1 OK (7,2s), T2 OK
  (9,4s), T6 OK (11,0s).
Dépasse largement le plancher de CLAUDE.md §4 étape 10 (« au moins un modèle réussit presque
à 100% en classic sur T1/T2 ») — les deux modèles le font.

**4. Essai à blanc de `run_all.sh`** (config réduite à T1×2 répétitions, liste de modèles
`gpt-oss-20b nonexistent-model qwen3.5-9b` pour forcer le chemin d'échec, dans
`results_dryrun/` séparé — jamais touché `results/raw/` ni `configs/models.yaml` réels) :
- enchaînement serve→smoke→série→stop→VRAM validé pour les 2 vrais modèles ;
- saut propre du modèle inexistant (`serve_failed`, pas de blocage de la suite) ;
- `run_status.json` et `run.log` mis à jour à chaque étape, `DONE` créé à la fin ;
- **reprise validée** : relancé le même essai à blanc, les 8 essais déjà faits sont tous
  sautés (`skipping already-completed trial ...`), compteur inchangé, VRAM toujours libre.
- **Incident en cours de route (hors scope du code du benchmark) : la VM est partagée.** Un
  collègue avait une session Claude Code active qui utilisait le port 8000
  (son appli `andriax`) et la VRAM (`ollama`) en parallèle, faussant d'abord le diagnostic (un
  `vllm serve` à nous restait bloqué en "Application startup complete" sans jamais répondre —
  en fait un conflit de port, pas un bug). On ne peut pas tuer les process d'un autre
  utilisateur sans `sudo`, et `sudo` demande un mot de passe interactif que je ne peux pas
  fournir : **décision avec Samir de ne pas forcer ça, et de changer notre port par défaut**
  (8000 → **8010**, confirmé libre) plutôt que de dépendre d'un port partagé sur cette VM.
  `scripts/serve_vllm.sh` et `src/cmbench/cli.py` mis à jour partout où 8000 était en dur,
  toujours surchageable via `CMBENCH_VLLM_PORT`.
- **Bug trouvé et corrigé en chemin (indépendant de l'incident de port) :** un process
  `VLLM::EngineCore` orphelin peut garder la VRAM occupée après un `kill -9` du process
  principal (SIGKILL sans arrêt propre) — observé en manipulant manuellement un serveur
  bloqué. `stop_vllm.sh` traque et tue maintenant systématiquement tout `VLLM::EngineCore`
  restant après avoir arrêté le process principal (sûr car un seul modèle tourne à la fois,
  cf. CLAUDE.md §2). `run_all.sh` vérifie désormais le code de sortie de `stop_vllm.sh` et
  s'arrête avec un statut `vram_not_freed` au lieu de continuer silencieusement si la VRAM
  n'est pas libérée.

**Durée projetée mise à jour** (à partir des latences réelles mesurées ci-dessus et au point
3, extrapolées × répétitions prévues — approximation : échantillon de 1-2 essais par cellule,
et la variante 60 outils est estimée avec les latences à 12 outils, probablement un peu
optimiste) :
- gpt-oss-20b : ~33 min (série principale) + ~14 min (variante 60 outils) ≈ **47 min**
- qwen3.5-9b : ~111 min (série principale) + ~40 min (variante 60 outils) ≈ **151 min**
- **Total calcul GPU : ~3,3h**, + changements de modèle (~10 min) ⇒ **environ 3,5h à 4,5h**
  de bout en bout. Plus bas que l'estimation du pilote (~6,5h) car : (a) le correctif du
  `provider_error` évite des essais qui plantaient vite puis auraient dû être repris, (b) les
  nouvelles données post-correctifs remplacent plusieurs échecs rapides du pilote par des
  essais qui tournent à terme (donc image plus complète mais pas forcément plus longue).

Résultats complets fusionnés dans `results/pilot_2026-11-23/*.jsonl` (archive, pas la série
finale — `results/raw/` a été vidé pour la série réelle). Plus aucun `timeout` artefact de
harness dans ce nouveau pilote ; les échecs restants sont soit le comportement intended du
protocole (T4 classic), soit un vrai échec du modèle (T5 gpt-oss), soit un bug connu et
documenté du parseur gpt-oss (indépendant du mode).

**⏸ En attente du feu vert de Samir pour lancer la série complète (280 essais, 2 modèles).**

## Feu vert reçu et série lancée (2026-11-24, 10h05)

Samir a donné le feu vert avec 5 conditions, toutes remplies avant le lancement :
1. `nvidia-smi` montré avant lancement (VRAM libre confirmée malgré le process du collègue).
2. Instantané GPU (`gpu_util_pct`, `gpu_mem_used_mib`, `gpu_mem_free_mib`, `gpu_compute_procs`)
   ajouté à chaque `TrialRecord` (`src/cmbench/runner.py`), testé (2 nouveaux tests dans
   `tests/test_runner.py`), vérifié en conditions réelles sur le premier essai du lancement.
3. Plus aucune tentative `sudo` sans lui demander explicitement au préalable — respecté.
4. Abandon de Gemma 4 et Qwen3.8-27B (et de Qwen3-32B-AWQ, proposé puis abandonné le même
   jour) consigné dans `DEVIATIONS.md`, avec le texte d'erreur exact de chaque cas.
5. Lancement confirmé : session `tmux cmbench` active, `run_status.json` qui avance,
   commandes de suivi données avant qu'il éteigne son PC.

Série lancée dans `tmux cmbench` sur la VM à 10h05. `gpt-oss-20b` a terminé ses 140 essais
(série principale + variante 60 outils) sans incident. `qwen3.5-9b` était en cours.

## Mise en pause à la demande de Samir (2026-11-24, 14h10) — VM laissée au collègue

Samir a demandé de libérer la VM pour son collègue. Mise en pause propre :
- `Ctrl-C` envoyé à la session `tmux cmbench` (interrompt `run_all.sh`, donc plus aucun
  nouvel essai lancé) ; l'essai en cours (`qwen3.5-9b__classic__T5__tools12__rep3`) a été
  interrompu **avant** d'être écrit dans le JSONL — aucun résultat corrompu ou partiel.
- vLLM (`qwen3.5-9b`) arrêté, VRAM vérifiée libre : `1 MiB` utilisé, `21766 MiB` libre, `0`
  process compute actif (`nvidia-smi`).
- Session `tmux cmbench` terminée (plus de serveur tmux actif sur la VM).

**État exact à la pause :**
- `gpt-oss-20b` : **140/140 essais** (série principale + variante 60 outils) — complet.
- `qwen3.5-9b` : **99/140 essais** — incomplet, 41 essais restants.

**Bug trouvé en pausant (à corriger avant la prochaine relance) :** `run_all.sh` ligne 65
(`uv run cmbench bench --model "$MODEL_ID"`) ne vérifie pas le code de sortie de `cmbench
bench`. Résultat : quand ce sous-processus est interrompu (Ctrl-C) ou plante avant d'avoir
traité tous les essais prévus, le script continue quand même vers `stop_vllm.sh` puis écrit
le statut `"done"` pour ce modèle et, à la fin de la boucle, crée `results/DONE` — alors que
`qwen3.5-9b` n'a en réalité traité que 99/140 essais. `results/run_status.json` disait donc
`qwen3.5-9b: "done"` et `results/DONE` existait, ce qui est **faux** : j'ai supprimé
`results/DONE` sur la VM et il ne faut pas se fier à ce fichier pour l'instant. À corriger :
faire échouer `run_all.sh` (ou au moins ne pas écrire `"done"`/créer `DONE`) si `cmbench
bench` ne se termine pas avec le code 0.

**Pour reprendre plus tard :** relancer `bash scripts/run_all.sh` dans un nouveau `tmux new
-s cmbench` reprendra correctement — les 99 essais déjà dans `results/raw/qwen3.5-9b.jsonl`
seront sautés (`skipping already-completed trial ...`), `gpt-oss-20b` sera aussi re-smoke-
testé puis entièrement sauté (140/140 déjà fait), et seuls les 41 essais restants de
`qwen3.5-9b` tourneront. Avant de relancer : corriger le bug `run_all.sh` ci-dessus (sinon
sans conséquence sur les données, juste sur la fiabilité du statut/`DONE`), et vérifier que
le collègue a bien libéré la VM (`nvidia-smi`, `ps aux | grep -i andriax`).

## Reprise puis seconde pause (2026-11-24, 15h02-15h06)

VRAM vérifiée libre (`1 MiB` utilisé) avant de relancer : `tmux new -s cmbench 'bash
scripts/run_all.sh'`. Comme prévu, `gpt-oss-20b` a été re-smoke-testé puis entièrement sauté
(140/140 déjà fait, statut resté `"done"`), et `qwen3.5-9b` a repris exactement sur l'essai
interrompu à la pause précédente (`qwen3.5-9b__classic__T5__tools12__rep3`).

Samir a redemandé une pause ~4 minutes plus tard. Même procédure (`Ctrl-C` sur `tmux
cmbench`, VRAM vérifiée libre côté nous). **Cette fois le correctif du bug `run_all.sh`
(ajouté à la pause précédente) a fonctionné comme prévu** : le log montre explicitement
`cmbench bench for qwen3.5-9b did not exit cleanly -- halting run_all.sh (not marking
done)`, statut écrit `bench_incomplete` (pas `"done"`), et **aucun** faux `results/DONE`
créé cette fois — plus besoin de corriger après coup. `run_status.json` remis à
`paused_by_user_99_of_140` par cohérence avec la pause précédente.

État inchangé : `qwen3.5-9b` toujours à **99/140**, l'essai `rep3 T5` de nouveau interrompu
avant écriture (aucune corruption). Note en passant : la VRAM observée à l'arrêt était à
`426 MiB` (pas `1 MiB`) — vérifié que ce n'est pas un reste de notre côté (aucun process
`vllm`/`EngineCore` restant), c'est le `llama-server` du collègue qui avait redémarré entre
les deux pauses.

## Série complète terminée (2026-11-24, 16h19)

Reprise finale (15h20) : VRAM vérifiée libre, relancée dans `tmux cmbench`. `gpt-oss-20b`
re-sauté (140/140). `qwen3.5-9b` a terminé ses 41 essais restants sans incident et sans
intervention — **280/280 essais au total**, `results/DONE` créé (légitimement cette fois :
`run_status.json` affiche `"done"` pour les deux modèles). VRAM libérée (1 MiB) à l'arrêt.
Résultats finaux commités sur la VM puis rapatriés en local (`git pull vm master`).

## Contrôles de cohérence (CLAUDE.md §4 étape 10) — tous passés

1. **Plancher T1/T2 classic** : les **deux** modèles réussissent 10/10 sur T1 et T2 en
   classic (largement au-dessus du plancher « au moins un modèle presque à 100% »).
2. **Comptage des essais** : chaque modèle a exactement 140/140 essais, répartis exactement
   comme prévu par `configs/experiment.yaml` (16 combinaisons mode×tâche×nb_outils, 10 réps
   pour les 12 combinaisons de la série principale, 5 réps pour les 4 combinaisons T3/T5 à
   60 outils) — aucune combinaison manquante ou en trop.
3. **Totaux tokens/temps GPU** : sommés directement depuis les 280 lignes JSONL (pas
   recalculés à la main) : **7 872 185 tokens au total**, **~4,89h de temps GPU cumulé**
   (gpt-oss-20b ≈2,13h, qwen3.5-9b ≈2,76h) — bien au-dessus de la pire estimation (~6,5h)
   car calculé en GPU-secondes réel et non en latence d'essai (le temps GPU mesuré par essai
   est net, sans l'attente réseau/retry).
4. **Relecture manuelle de 11 traces tirées au hasard** (6 succès, 5 types d'échec
   différents, les deux modèles, les deux modes) : **aucun bug du correcteur trouvé**. Le
   correcteur classe correctement : succès (réponse = attendu), `wrong_answer` (JSON
   syntaxiquement valide mais faux), `context_overflow`/`provider_error`/`timeout` (erreurs
   serveur/réseau, pas de JSON à comparer), `tool_error` vs `max_turns` (la règle « une erreur
   d'outil dans les 15 tours → classé tool_error, pas max_turns générique » fonctionne comme
   prévu dans le code).

**Découverte en relisant les traces (à documenter dans `REPORT.md` §5 Failure analysis) :**
gpt-oss-20b laisse parfois fuiter un jeton spécial du format Harmony dans le *nom* de l'outil
appelé (ex. `list_invoices<|channel|>commentary` au lieu de `list_invoices`), sur des tâches
avec beaucoup de contexte (T3, 12 et 60 outils). Le nom corrompu ne correspond à aucun outil
MCP connu → `ERROR: Unknown tool: ...` renvoyé au modèle, qui retente parfois exactement le
même nom corrompu plusieurs fois avant d'abandonner (`FINAL_ANSWER: []`, classé à raison
`wrong_answer`) ou d'épuiser ses tours (classé à raison `tool_error`, cf. règle ci-dessus).
Distinct du bug client déjà corrigé plus haut (point 1 du second passage de Samir) : ici
vLLM renvoie parfois une erreur serveur franche (500, `provider_error`, 4 essais) quand ce
jeton corrompu casse carrément l'en-tête du message côté serveur, pas seulement les
arguments JSON. C'est un vrai échec du modèle/parseur sous cette configuration (tâche
lourde en contexte), pas un bug de notre harness — à présenter comme tel.

## Rapport généré (2026-11-24, après la fin de la série)

Construit le pipeline de rapport complet (CLAUDE.md §2/§11), manquant jusqu'ici :
- `src/cmbench/report/aggregate.py` : JSONL → CSV (`results/summary/`) : `by_cell` (table
  principale), `codemode_effect`, `tool_bloat_effect`, `failure_breakdown`,
  `context_overflow_by_model`. 7 tests ajoutés (gère aussi le cas limite d'un groupe vide,
  trouvé par un test qui a cassé avant la correction — `tool_bloat_effect`/`codemode_effect`
  plantaient sur un `DataFrame` sans colonnes quand aucune tâche T3/T5 n'était présente).
- `src/cmbench/report/plots.py` : les 6 graphiques requis (CLAUDE.md §6), style du blog
  (CLAUDE.md §7) : 1600px, fond blanc, DejaVu Sans (Inter non installée sur cette machine,
  repli explicitement autorisé), couleurs classic/codemode/succès/échec/modèles conformes.
- `src/cmbench/report/render.py` : génère `REPORT.md` (anglais) entièrement à partir des CSV
  — aucun chiffre recopié à la main. Inclut `DEVIATIONS.md` directement comme source de la
  section « Deviations from plan » (fusion propre des niveaux de titres).
- Nouvelle commande `cmbench report` (enchaîne aggregate → plots → render).
- `docs/METHODOLOGY.md`, `docs/REPRODUCE.md` (anglais), `ARTICLE-NOTES.md` (français, avec
  chiffres sourcés + 3 anecdotes de traces + limites), `README.md` finalisé (CLAUDE.md §8).
- Suite complète : **98 tests passent**, ruff et mypy propres (`pandas-stubs` ajouté en dev
  dependency pour que mypy type correctement les DataFrames).

**Résultats clés du rapport final** (sourcés dans `results/summary/*.csv`) :
- 280 essais, ~4,9h de temps GPU cumulé, ~7,87M tokens, ~2,44 $ de coût GPU estimé.
- H7 confirmée nettement : les 20 échecs `context_overflow` sont tous en classic (0 en
  codemode).
- T4 : 0% de réussite classique pour les **deux** modèles, 80-100% en codemode — l'exemple le
  plus net et symétrique du jeu de données.
- H5 confirmée mais nuancée : gpt-oss-20b est « sauvé » par le Code Mode sur T3-T5 (0%→80-100%
  de réussite), qwen3.5-9b réussissait déjà en classique sur ces tâches et gagne surtout en
  tokens, pas en réussite.
- H6 confirmée : 12→60 outils fait grimper les tokens d'entrée classic de +55% à +114% sur
  T3/T5.
- Découverte additionnelle documentée dans `REPORT.md` §5 (Failure analysis) : fuite d'un
  jeton de format Harmony dans le nom d'outil appelé par gpt-oss-20b (voir section précédente
  de ce fichier), avec 2 traces concrètes citées.

## Variante T4p (sonde paginée) lancée (2026-11-24, 17h19-17h20)

Suite à l'analyse indépendante dans `ANALYSE-CLAUDE.md` (point A) : le 0% classique sur T4 vient
de la conception de l'outil (`list_invoices` sans plafond renvoie ~900 lignes en un seul appel,
au-delà de la fenêtre de 32 768 tokens), pas forcément d'une limite de raisonnement du modèle.

**Ajouté** :
- Nouvelle tâche `T4p` (`src/cmbench/tasks/tasks.yaml`), énoncé identique à T4, notée par le même
  correcteur (`graders.grade_task` traite "T4" et "T4p" de façon identique).
- `CrmTools.list_invoices_page` (`mcp_server/tools.py`) : variante paginée de `list_invoices` —
  plafonne `limit` côté serveur (200 par défaut) et renvoie `total_count`/`has_more`, pour qu'un
  modèle qui boucle sur `offset` puisse récupérer toutes les factures en retard sans jamais
  dépasser le plafond par appel.
- `build_server(..., paginated_invoices=True, invoices_page_limit=200)` (`mcp_server/server.py`) :
  bascule l'outil `list_invoices` exposé vers cette variante paginée — n'affecte aucune autre
  tâche/essai (flag `False` par défaut, comportement des 280 essais déjà enregistrés inchangé).
- `configs/experiment.yaml` : section `paginated_variant` (tasks: [T4p], modes: [classic],
  repetitions: 10) — mode classique uniquement (le Code Mode réussit déjà T4 avec l'outil non
  paginé, rien à sonder côté codemode).
- `build_trial_specs`/`run_trial`/`run_all_trials_for_model` (`runner.py`) câblés pour générer et
  exécuter ces essais en plus de la série existante (additif, essais déjà complétés toujours
  sautés par id).
- Tests ajoutés : `test_list_invoices_page_caps_limit_and_reports_pagination`,
  `test_call_tool_list_invoices_paginated_caps_limit`,
  `test_server_without_pagination_keeps_list_invoices_unbounded`,
  `test_build_trial_specs_paginated_variant_is_additive`, `test_t4p_grades_identically_to_t4`,
  mise à jour de `test_loads_all_6_tasks`. **103 tests passent**, ruff et mypy propres.

**Lancé** : `tmux new-session -d -s cmbench 'bash scripts/run_all.sh'` sur la VM (GPU vérifié
libre avant lancement : 1 MiB utilisé, 21 766 MiB libres, aucun process). `run_all.sh` boucle sur
les 2 modèles sans modification — les 140 essais déjà faits par modèle sont sautés, seuls les 10
nouveaux essais `T4p` tournent réellement. Confirmé en direct dans `results/run.log` : smoke test
OK, 140 essais sautés, premier essai `gpt-oss-20b__classic__T4p__tools12__rep6` lancé à 17h20.
Total attendu : 20 nouveaux essais (10 par modèle), classique uniquement.

**Prochaine étape** (à la relance de Samir) : une fois les 20 essais terminés (`results/DONE`
recréé), corriger `REPORT.md` selon les points C à F d'`ANALYSE-CLAUDE.md`, régénérer rapport et
graphiques (`cmbench report`), mettre à jour `ARTICLE-NOTES.md`.

## Corrections du rapport selon ANALYSE-CLAUDE.md, points C à F (2026-11-25)

Suite à la relecture indépendante `ANALYSE-CLAUDE.md`, correction de `REPORT.md` (régénéré, pas
édité à la main) sur les points demandés :

- **C (tokens non comparables)** : `codemode_effect` (`aggregate.py`) calcule désormais une
  colonne `tokens_comparable` (vrai seulement si classic **et** codemode ont un taux de réussite
  > 0 sur cette paire modèle/tâche). `render.py` affiche `n/a¹` avec une note de bas de page
  partout où ce n'est pas comparable (ex. gpt-oss sur T3/T4/T5 classique, où le `+919%` cité par
  l'analyse n'apparaît plus). H1 (executive summary + table d'hypothèses) ne cite plus que les
  paires comparables (qwen3.5-9b sur T3/T5), avec une formulation explicite au lieu d'un
  intervalle ambigu type « X% to Y% » (corrige aussi le point F sur la formulation de H1).
- **D (tool bloat sur les deux modes)** : `_bloat_range_text()` calcule désormais la fourchette
  par mode ; l'executive summary, §4.3 et la ligne H6 citent explicitement les deux (classique
  **et** codemode), avec l'explication (le `tools.py` généré liste aussi les 60 outils).
- **E (H3 infirmée sur les tâches lourdes)** : verdict H3 reformulé en « Confirmed on simple
  tasks; disconfirmed on data-heavy tasks », présenté comme un résultat secondaire à part entière,
  pas une exception à mettre de côté.
- **F (détails)** : ajout de la déviation sur la limite de 900s (absente de `DEVIATIONS.md`,
  maintenant documentée avec sa justification complète) ; explication de T4/T4p corrigée partout
  (§4.1 note, §4.5, exemple de trace §5, H7) : l'appel qui déborde porte sur *toutes* les factures
  en retard de *tous* les clients (filtre par statut seul), pas sur « un client avec beaucoup de
  factures ».
- **Nouvelle §4.5** dans `REPORT.md` : résultat de la sonde `T4p` (voir section précédente),
  présentée séparément de la grille principale à 6 tâches (exclue de `_main_results_table`, du
  tableau H7/context-overflow du cœur de l'expérience, et de la figure de répartition des échecs
  — `cli.py`/`plots.py` filtrent explicitement `task_id != "T4p"` partout où c'était nécessaire ;
  bug trouvé et corrigé au passage : les 4 fonctions de `plots.py` faisaient un `.loc[(t,
  "codemode"), ...]` sur chaque tâche de la grille 12-outils, qui plantait sur `T4p` (pas de ligne
  codemode) avant la correction).
- Tests ajoutés : `test_codemode_effect_flags_token_comparison_as_not_comparable_when_classic_fails`.
  **104 tests passent**, ruff et mypy propres. Rapport, CSV et figures régénérés avec
  `uv run cmbench report` à partir des 300 essais réels (rien recopié à la main).
- `ARTICLE-NOTES.md` mis à jour en conséquence (chiffres corrigés, anecdote T4 corrigée, nouvelle
  anecdote sur la sonde T4p, chiffres de tokens « propres » recentrés sur qwen3.5-9b).

## Refonte des figures et anonymisation avant publication (2026-11-25)

Suite à une relecture avant mise en public du dépôt :

- **Figures** : `src/cmbench/report/plots.py` refondu dans le style publication scientifique déjà
  utilisé par `scripts/paper_figures.py` (police serif STIX, palette Okabe-Ito, IC 95% de Wilson,
  médiane + IQR, hachures « non comparable » sous 50% de réussite, `T4p` inclus sur la figure de
  taux de réussite). `scripts/paper_figures.py` appelle désormais `plots.py` au lieu de dupliquer
  ses couleurs/statistiques. `results/figures/` ne contient plus qu'un seul jeu de 8 figures
  (01-06 = les 6 exigées par `CLAUDE.md` §6, 07-08 = schéma d'architecture et nuage de points sur
  la limite de contexte) ; l'ancien dossier `results/figures/paper/` (doublon) a été supprimé.
  `REPORT.md` embarque les 8 figures avec une légende numérotée (« *Figure N. …* »).
  `tests/test_plots.py` ajouté (10 tests) ; bug trouvé et corrigé au passage (`detail` manquant
  devient un `NaN` flottant après un aller-retour par un `DataFrame`/CSV, pas un `None` — faisait
  planter le nuage de points sur la limite de contexte). **113 tests passent, ruff et mypy
  propres.**
- **Anonymisation** : recherche de secrets/informations internes dans tout le dépôt (y compris
  `results/raw/traces` et l'historique git). Aucun secret (aucune clé API, aucun `.env` jamais
  committé). Deux éléments internes trouvés et corrigés : le nom d'hôte/domaine/IP réels de la VM
  de calcul dans `DEVIATIONS.md`, `docs/METHODOLOGY.md`, un commentaire de code
  (`sandbox/runner.py`) et plusieurs endroits de ce fichier ; le nom réel
  d'un collègue dans ce fichier. Corrigés dans les fichiers actuels, **puis l'historique git
  entier a été réécrit** (`git filter-repo --replace-text` puis `--replace-message`, sur
  autorisation explicite de Samir) pour que ces deux noms ne soient plus retrouvables dans aucun
  ancien commit ni message de commit. Une sauvegarde complète du dépôt a été faite avant
  l'opération. **Tous les hash de commit ont changé** (le tag `v1.0.0`, déjà poussé sur GitHub
  avec l'ancienne IP/l'ancien nom, a été conservé tel quel -- déjà exposé publiquement, le
  réécrire ne retirerait rien -- un nouveau tag `v1.1.0` a été créé sur l'état final). Remotes
  (`github`, `vm`) toujours configurés mais **plus synchronisés avec le nouvel historique** : un
  `push --force` vers les deux est nécessaire (commandes données à Samir, pas exécuté par Claude
  Code -- CLAUDE.md §0.2).
- **README.md** : lien GitHub dans le bloc BibTeX corrigé vers le vrai nom du dépôt
  (`code-mode-bench-`, avec le trait d'union -- `code-mode-bench` sans trait d'union, cité dans
  `../CONTEXTE.md` et dans la demande initiale de Samir, n'existe pas sur GitHub ; vérifié via
  l'API publique GitHub). Lien vers l'article 1 (déjà publié) ajouté dans le TL;DR. Nombre
  d'essais corrigé (280 -> 300, oubli lors de l'ajout de `T4p`).

## Commandes utiles

```bash
# remplacer your-gpu-vm par l'alias SSH configuré en local pour la VM de calcul
ssh your-gpu-vm "nvidia-smi --query-gpu=memory.used,memory.free --format=csv"
ssh your-gpu-vm "source ~/code-mode-bench/.venv-vllm/bin/activate && python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
ssh your-gpu-vm "tail -n 30 ~/code-mode-bench/results/run.log"
ssh your-gpu-vm "cat ~/code-mode-bench/results/run_status.json"
ssh your-gpu-vm "ls ~/code-mode-bench/results/DONE 2>&1"  # existe = la série (y compris T4p) est finie
```
