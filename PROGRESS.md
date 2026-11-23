# PROGRESS

> Tenu à jour à chaque étape, pour reprendre après une coupure (abonnement Pro limité en usage).
> Lire aussi `../CLAUDE.md` et `../PLAN.md` avant de reprendre.

## Fait

- **Étape 1 (vérifications VM)** : connexion SSH `redacted-gpu-vm` (alias, résout vers REDACTED-IP,
  `redacted-gpu-vm.example`) OK. IP corrigée dans `CLAUDE.md` et `../CONTEXTE.md` (était
  `.199` par erreur). GPU L40S-24Q, 1 MiB utilisé / 21 766 MiB libres (VRAM libre confirmée).
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
  dans `~/code-mode-bench` sur la VM, remote `vm` ajouté en local (`redacted-gpu-vm:code-mode-bench`),
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

Résultats complets fusionnés dans `results/pilot_2026-11-23/*.jsonl` (archive, pas la série
finale — `results/raw/` a été vidé pour la série réelle). Plus aucun `timeout` artefact de
harness dans ce nouveau pilote ; les échecs restants sont soit le comportement intended du
protocole (T4 classic), soit un vrai échec du modèle (T5 gpt-oss), soit un bug connu et
documenté du parseur gpt-oss (indépendant du mode).

**⏸ En attente du feu vert de Samir pour lancer la série complète (280 essais, 2 modèles).**

## Commandes utiles

```bash
ssh redacted-gpu-vm "nvidia-smi --query-gpu=memory.used,memory.free --format=csv"
ssh redacted-gpu-vm "source ~/code-mode-bench/.venv-vllm/bin/activate && python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```
