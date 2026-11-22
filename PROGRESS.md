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

## Prochaine étape

1. Test de fumée gpt-oss-20b (déjà en cache) avec vLLM 0.19.0, puis les 3 autres (dont test
   empirique qwen3_coder vs qwen3_xml pour Qwen3.8-27B, vérification du chargement AWQ de
   Gemma 4 avec 0.19.0 précisément, et question ouverte sur `--chat-template` pour Gemma 4 —
   voir `configs/models.yaml`).
2. **⏸ Présenter à Samir le tableau de compatibilité final + résultats du test de fumée avant
   le pilote.**

## Commandes utiles

```bash
ssh redacted-gpu-vm "nvidia-smi --query-gpu=memory.used,memory.free --format=csv"
ssh redacted-gpu-vm "source ~/code-mode-bench/.venv-vllm/bin/activate && python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```
