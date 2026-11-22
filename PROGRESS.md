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
  (index `cu128`), sans toucher à `~/venv-vllm`. Version choisie : **vLLM 0.32.0** (version
  explicitement citée comme testée sur la fiche HF du quant AWQ de Gemma 4). Installation en
  cours sur la VM (`pip install vllm==0.32.0 --extra-index-url https://download.pytorch.org/whl/cu128`
  dans `~/code-mode-bench/.venv-vllm`) — **vérifier `torch.cuda.is_available()` avant tout
  téléchargement de modèle**, comme demandé.

- **Squelette du dépôt** : `git init` local dans `code-mode-bench/` (git config local, pas
  touché à la config git globale de Samir). Structure `src/cmbench/...`, `tests/`, `configs/`,
  `scripts/`, `docs/`, `results/`. `pyproject.toml` (hatch + uv, Python ≥3.11), `Makefile`,
  `.gitlab-ci.yml` (lint/types/tests, sans GPU ni API), `.gitignore`, `.gitattributes` (LF forcé),
  `.env.example` (URL vLLM local, clé factice `EMPTY`), `LICENSE` (MIT, © 2026 Samir Ait-Abbou).
  `configs/models.yaml` (4 modèles, infos sourcées) et `configs/experiment.yaml` (tâches, modes,
  répétitions, limites — 560 essais au total, conforme à PLAN.md §6). Commit initial fait.

## En cours

- Installation de vLLM 0.32.0 + PyTorch (cu128) dans `~/code-mode-bench/.venv-vllm` sur la VM
  (tâche de fond, pas encore confirmée terminée). Prochaine vérif : `torch.cuda.is_available()`.

## Prochaine étape

1. Confirmer l'installation vLLM 0.32.0 + `torch.cuda.is_available() == True` sur la VM.
2. Écrire `src/cmbench/data/seed.py` (génération SQLite déterministe seed=42, volumes de
   PLAN.md §3) + ses tests (déterminisme, pas d'égalité en top 3/5, ≥5 clients pour T4).
3. Serveur MCP (`src/cmbench/mcp_server/server.py`, 12 outils + `extra_tools.py` pour la
   variante 60 outils) + tests.
4. Modes `classic`/`codemode`, sandbox, pont, fournisseur vLLM (`openai_compat.py`), avec tests
   sur un fournisseur simulé (CI sans GPU).
5. `scripts/serve_vllm.sh`, `scripts/stop_vllm.sh`, `scripts/run_all.sh`, runner avec reprise.
6. Test de fumée gpt-oss-20b (déjà en cache), puis les 3 autres (dont test empirique
   qwen3_coder vs qwen3_xml pour Qwen3.8-27B).
7. **⏸ Présenter à Samir le tableau de compatibilité final + résultats du test de fumée avant
   le pilote.**

## Commandes utiles

```bash
ssh redacted-gpu-vm "nvidia-smi --query-gpu=memory.used,memory.free --format=csv"
ssh redacted-gpu-vm "source ~/code-mode-bench/.venv-vllm/bin/activate && python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```
