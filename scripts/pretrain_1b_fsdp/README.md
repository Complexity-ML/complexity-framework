# Reprise FSDP 1B — 70B tokens, 4 GPU

Préentraînement uniquement. Le refit sera une phase séparée.
Sources d’origine : `/Users/boris/Documents/Codex/2026-09-07/ok-x20/outputs/pretrain-1b-70b/`.
Le modèle, la recette et le lecteur sont conservés ; le lanceur est portable,
la reprise obligatoire, et la restauration récupère le plan exact depuis HF.

## Checkpoint vérifié le 27 septembre 2026

- Dépôt : `Pacific-i64/TR-HASH-MoE-1B-70B-Agentic-Pretraining`.
- Révision : `0ba0e730f9ef078a669e5fd84277bdb60903f57f`.
- Checkpoint : `step_0018000` ; 2 359 296 000 tokens déjà entraînés.
- Cible : 534 057 updates ; 69 999 919 104 tokens au total.
- Restant : 516 057 updates ; 67 640 623 104 tokens.
- 4 ranks, microbatch 2, contexte 16 384 ; 131 072 tokens/update.

Le manifeste, le hash de la recette et de l’état, ainsi que la présence et les
tailles des 11 fichiers distants ont été vérifiés (12 164 821 782 octets).
Les poids n’ont pas été téléchargés pour cet audit : leurs hashes seront
vérifiés par `restore.py` sur le serveur. Aucune reprise GPU n’est encore validée.

## Sur le serveur, depuis la racine du framework

Activer l’environnement CUDA/PyTorch avec Triton, Liger, Hugging Face Hub,
NumPy et TensorBoard, puis s’authentifier avec `hf auth login`.

```bash
python scripts/pretrain_1b_fsdp/restore.py --revision 0ba0e730f9ef078a669e5fd84277bdb60903f57f
bash scripts/pretrain_1b_fsdp/run.sh --limit 18002 --verify-resume
```

Le premier log `resumed` doit indiquer `step: 18000`. Le test effectue deux
updates et vérifie une sauvegarde/relecture du modèle, de l’optimiseur et des
RNG. Il reste nécessaire de mesurer la mémoire sur les 4 RTX 5090.
`--limit` est une étape absolue ; le planning de LR reste celui du run 70B.
Le lanceur refuse de partir de zéro. Ne pas exécuter `prepare.py` pour cette
reprise : `restore.py` télécharge le plan, le schedule et le tokenizer originaux.

Après validation, `bash scripts/pretrain_1b_fsdp/run.sh` reprend le dernier
checkpoint local complet. L’uploader doit être configuré avant un long run :
l’entraînement attend dès que trois checkpoints locaux ne sont pas envoyés.

`upload.py` conserve la politique historique : trois checkpoints réguliers,
un checkpoint d’interruption, suppression des anciens chemins distants. La purge de l’historique HF est
désactivée par défaut (`purge_history_after_eviction: false`).
Il utilise `runtime/hf-token`. Son lancement et cette politique restent à
vérifier avant exploitation ; l’audit n’a modifié aucun dépôt HF.
Les documents `README.original.md` et `TRAINING.md` décrivent l’ancien serveur.
