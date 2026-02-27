# D-FINE – Beach Volleyball Detection

Dieses Repository basiert auf [D-FINE](https://arxiv.org/abs/2410.13842) und wird als Detektor für das Beach-Volleyball-Tracking-System eingesetzt. Die Detektionsergebnisse werden als Detection-Datenbank im MOTRv2-Format exportiert und anschließend von [MOTRv2](../MOTRv2) für das Multi-Object-Tracking genutzt.

Das originale README ist unter [README_ORIGINAL.md](README_ORIGINAL.md) verfügbar.

---

## Installation

```bash
conda create -n dfine python=3.11.9
conda activate dfine
pip install -r requirements.txt
```

---

## Inferenz (Beach Volleyball → MOTRv2-Format)

Gewichte separat herunterladen und im Root-Verzeichnis ablegen.

```bash
python tools/inference/torch_inf.py \
    -c configs/dfine/objects365/dfine_hgnetv2_l_obj365.yml \
    -r dfine_l_obj365.pth \
    -i "Sequenz_Beach.mp4" \
    --motrv2 \
    --sequence-name "volleyball/test/test1" \
    --allowed-classes "0,1,36,156,240" \
    --motrv2-score-threshold 0.3
```

**Parameter:**
- `-c` – Modell-Konfigurationsdatei
- `-r` – Gewichte des vortrainierten Modells
- `-i` – Eingabevideo
- `--motrv2` – Aktiviert MOTRv2-kompatiblen JSON-Output (Detection-DB)
- `--sequence-name` – Name der Sequenz im MOT-Format (z.B. `volleyball/test/test1`)
- `--allowed-classes` – Kommagetrennte COCO-Klassen-IDs (0=Person, 1=Bicycle, 36=Sportball, etc.)
- `--motrv2-score-threshold` – Minimaler Konfidenz-Score für Einträge in der Detection-DB

---

## Hinzugefügte Dateien

### Inferenz-Tools (`tools/inference/`)

| Datei | Beschreibung |
|---|---|
| `torch_inf.py` | Modifizierte Inferenz – erweitert um `--motrv2`-Flag für MOTRv2-kompatiblen JSON-Output sowie Klassen-Filterung |
| `batch_process_dataset.py` | Batch-Verarbeitung des gesamten Volleyball-Datensatzes (alle Sequenzen) |
| `batch_processor.py` | Hilfsskript: verarbeitet mehrere Videosequenzen und erzeugt eine gemeinsame Detection-DB |
| `create_detection_db_for_finetune.py` | Erstellt Detection-DB aus allen Finetuning-Bildern für MOTRv2-Training |
| `process_video_interactive.py` | Interaktive Bearbeitung: Hofauswahl per Maus auf dem ersten Frame, dann Detektion auf ROI beschränkt |
| `process_video_manual_coords.py` | Videobearbeitung mit manuell angegebenen Koordinaten (kein GUI, für Cluster-Nutzung) |

### SLURM-Jobs

| Datei | Beschreibung |
|---|---|
| `run_dfine_training.slurm` | D-FINE Finetuning auf Volleyball-Daten auf dem Cluster |
| `slurm_torch_inf_gt.slurm` | Inferenz mit Ground-Truth-Evaluation auf dem Cluster |

### Konfigurationen (`configs/dfine/custom/`)

Finetuning-Konfigurationen für das Volleyball-Custom-Dataset (verschiedene Modellgrößen: n, s, m, l, x). Basieren auf den Objects365-Vortrainings-Gewichten.

### Sonstige Dateien

| Datei | Beschreibung |
|---|---|
| `det_db_motrv2_Volleyball_only.json` | Fertige Detection-DB für MOTRv2 (nur Volleyball-Sequenzen, direkt verwendbar) |
| `motrv2_training_config.json` | Trainings-Konfiguration für den kombinierten MOTRv2-Workflow |
| `directory_struktur.txt` | Dokumentation der Verzeichnisstruktur des Projekts |
