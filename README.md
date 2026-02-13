# 🏗️ BridgeMaster3D

**KI-gestützte 3D-Brückenoptimierung im Browser.**

BridgeMaster3D generiert strukturoptimierte Brückendesigns mithilfe von Topologieoptimierung (SIMP-Methode) und Finite-Elemente-Analyse — alles direkt im Browser über eine Gradio-Oberfläche.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Taichi](https://img.shields.io/badge/Taichi-GPU_Accelerated-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **3D-Topologieoptimierung** mit SIMP (Solid Isotropic Material with Penalization)
- **GPU-beschleunigte FEA** über Taichi (CUDA)
- **Automatische Materialmengenberechnung** basierend auf Sicherheitsfaktor
- **Interaktive 3D-Vorschau** mit Live-Parameter
- **STL & OBJ Export** für 3D-Druck und CAD
- **Intelligentes Pruning**: Entfernt automatisch schwebende und nicht-tragende Teile
- **Verschiedene Materialien**: Stahl, Beton, Holz, Carbon, PLA
- **Symmetrische Optimierung** für realistische Brückendesigns

---

## ⚠️ Systemanforderungen

> **WICHTIG: Eine leistungsstarke NVIDIA-Grafikkarte ist erforderlich!**

| Komponente | Minimum | Empfohlen |
|---|---|---|
| **GPU** | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3070+ |
| **VRAM** | 4 GB | 8+ GB |
| **RAM** | 8 GB | 16 GB |
| **Python** | 3.10 | 3.12+ |
| **OS** | Windows 10 / Linux | Windows 11 |

**Rechenzeit:** Ca. **5 Minuten pro Brücke** auf einer RTX 4070 bei Standardeinstellungen (60×20×15 Voxel). Größere Brücken oder höhere Auflösungen brauchen entsprechend länger.

Die Software nutzt **CUDA** zur GPU-Beschleunigung. Ohne NVIDIA-GPU fällt sie auf CPU zurück — das dauert dann **deutlich** länger (10–30× langsamer).

---

## 🚀 Installation

```bash
# Repository klonen
git clone https://github.com/DEIN-USERNAME/BridgeMaster3D.git
cd BridgeMaster3D

# Abhängigkeiten installieren
pip install -r requirements.txt
```

---

## ▶️ Starten

**Option 1:** Doppelklick auf `START_BridgeMaster.bat`

**Option 2:** Manuell in der Konsole:
```bash
python bridge_ui.py
```

Die App öffnet sich unter `http://127.0.0.1:7860` im Browser.

---

## 🎮 Bedienung

1. **Brückenmaße einstellen** (Länge, Breite, Höhe)
2. **Straßenverlauf anpassen** (Biegung, Stärke, Kopffreiheit)
3. **Material wählen** (Stahl, Beton, Holz, Carbon, PLA)
4. **Vorschau erneuern** um das Setup zu prüfen
5. **Optimierung starten** — die KI berechnet das optimale Design
6. **Ergebnis** im 3D-Viewer inspizieren und als STL/OBJ herunterladen

---

## 📁 Projektstruktur

```
BridgeMaster3D/
├── bridge_ui.py           # Gradio-Oberfläche (Frontend)
├── bridge_engine.py       # Taichi-FEA-Solver (Backend)
├── START_BridgeMaster.bat # Schnellstart-Skript
├── requirements.txt       # Python-Abhängigkeiten
└── README.md              # Diese Datei
```

---

## 🔧 Technologie

- **[Taichi](https://www.taichi-lang.org/)** — GPU-Computing-Framework für die FEA
- **[Gradio](https://www.gradio.app/)** — Web-UI-Framework
- **[NumPy](https://numpy.org/)** — Numerische Berechnungen
- **[Matplotlib](https://matplotlib.org/)** — 3D-Vorschau-Rendering

---

## 📄 Lizenz

MIT License — siehe [LICENSE](LICENSE) für Details.
