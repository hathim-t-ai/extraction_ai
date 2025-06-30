# Label Studio Setup & Launch Guide

Date: June 30, 2025  <!-- always use actual current date -->

This document captures the full reasoning, step-by-step commands, and configuration decisions for integrating Label Studio into the Extraction-AI project for data annotation.

---

## 1. Objectives

1. Provide an easy, reproducible way for any contributor to spin-up a local Label Studio instance inside the existing Rye-managed virtual environment.
2. Keep all dependencies isolated from the system Python and fully tracked in *pyproject.toml* under `[tool.rye].dev-dependencies`.
3. Ensure Mac OS compatibility (tested on Darwin 21.3.0) while keeping instructions generic enough for Linux/Windows users to adapt.

## 2. High-Level Plan (pseudocode)

```pseudo
# Pre-requisites
Confirm Homebrew installed (macOS) → brew update
Install libmagic via Homebrew (required by python-magic used by Label Studio)

# Rye environment
Ensure Rye is installed → curl … | bash (only once per machine)
rye sync           # installs project deps & sets up virtualenv

# Add Label Studio
rye add --dev label-studio>=1.12.0  # already committed in pyproject.toml
rye sync                            # pull new package & sub-deps

# Launch
rye run label-studio start          # first-time run creates ~/.labelstudio

# Access UI
Open http://localhost:8200 → create admin user
Create a new project, define labeling config (e.g. JSON bounding boxes or table annotation)
Import sample PDFs or images located in data/raw_pdfs/* for annotation

# Persist configuration (optional)
Add *.labelstudio* directory to .gitignore (user-specific)
```

## 3. Detailed Steps

### 3.1 System Packages (macOS)
```bash
# Homebrew install if missing
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew update
brew install libmagic # required by python-magic used internally
# Optional extras
brew install tesseract imagemagick git
```

### 3.2 Install / Update Rye (if missing)
```bash
curl -sSf https://rye.astral.sh/get | bash      # one-liner installer
exec $SHELL                                     # reload shell to add rye to PATH
```

### 3.3 Sync Project Dependencies
```bash
cd /Users/hathimamir/extraction_ai
rye sync  # installs everything listed in pyproject.toml incl. Label Studio
```

### 3.4 Launch Label Studio
```bash
rye run label-studio start -p 8200
# or specify a dedicated workspace location
# rye run label-studio start ~/labelstudio --port 8200
```
The first launch creates a `~/.labelstudio` folder holding user DB, media, and config.

Open the UI at http://localhost:8200 and follow the onboarding wizard to:
1. Create an admin account.
2. Create a "Bank-Statements" project.
3. Use the **Segmentation** setup if annotating text regions, or **OCR/Text** template for line-by-line labeling.
4. Upload PDFs from `data/raw_pdfs/BankStatements/*` or drag-and-drop images.
5. Start labeling.

### 3.4.1 Docker One-Liner (zero Python deps)
```bash
docker run --name labelstudio -d -p 8200:8200 -v $(pwd)/labelstudio_data:/label_studio/data heartexlabs/label-studio:latest
```
This pulls the latest stable container and mounts a local volume so that your tasks and annotations persist. Use `docker logs -f labelstudio` to tail server logs.

### 3.5 CLI Shortcuts
```bash
# list current projects
rye run label-studio list

# export annotations as JSON/CSV
rye run label-studio export <project_id> --export-type JSON
```

## 4. Next Steps & Automation Ideas
- Add `tools/label_studio_sync.py` to sync finished annotations into `data/annotations/` (future task).
- CI job to validate that Label Studio can launch headlessly (using xvfb).

---

## 5. Decisions & Rationale
- **Rye vs pip**: follows project guideline #7 to maintain a single dependency manager.
- **Dev-dependency**: Label Studio is a tooling dependency, not runtime code, hence placed under `[tool.rye].dev-dependencies`.
- **Port 8200**: chosen to avoid clash with common dev servers (8000, 8080).
- **libmagic**: mandatory for file type detection when importing data.

---

## 6. Known Issues / False Paths
| Date | Attempt | Outcome |
|------|---------|---------|
| 30-Jun-2025 | Tried using `pipx install label-studio` outside Rye | Works, but violates dependency policy. Switched to Rye-managed install. |

---

*(End of file)* 