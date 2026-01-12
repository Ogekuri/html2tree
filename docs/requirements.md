---
title: "Requisiti html2tree"
description: Specifica dei requisiti software
version: "0.2"
date: "2026-01-12"
author: "Codex CLI"
scope:
  paths:
    - "**/*.py"
    - "**/*.ipynb"
    - "**/*.c"
    - "**/*.h"
    - "**/*.cpp"
  excludes:
    - ".*/**"
visibility: "draft"
tags: ["markdown", "requisiti", "bozza"]
---

# Requisiti html2tree
**Versione**: 0.2
**Autore**: Codex CLI  
**Data**: 2026-01-12

## Indice
<!-- TOC -->
- [Requisiti html2tree](#requisiti-html2tree)
  - [Indice](#indice)
  - [1. Introduzione](#1-introduzione)
    - [1.1 Regole del documento](#11-regole-del-documento)
    - [1.2 Scopo del progetto](#12-scopo-del-progetto)
  - [2. Requisiti di progetto](#2-requisiti-di-progetto)
    - [2.1 Funzioni di progetto](#21-funzioni-di-progetto)
    - [2.2 Vincoli di progetto](#22-vincoli-di-progetto)
    - [2.3 Componenti e librerie richieste](#23-componenti-e-librerie-richieste)
  - [3. Requisiti](#3-requisiti)
    - [3.1 Progettazione e implementazione](#31-progettazione-e-implementazione)
    - [3.2 Funzioni](#32-funzioni)
  - [4. Requisiti di test](#4-requisiti-di-test)
  - [5. Cronologia delle revisioni](#5-cronologia-delle-revisioni)
<!-- TOC -->

## 1. Introduzione

### 1.1 Regole del documento
Questo documento deve rispettare sempre le seguenti regole:
- Questo documento deve essere scritto in italiano.
- I requisiti devono essere formattati come elenco puntato, utilizzando le parole chiave "deve" o "dovra" per indicare azioni obbligatorie.
- Ogni ID requisito (per esempio **PRJ-001**, **CTN-001**, **DES-001**, **REQ-001**, **TST-001**) deve essere univoco; non assegnare lo stesso ID a requisiti diversi.
- Ogni ID requisito deve iniziare con la stringa che identifica il gruppo:
  * Tutti i requisiti di funzione di progetto iniziano con **PRJ-**
  * Tutti i requisiti di vincolo di progetto iniziano con **CTN-**
  * Tutti i requisiti di progettazione e implementazione iniziano con **DES-**
  * Tutti i requisiti funzionali iniziano con **REQ-**
  * Tutti i requisiti di test iniziano con **TST-**
- Ogni requisito deve essere identificabile, verificabile e testabile.
- A ogni modifica di questo documento:
  - Aggiornare la data a quella odierna sia nel corpo del documento sia nell'intestazione.
  - Incrementare il numero di versione sia nel corpo del documento sia nell'intestazione.
  - Aggiungere una nuova riga alla tabella della cronologia revisioni descrivendo il cambiamento.

### 1.2 Scopo del progetto
html2tree e una applicazione CLI Python per la gestione di comandi informativi e operativi (help, versione, upgrade, uninstall) con controllo online della versione e supporto alla distribuzione tramite uv/uvx.

## 2. Requisiti di progetto

### 2.1 Funzioni di progetto
- **PRJ-001**: Il progetto deve fornire una applicazione CLI denominata `html2tree`.
- **PRJ-002**: Il progetto deve fornire funzionalita di gestione versione, help, upgrade, uninstall e controllo versione online.
- **PRJ-003**: Il progetto deve includere script di automazione locale e pipeline di rilascio GitHub.

### 2.2 Vincoli di progetto
- **CTN-001**: Il progetto deve essere implementato in Python e organizzato sotto `src/html2tree/`.
- **CTN-002**: Il progetto deve essere eseguibile con `uvx` e installabile con `uv`.
- **CTN-003**: Il progetto deve includere `requirements.txt` con `build`, `setuptools`, `wheel`, `pytest`.
- **CTN-004**: Il progetto deve includere `pyproject.toml` con i metadati necessari al packaging.
- **CTN-005**: Il workflow GitHub deve essere attivato da tag `v*` e contenere i passi indicati nella richiesta.
- **CTN-006**: Il controllo versione online deve usare un timeout di 1 secondo per evitare blocchi percepibili.

### 2.3 Componenti e librerie richieste
- **CTN-007**: Il progetto deve dipendere dai pacchetti `build`, `setuptools`, `wheel`, `pytest` in `requirements.txt`.
- **CTN-008**: Il progetto deve integrare `uv`/`uvx` per installazione ed esecuzione live.

## 3. Requisiti

### 3.1 Progettazione e implementazione
- **DES-001**: I sorgenti della CLI devono risiedere in `src/html2tree/cli.py`.
- **DES-002**: I punti di ingresso devono essere implementati in `src/html2tree/__init__.py` e `src/html2tree/__main__.py`.
- **DES-003**: La versione deve essere definita in `src/html2tree/version.py` nel formato `major.minor.patch`, inizialmente `0.0.0`.
- **DES-004**: Il workflow di rilascio deve essere definito in `.github/workflows/release-uvx.yml` e includere i passi: Checkout, Set up Python, Set up uv, Install build dependencies, Build distributions, Attest build provenance, Create GitHub Release and upload assets.
- **DES-005**: Devono essere presenti gli script di root `venv.sh`, `tests.sh` e `html2tree.sh`.
- **DES-006**: La struttura del progetto deve includere almeno i seguenti file e cartelle (escludendo directory con prefisso punto):
```
html2tree/
├─ LICENSE
├─ README.md
├─ docs/
│  ├─ requirements.md
│  └─ requirements_DRAFT.md
├─ requirements.txt
├─ pyproject.toml
├─ venv.sh
├─ tests.sh
├─ html2tree.sh
└─ src/
   └─ html2tree/
      ├─ __init__.py
      ├─ __main__.py
      ├─ cli.py
      └─ version.py
```
- **DES-007**: Devono essere presenti unit test per la CLI che coprano `--version`, `--help`, `--upgrade` (con mock di `uv`), `--uninstall` (con mock di `uv`) e il controllo versione online.

### 3.2 Funzioni
- **REQ-001**: L'opzione `--help` deve stampare la schermata di usage e terminare l'esecuzione.
- **REQ-002**: In assenza di parametri, il programma deve stampare la schermata di usage e terminare l'esecuzione.
- **REQ-003**: La schermata di usage deve indicare il nome `html2tree` e la versione corrente.
- **REQ-004**: L'opzione `--version` e il suo alias `--ver` devono stampare la versione nel formato `"0.1.2"` e terminare l'esecuzione.
- **REQ-005**: L'opzione `--upgrade` deve eseguire `uv tool install usereq --force --from git+https://github.com/Ogekuri/html2tree.git`.
- **REQ-006**: L'opzione `--uninstall` deve eseguire `uv tool uninstall html2tree`.
- **REQ-007**: Dopo la validazione dei parametri e prima di ogni altra operazione, il programma deve verificare online la presenza di una nuova versione.
- **REQ-008**: La verifica online deve chiamare `https://api.github.com/repos/Ogekuri/html2tree/releases/latest` con timeout di 1 secondo e determinare l'ultima versione dalla risposta JSON.
- **REQ-009**: Se la chiamata API fallisce o la versione non e determinabile, il programma deve considerare la versione corrente come ultima e non emettere avvisi.
- **REQ-010**: Se la versione disponibile e maggiore di quella corrente, il programma deve stampare il messaggio: `A new version of html2tree is available: current X.Y.Z, latest A.B.C. To upgrade, run: html2tree --upgrade` con i valori correnti.
- **REQ-011**: Lo script `venv.sh` deve verificare se esiste `.venv`, rimuoverla se presente, creare un nuovo ambiente e installare tutte le dipendenze di `requirements.txt`.
- **REQ-012**: Lo script `tests.sh` deve eseguire tutti gli unit test implementati.
- **REQ-013**: Lo script `html2tree.sh` deve eseguire il programma passando tutti gli argomenti ricevuti dalla riga di comando.

## 4. Requisiti di test

| ID Requisito di Test | Requisito collegato / Contesto | Procedura di test (chiara e riproducibile) |
|----------------------|---------------------------------|-------------------------------------------|
| **TST-001**          | **REQ-004**                     | Eseguire la CLI con `--version` e `--ver`. Verificare che stampi la versione nel formato richiesto e termini senza errori. |
| **TST-002**          | **REQ-001**, **REQ-002**, **REQ-003** | Eseguire la CLI con `--help` e senza parametri. Verificare che stampi la schermata di usage con nome e versione e termini senza errori. |
| **TST-003**          | **REQ-005**                     | Simulare `--upgrade` con mocking della chiamata a `uv` e verificare che non venga eseguito un upgrade reale. |
| **TST-004**          | **REQ-006**                     | Simulare `--uninstall` con mocking della chiamata a `uv` e verificare che non venga eseguita una disinstallazione reale. |
| **TST-005**          | **REQ-007**, **REQ-008**, **REQ-009**, **REQ-010** | Simulare risposte API per versione uguale, maggiore e errore di rete; verificare comportamento e messaggio atteso. |
| **TST-006**          | **DES-007**                     | Eseguire `pytest` e verificare che i test coprano `--version`, `--help`, `--upgrade` e `--uninstall` con mocking di `uv` e il controllo versione online con risposte simulate. |

## 5. Cronologia delle revisioni

| Data | Versione | Motivo e descrizione del cambiamento |
|------|----------|--------------------------------------|
| 2026-01-12 | 0.2 | Aggiunti requisiti per i test unitari della CLI. |
| 2026-01-12 | 0.1 | Bozza iniziale dei requisiti. |
