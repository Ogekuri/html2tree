---
title: "Requisiti html2tree"
description: Specifica dei requisiti software
version: "1.2"
date: "2026-01-15"
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
**Versione**: 1.2
**Autore**: Codex CLI
**Data**: 2026-01-15

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
html2tree e una applicazione CLI Python che converte una directory di esportazione HTML (contenente `document.html`, `toc.html` e `assets/`) in un set di artefatti Markdown e asset (immagini e tabelle), generando un manifest JSON e una pipeline di post-processing opzionale per normalizzare il Markdown, arricchire il manifest, rimuovere immagini piccole, estrarre equazioni (Pix2Tex) e produrre annotazioni (Gemini) in formato Markdown adatto a flussi RAG. La CLI include anche comandi informativi e operativi (help, versione, upgrade, uninstall) con controllo online della versione e supporto alla distribuzione tramite uv/uvx.

## 2. Requisiti di progetto

### 2.1 Funzioni di progetto
- **PRJ-001**: Il progetto deve fornire una applicazione CLI denominata `html2tree`.
- **PRJ-002**: Il progetto deve fornire funzionalita di gestione versione, help, upgrade, uninstall e controllo versione online.
- **PRJ-003**: Il progetto deve includere script di automazione locale e pipeline di rilascio GitHub.
- **PRJ-004**: Il progetto deve convertire una directory HTML in un output composto da Markdown, TOC Markdown, manifest JSON, immagini e tabelle esportate.
- **PRJ-005**: Il progetto deve offrire una pipeline di post-processing (attivabile da CLI) che normalizza il Markdown e arricchisce gli artefatti con rimozione immagini piccole, estrazione equazioni e annotazioni.

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
- **CTN-009**: Il progetto deve includere `markdownify` tra le dipendenze runtime.
- **CTN-010**: Il progetto deve includere `beautifulsoup4` tra le dipendenze runtime.
- **CTN-011**: Il progetto deve includere `pillow`, `matplotlib`, `pylatexenc` tra le dipendenze runtime.
- **CTN-012**: Le funzionalità opzionali di post-processing devono gestire dipendenze runtime opzionali (per esempio Pix2Tex e SDK Gemini) terminando con errore esplicito quando una fase è abilitata ma la dipendenza non è disponibile.
- **CTN-013**: Il progetto deve includere `cairosvg` tra le dipendenze runtime per abilitare la conversione di immagini SVG in PNG durante la fase di annotazione.

## 3. Requisiti

### 3.1 Progettazione e implementazione
- **DES-001**: I sorgenti della CLI devono risiedere in `src/html2tree/cli.py`.
- **DES-002**: I punti di ingresso devono essere implementati in `src/html2tree/__init__.py` e `src/html2tree/__main__.py`.
- **DES-003**: La versione deve essere definita in `src/html2tree/version.py` nel formato `<major>.<minor>.<patch>`.
- **DES-004**: Il workflow di rilascio deve essere definito in `.github/workflows/release-uvx.yml` e includere i passi: Checkout, Set up Python, Set up uv, Install build dependencies, Build distributions, Attest build provenance, Create GitHub Release and upload assets.
- **DES-005**: Devono essere presenti gli script di root `venv.sh`, `tests.sh` e `html2tree.sh`.
- **DES-006**: La struttura del progetto deve includere almeno i seguenti file e cartelle (escludendo directory con prefisso punto):
```
html2tree/
├─ LICENSE
├─ README.md
├─ docs/
│  ├─ requirements.md
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
      ├─ core.py
      ├─ latex.py
      └─ version.py
```
- **DES-007**: Devono essere presenti unit test per la CLI che coprano `--version`, `--help`, `--upgrade` (con mock di `uv`), `--uninstall` (con mock di `uv`) e il controllo versione online.
- **DES-008**: La pipeline di conversione deve essere implementata in `src/html2tree/core.py` ed esposta tramite una funzione `run_processing_pipeline`.
- **DES-009**: La pipeline di post-processing deve essere implementata in `src/html2tree/core.py` ed esposta tramite una funzione `run_post_processing_pipeline`.
- **DES-010**: Il framework di validazione LaTeX deve essere implementato in `src/html2tree/latex.py` e utilizzato dalla fase Pix2Tex durante il post-processing.
- **DES-011**: La pipeline di conversione PDF (`pdf2tree`) deve, in modalita di sola processing (assenza di `--post-processing` e `--post-processing-only`), eseguire esclusivamente la fase di processing producendo `document.md` con immagini e tabelle già posizionate, e una copia di backup `document.processing.md` identica al risultato di processing.

### 3.2 Funzioni
- **REQ-001**: L'opzione `--help` deve stampare la schermata di usage e terminare l'esecuzione.
- **REQ-002**: In assenza di parametri, il programma deve stampare la schermata di usage e terminare l'esecuzione.
- **REQ-003**: La schermata di usage deve indicare il nome `html2tree` e la versione corrente.
- **REQ-004**: L'opzione `--version` e il suo alias `--ver` devono stampare la versione nel formato `"<major>.<minor>.<patch>"` e terminare l'esecuzione.
- **REQ-005**: L'opzione `--upgrade` deve eseguire `uv tool install usereq --force --from git+https://github.com/Ogekuri/html2tree.git`.
- **REQ-006**: L'opzione `--uninstall` deve eseguire `uv tool uninstall html2tree`.
- **REQ-007**: Dopo la validazione dei parametri e prima di ogni altra operazione, il programma deve verificare online la presenza di una nuova versione.
- **REQ-008**: La verifica online deve chiamare `https://api.github.com/repos/Ogekuri/html2tree/releases/latest` con timeout di 1 secondo e determinare l'ultima versione dalla risposta JSON.
- **REQ-009**: Se la chiamata API fallisce o la versione non e determinabile, il programma deve considerare la versione corrente come ultima e non emettere avvisi.
- **REQ-010**: Se la versione disponibile e maggiore di quella corrente, il programma deve stampare il messaggio: `A new version of html2tree is available: current X.Y.Z, latest A.B.C. To upgrade, run: html2tree --upgrade` con i valori correnti.
- **REQ-011**: Lo script `venv.sh` deve verificare se esiste `.venv`, rimuoverla se presente, creare un nuovo ambiente e installare tutte le dipendenze di `requirements.txt`.
- **REQ-012**: Lo script `tests.sh` deve eseguire tutti gli unit test implementati.
- **REQ-013**: Lo script `html2tree.sh` deve eseguire il programma passando tutti gli argomenti ricevuti dalla riga di comando.
- **REQ-014**: L'opzione `--from-dir` deve accettare una directory sorgente contenente `document.html`, `toc.html` e `assets/`, mentre `--to-dir` deve impostare la directory di output; in assenza di `--from-dir`/`--to-dir` la CLI deve terminare con errore (eccetto i comandi `--help`, `--version|--ver`, `--upgrade`, `--uninstall`, `--write-prompts`).
- **REQ-015**: La CLI deve convertire `document.html` in Markdown tramite `markdownify`, rimuovendo i tag `<script>` e `<style>`, e deve salvare nell'output: `<document>.md`, `<document>.md.processing.md` e `<document>.toc`.
- **REQ-016**: La conversione deve creare l'output con le sottocartelle `assets/` e `tables/`, e deve generare un manifest JSON `<document>.json` con almeno le chiavi `source_html`, `markdown`, `tables`, `images` e percorsi relativi agli artefatti.
- **REQ-017**: La conversione deve copiare gli asset dalla sorgente in `assets/` dell'output e deve aggiornare i link immagine nel Markdown affinché puntino a `assets/<file>` (gestendo correttamente percorsi relativi, prefissi `file://`, query/fragment e link multilinea prodotti dalla conversione).
- **REQ-018**: Per ogni tabella HTML identificata la conversione deve esportare in `tables/` un file `.md` e un file `.csv`; nel file `<document>.md` deve essere inserito, nella posizione originale del tag `<table>`, il contenuto della tabella in formato Markdown e i riferimenti ai file esportati (link `[Markdown](tables/<nome>.md)` e `[CSV](tables/<nome>.csv)`).
- **REQ-019**: Le opzioni `--post-processing` e `--post-processing-only` devono essere supportate e mutualmente esclusive: `--post-processing` deve eseguire conversione + post-processing, mentre `--post-processing-only` deve saltare la conversione e operare su un output esistente.
- **REQ-020**: In modalità `--post-processing-only` la CLI deve verificare la presenza del file Markdown e del backup `.processing.md` nella cartella di output, ripristinare il Markdown dal backup prima di avviare il post-processing e rigenerare `.toc` e manifest senza richiedere un manifest preesistente.
- **REQ-021**: Il post-processing deve normalizzare il Markdown ripristinato eseguendo: `normalize_markdown_format` (conversione dei tag HTML `<br>` in newline), `remove_markdown_index` (rimozione del contenuto iniziale fino alla prima voce TOC preservando eventuali marker di pagina), `normalize_markdown_headings` (riallineamento livelli heading alla TOC), `clean_markdown_headings` (degradazione delle intestazioni non presenti in TOC in testo maiuscolo in grassetto) e, se non disabilitato, inserimento di una TOC Markdown derivata da `toc.html`; al termine deve rigenerare e salvare su disco il file `<document>.toc`.
- **REQ-022**: Il post-processing deve validare la TOC estratta dal file `<document>.toc` confrontandola con `toc.html` su conteggio e titoli normalizzati (ignorando le differenze di livello) e, in caso di mismatch, deve segnalare l'errore senza interrompere le fasi successive; in modalità test la validazione può essere bypassata.
- **REQ-023**: Il post-processing deve includere una fase `remove-small-images` abilitata di default (disattivabile con `--disable-remove-small-images`) che misura le dimensioni effettive delle immagini elencate nel manifest e, quando entrambe le dimensioni risultano inferiori alle soglie configurate, rimuove le voci corrispondenti dal manifest e i riferimenti dal Markdown (inclusi blocchi equazione/annotazione), senza cancellare i file su disco.
- **REQ-024**: La CLI deve offrire i parametri `--min-size-x` e `--min-size-y` (interi positivi, default 100 px) applicati alla fase `remove-small-images`; l’immagine deve essere rimossa solo se entrambe le dimensioni risultano inferiori alle soglie impostate.
- **REQ-025**: Quando il post-processing è attivo e viene fornito `--enable-pic2tex` (senza `--disable-pic2tex`), la fase Pix2Tex deve eseguire l'OCR LaTeX sulle immagini del manifest e, se l'output supera la soglia `--equation-min-len` ed è validato dal framework LaTeX, deve aggiornare `images[*].type` a `equation`, aggiungere `images[*].equation` e inserire la formula nel Markdown immediatamente prima dell'immagine.
- **REQ-026**: Il parametro `--equation-min-len` deve essere un intero > 0 con default 5; se l'output Pix2Tex ha lunghezza inferiore, l'immagine deve restare `image` senza modifiche al Markdown.
- **REQ-027**: Il framework di validazione LaTeX deve validare una formula solo se supera in sequenza: bilanciamento delimitatori (inclusi `\\left/\\right`), parsing con `pylatexenc`, controllo di compatibilità MathJax e tentativo di rendering con `matplotlib` (backend Agg); in caso di fallimento la formula deve essere considerata non valida.
- **REQ-028**: L'inserimento delle formule LaTeX nel Markdown deve racchiudere la formula tra le righe `**----- Start of equation: <nome immagine> -----**` e `**----- End of equation: <nome immagine> -----**`, usando il nome del file immagine associato e il blocco formula in formato `$$...$$`.
- **REQ-029**: Il post-processing deve includere una fase opzionale di annotazione eseguita dopo Pix2Tex: l'annotazione delle immagini deve essere abilitata per impostazione predefinita e disattivabile con `--disable-annotate-images`, mentre l'annotazione delle equazioni deve attivarsi esplicitamente con `--enable-annotate-equations`; quando l'annotazione è attiva, la CLI deve richiedere una chiave Gemini (`--gemini-api-key` o env `GEMINI_API_KEY`) e terminare con errore in assenza.
- **REQ-030**: Le annotazioni devono essere inserite nel Markdown subito dopo il link immagine e racchiuse tra le righe `**----- Start of annotation: <nome immagine> -----**` e `**----- End of annotation: <nome immagine> -----**`; il manifest deve includere il campo `annotation` per ogni immagine/equazione annotata.
- **REQ-031**: La CLI deve supportare `--prompts <file>` (caricamento) e `--write-prompts <file>` (scrittura e uscita) per gestire prompt Gemini in JSON con chiavi obbligatorie `prompt_equation`, `prompt_non_equation`, `prompt_uncertain`; in caso di file mancante o invalido la CLI deve terminare con errore senza avviare conversione o post-processing.
- **REQ-032**: La selezione del prompt per l'annotazione deve seguire queste regole: se la fase Pix2Tex è stata eseguita, le immagini non classificate come equazioni usano `prompt_non_equation` e le equazioni (quando abilitate) usano `prompt_equation`; se Pix2Tex non è stata eseguita, tutte le annotazioni devono usare `prompt_uncertain`.
- **REQ-033**: In modalità test (attivabile tramite env `HTML2TREE_TEST_MODE` o esecuzione sotto `pytest`) la fase Pix2Tex deve usare una formula deterministica (sovrascrivibile via env `HTML2TREE_TEST_PIX2TEX_FORMULA`) e la fase di annotazione deve usare testi deterministici (sovrascrivibili via env `HTML2TREE_TEST_GEMINI_IMAGE_ANNOTATION` e `HTML2TREE_TEST_GEMINI_EQUATION_ANNOTATION`) senza effettuare chiamate di rete o inizializzare modelli reali.
- **REQ-034**: Il post-processing deve includere una fase di cleanup abilitata di default (disattivabile con `--disable-cleanup`) che rimuove tutte le righe di marker `--- start/end of page.page_number=<num> ---` dal Markdown prima dell’arricchimento del manifest, preservando eventuali TOC inserite.
- **REQ-035**: Il manifest JSON deve includere per ogni nodo della TOC e per ogni voce `tables`/`images` un ID univoco e stabile e le relazioni `parent_id`, `prev_id`, `next_id`; i nodi TOC devono inoltre includere gli array `tables` e `images` con gli ID delle entità figlie.
- **REQ-036**: Il manifest JSON deve annotare ogni nodo `markdown.toc_tree`, ogni voce `tables` e ogni voce `images` con i campi `start_line`, `end_line`, `start_char`, `end_char` che delimitano il blocco Markdown corrispondente; per le tabelle l’intervallo deve includere i link ai file esportati e il blocco tabellare immediatamente precedente, per le immagini deve includere eventuali blocchi equazione/annotazione e il link all’immagine, e per i nodi TOC deve seguire l’ordine di lettura della TOC (i nodi padre non includono il contenuto dei figli).
- **REQ-037**: In modalità `--verbose` il post-processing deve riportare l’avanzamento e l’esito delle fasi `remove-small-images`, Pix2Tex e annotazione per ogni immagine; con `--debug` deve includere dettagli tecnici (output Pix2Tex grezzo, contenuto annotazioni, eccezioni).
- **REQ-038**: Se una fase opzionale è attiva ma la relativa dipendenza runtime non è disponibile (per esempio Pix2Tex o SDK Gemini), la CLI deve terminare con errore esplicito prima o durante l’esecuzione della fase.
- **REQ-039**: La CLI deve terminare con errore se `--to-dir` esiste ed è non vuota al momento dell’avvio (eccetto in modalità `--post-processing-only`).
- **REQ-040**: La TOC Markdown generata (`<document>.toc`) deve includere, accanto alla sezione di appartenenza, i collegamenti ai file `.md` e `.csv` di ciascuna tabella esportata (presentati come allegati visibili nella TOC).
- **REQ-041**: In modalità processing della CLI, il file `document.md` deve contenere tutti gli heading h1..h6 del sorgente convertiti in heading Markdown (`#`..`######`), tutte le immagini con link `![](images/<nome>)` e tutte le tabelle in linea nella posizione originale con contenuto Markdown, più i link `[Markdown](tables/<nome>.md)` e `[CSV](tables/<nome>.csv)` ai file esportati.
- **REQ-042**: Durante la sola fase di processing della CLI, devono essere creati i file delle tabelle sotto `tables/` (`.md` e `.csv`) e `document.processing.md` come copia identica del `document.md` di processing; non devono essere generati né manifest JSON né file `.toc`.
- **REQ-043**: Durante la fase di processing la TOC del documento HTML `toc.html` viene copiata nella directory di destinazione per poter essere riutilizzata nelle fasi di post-processing.
- **REQ-044**: La pipeline di post-processing deve iniziare verificando l'esistenza del file di backup .processing.md nella directory di output.
- **REQ-045**: La pipeline deve ripristinare il file Markdown principale (.md) copiando il contenuto del backup .processing.md per garantire uno stato pulito prima di ogni esecuzione.
- **REQ-046**: Durante la fase di post-processing la TOC del documento HTML viene ricaricata dal documento `toc.html` presente nella cartella di output.
- **REQ-047**: La pipeline deve eseguire la normalizzazione del formato Markdown convertendo tutti i tag HTML <br>, <br/> e <br /> in caratteri newline standard.
- **REQ-048**: La pipeline deve riallineare i livelli delle intestazioni Markdown ( #, ##, ecc.) affinché corrispondano esattamente alla gerarchia definita nella TOC del HTML `toc.html`.
- **REQ-049**: Le intestazioni Markdown non presenti nella TOC del HTML `toc.html` devono essere "degradate" convertendole in testo semplice in grassetto maiuscolo, rimuovendo il marcatore di intestazione.
- **REQ-050**: Se non disabilitato tramite flag --disable-toc, la pipeline deve inserire all'inizio del documento una sezione "TOC Markdown" con intestazione ** TOC ** e link interni alle sezioni.
- **REQ-051**: La pipeline deve rigenerare il file .toc (formato Markdown) nella directory di output basandosi sulle intestazioni normalizzate presenti nel file .md e i riferimenti a tabelle e immagini.
- **REQ-052**: La pipeline deve validare la coerenza tra la TOC estratta dal `toc.html` e quella generata dal Markdown (.toc), confrontando numero di voci e titoli normalizzati.
- **REQ-053**: In caso di mancata corrispondenza della TOC, la pipeline deve registrare un errore ma proseguire l'esecuzione (salvo diversa configurazione in modalità test ristretta).
- **REQ-054**: La pipeline deve costruire una versione preliminare del manifest JSON consolidando le informazioni da HTML `toc.html`, Markdown, file di immagini e tabelle esportate.
- **REQ-055**: Il manifest deve includere per ogni asset (immagini, tabelle) il contesto gerarchico (context_path) derivato dalla posizione nella struttura TOC.
- **REQ-056**: Se non disabilitato tramite --disable-remove-small-images, la pipeline deve identificare le immagini con dimensioni (x, y) inferiori alle soglie configurate.
- **REQ-057**: Le immagini identificate come "troppo piccole" devono essere rimosse dal manifest e tutti i loro riferimenti (link, annotazioni, blocchi equazione) devono essere eliminati dal testo Markdown.
- **REQ-058**: I file fisici delle immagini rimosse devono essere mantenuti su disco per preservare l'integrità dell'output originale.
- **REQ-059**: Se abilitato tramite --enable-pic2tex la pipeline deve eseguire il modello Pix2Tex su tutte le immagini presenti nel manifest.
- **REQ-060**: Le immagini riconosciute come equazioni valide (superamento validazione LaTeX e lunghezza minima) devono essere marcate come type: "equation" nel manifest.
- **REQ-061**: Per ogni equazione validata, la pipeline deve inserire nel Markdown il blocco LaTeX corrispondente, racchiuso tra i marcatori `----- Start of equation -----` e `----- End of equation -----`, posizionato immediatamente prima dell'immagine.
- **REQ-062**: Se abilitata l'annotazione (per immagini o equazioni), la pipeline deve interrogare l'API Gemini per generare descrizioni testuali o trascrizioni.
- **REQ-063**: Le annotazioni generate devono essere inserite nel manifest nel campo annotation e nel Markdown come blocchi di testo successivi all'immagine, racchiusi tra `----- Start of annotation -----` e `----- End of annotation -----`.
- **REQ-064**: La pipeline deve selezionare il prompt corretto (prompt_equation, prompt_non_equation, prompt_uncertain) in base alla classificazione precedente (Pix2Tex) o all'incertezza se Pix2Tex non è stato eseguito.
- **REQ-065**: Se non disabilitato tramite --disable-cleanup, la pipeline deve rimuovere dal Markdown finale tutti i marcatori tecnici di inizio/fine pagina (--- start/end of page... ---).
- **REQ-066**: La pipeline deve assegnare identificativi univoci (id) a tutti i nodi della TOC, tabelle e immagini nel manifest.
- **REQ-067**: Il manifest deve essere arricchito con le relazioni di navigazione parent_id, prev_id, next_id per ogni entità, riflettendo l'ordine di lettura del documento.
- **REQ-068**: I nodi TOC devono essere popolati con liste degli ID delle tabelle (tables) e immagini (images) contenute nelle rispettive sezioni.
- **REQ-069**: La pipeline deve calcolare e inserire nel manifest gli intervalli precisi di righe (start_line, end_line) e caratteri (start_char, end_char) che localizzano ogni entità nel file Markdown finale.
- **REQ-070**: Al termine di tutte le elaborazioni, la pipeline deve salvare su disco la versione definitiva del file Markdown (sovrascrivendo l'originale) e il file manifest JSON completo.
- **REQ-071**: In caso di mismatch della TOC rilevato nella fase 3, la pipeline deve terminare restituendo un codice di uscita specifico (EXIT_POSTPROC_DEP).
- **REQ-072**: Durante la fase `run_annotation_phase`, se un'immagine ha MIME `image/svg+xml` o `image/svg`, la pipeline deve generare un file PNG con lo stesso nome base e suffisso `.png` nello stesso percorso dell'asset originale e deve inviare esclusivamente quel PNG per la richiesta di annotazione; le immagini SVG non devono mai essere inviate direttamente in annotazione perché il MIME non è supportato.
- **REQ-073**: Durante il post-processing, i campi `tables` e `images` del manifest JSON devono essere ricostruiti esclusivamente a partire dalle referenze presenti nel file Markdown ripristinato da `<name>.md.processing.md`, mantenendo l'ordine di apparizione nel testo e ignorando asset presenti su disco ma non citati nel Markdown ripristinato.


## 4. Requisiti di test

| ID Requisito di Test | Requisito collegato / Contesto | Procedura di test (chiara e riproducibile) |
|----------------------|---------------------------------|-------------------------------------------|
| **TST-001**          | **REQ-004**                     | Eseguire la CLI con `--version` e `--ver`. Verificare che stampi la versione nel formato richiesto e termini senza errori. |
| **TST-002**          | **REQ-001**, **REQ-002**, **REQ-003** | Eseguire la CLI con `--help` e senza parametri. Verificare che stampi la schermata di usage con nome e versione e termini senza errori. |
| **TST-003**          | **REQ-005**                     | Simulare `--upgrade` con mocking della chiamata a `uv` e verificare che non venga eseguito un upgrade reale. |
| **TST-004**          | **REQ-006**                     | Simulare `--uninstall` con mocking della chiamata a `uv` e verificare che non venga eseguita una disinstallazione reale. |
| **TST-005**          | **REQ-007**, **REQ-008**, **REQ-009**, **REQ-010** | Simulare risposte API per versione uguale, maggiore e errore di rete; verificare comportamento e messaggio atteso. |
| **TST-006**          | **DES-007**                     | Eseguire `pytest` e verificare che i test coprano `--version`, `--help`, `--upgrade` e `--uninstall` con mocking di `uv` e il controllo versione online con risposte simulate. |
| **TST-007**          | **REQ-014**, **REQ-015**, **REQ-016**, **REQ-017**, **REQ-018** | Convertire `html_sample/` in una directory temporanea e verificare la creazione di Markdown, manifest JSON, `assets/` e `tables/` con file `.md` e `.csv`. |
| **TST-008**          | **REQ-019**, **REQ-020**         | Eseguire `--post-processing-only` su un output esistente e verificare l'aggiornamento di Markdown e manifest. |
| **TST-009**          | **REQ-031**, **REQ-032**         | Eseguire `--write-prompts` e `--prompts`, verificando che il file contenga i tre prompt richiesti e che la CLI li accetti e li usi secondo le regole di selezione. |
| **TST-010**          | **REQ-014**, **REQ-015**, **REQ-016**, **REQ-017**, **REQ-018** | Eseguire la conversione di `html_sample/` in `temp/html_sample/` con i flag `--verbose` e `--debug` attivati. Verificare che la conversione completi con successo e che tutti gli artefatti siano creati. Pulire la directory temporanea al termine del test. |
| **TST-011**          | **REQ-023**, **REQ-024**         | Eseguire `--post-processing-only` su un output contenente almeno una immagine sotto soglia e verificare che venga rimossa da Markdown/manifest ma non cancellata su disco. |
| **TST-012**          | **REQ-025**, **REQ-026**, **REQ-028**, **REQ-033** | Impostare `HTML2TREE_TEST_MODE=1`, eseguire `--post-processing-only --enable-pic2tex` e verificare che siano inseriti i blocchi `Start/End of equation` e che `images[*].type` diventi `equation` con `equation` valorizzato. |
| **TST-013**          | **REQ-029**, **REQ-030**, **REQ-031**, **REQ-033** | Impostare `HTML2TREE_TEST_MODE=1`, eseguire `--post-processing-only` con `--gemini-api-key` fittizia e verificare che vengano inseriti i blocchi `Start/End of annotation` e che `images[*].annotation` sia valorizzato. |
| **TST-014**          | **REQ-018**, **REQ-041**         | Convertire una sorgente HTML contenente almeno una tabella; verificare che in `<document>.md` la tabella appaia nella posizione originale (senza placeholder) con contenuto Markdown e link `[Markdown]/[CSV]`, e che tali link siano presenti in `<document>.toc`. |
| **TST-015**          | **REQ-072**, **CTN-013**         | Eseguire il post-processing su un output contenente un'immagine SVG; verificare che venga creato un PNG con lo stesso nome base nella stessa cartella asset, che la richiesta di annotazione utilizzi MIME `image/png` e che l'immagine SVG non venga mai inviata direttamente. |
| **TST-016**          | Controllo examples.sh (RUN_EXAMPLES_TEST) | Impostare `RUN_EXAMPLES_TEST=1` ed eseguire la suite di test. Verificare che il test esegua `examples.sh`, legga `examples.log` e fallisca se sono presenti errori nel log. |

## 5. Cronologia delle revisioni

| Data | Versione | Motivo e descrizione del cambiamento |
|------|----------|--------------------------------------|
| 2026-01-15 | 1.2 | Aggiunto test opzionale RUN_EXAMPLES_TEST per eseguire examples.sh e validare examples.log. |
| 2026-01-13 | 1.1 | Allineata generazione manifest post-processing: tabelle e immagini estratte solo dal Markdown ripristinato (.processing.md) ignorando asset non referenziati. |
| 2026-01-13 | 1.0 | Aggiunta gestione SVG→PNG per annotazione (dipendenza `cairosvg`, requisito funzionale e test dedicato). |
| 2026-01-13 | 0.9 | Allineati requisiti per processing PDF: Markdown con heading/immagini/tabelle, backup `.processing.md`, manifest/.toc solo in post-processing. |
| 2026-01-13 | 0.6 | Riscritti i requisiti di conversione e post-processing per essere completi e autonomi; dettaglio pipeline (normalizzazione, remove-small-images, Pix2Tex, annotazioni, arricchimento manifest) e rimozione di riferimenti esterni. |
| 2026-01-13 | 0.8 | Correzione posizionamento tabelle in Markdown e visibilità allegati Markdown/CSV nella TOC. |
| 2026-01-13 | 0.5 | Aggiunto TST-010: test di conversione con flag --verbose e --debug. |
| 2026-01-13 | 0.4 | Aggiornato REQ-018: il file document.md deve contenere sia il contenuto della tabella in formato Markdown sia i riferimenti ai file esportati. |
| 2026-01-12 | 0.3 | Aggiunti requisiti per conversione HTML, post-processing e dipendenze runtime. |
| 2026-01-12 | 0.2 | Aggiunti requisiti per i test unitari della CLI. |
| 2026-01-12 | 0.1 | Bozza iniziale dei requisiti. |
