# Mermaid — Entity Matching Pipeline

Mermaid source for the Amazon ML Challenge 2026 entity-resolution pipeline.
Every file is validated — it renders with `mermaid-cli` with zero errors.

## Files

| File | Renders at | Use |
|---|---|---|
| `pipeline-overview.mmd` | 2832 × 5550 | **Start here.** Whole pipeline, compact labels, both feedback loops. |
| `s1-candidate-generation.mmd` | 1542 × 1914 | Data → normalization → blocking → candidate_pairs |
| `s2-recall-check.mmd` | 1664 × 2588 | Ground truth check #1 + improvement loop |
| `s3-model.mmd` | 1018 × 2168 | Pairwise features → model → threshold |
| `s4-evaluation.mmd` | 1036 × 1814 | Ground truth check #2 + retrain loop |
| `s5-final-test.mmd` | 1136 × 2344 | Final test, ground truth hidden |
| `pipeline-full.mmd` | 3062 × 11512 | Everything in one graph. **Very tall — see caveat.** |

`.png` and `.svg` are generated next to each `.mmd`. Prefer the SVG — it scales.

## Rendering

```bash
npx mmdc -i pipeline-overview.mmd -o pipeline-overview.svg -c mermaid-config.json -C mermaid.css -b white
```

`mermaid-config.json` sets `wrappingWidth: 520` (without it, nodes are narrow
and labels wrap into unreadably tall boxes). `mermaid.css` left-aligns the
`<code>` evidence blocks so the TSV columns line up.

To render offline with an existing Chromium, point puppeteer at it:

```bash
echo '{"executablePath":"/path/to/chrome","args":["--no-sandbox"]}' > pconf.json
```

## Caveat: Mermaid cannot reproduce the Excalidraw layout

The Excalidraw version uses a deliberate two-column layout. Mermaid's dagre
engine assigns ranks automatically, and the back-edges (improvement loop,
retrain loop) force sections into a single tall column — `pipeline-full.mmd`
comes out 11,512 px tall and reorders the sections.

`flowchart LR` does not fix it; it collapses instead. `direction TB` inside a
subgraph is **ignored** whenever edges cross the subgraph boundary, which is
exactly what the feedback loops do.

**So: use `pipeline-overview.mmd` plus the per-section files.** They are the
readable artifacts. `pipeline-full.mmd` is kept for completeness only. For the
publication-quality single-page diagram, use the Excalidraw file
(`../entity-matching-pipeline.excalidraw`).

## Syntax notes

Two things that silently break Mermaid and are worth remembering:

- **Labels must be on one physical line.** A literal newline inside a node
  label is a parse error. Use `<br/>`.
- **HTML numeric entities do not work.** `&#35;` renders as the literal text
  `&#35;`. Mermaid's own escape is `#35;` (no ampersand) — but inside a quoted
  label, plain `#` and `(` `)` are fine, which is what these files use.
  Named entities like `&amp;`, `&lt;`, `&gt;`, `&nbsp;` do work.

## Colour semantics

blue = data flow · green = correct/validated · orange = improvement loop ·
red = error/missing · yellow = decision · purple = ML model · dark = data sample
