# Tests

The first focused tests cover deterministic validation-panel selection and
panel-based embedding metadata preselection.

For this first scaffold, use:

```bash
PYTHONPATH=src python3 -m compileall src
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m evo2_go.cli --help
PYTHONPATH=src python3 -m evo2_go.cli go-term-counts --help
```

The next slices should add tiny TSV/FASTA fixtures for representative CDS
mapping, GO term counting, true-path expansion, adjacency construction,
training resume behavior, metrics, and publication figures.
