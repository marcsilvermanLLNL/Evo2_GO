# Migration Notes

The remodel should proceed in small slices. Each slice should leave behind a
usable command and documentation, not only copied source files.

## First Slice

Moved into `src/evo2_go`:

- `database/records.py` from `src/python/object_classes.py`
- `database/map_gene_cds.py`
- `database/map_gene_cds_by_tax.py`
- `database/go_term_counts.py`
- `database/true_path_labels.py`
- `ontology/go_adjacency.py`
- `embeddings/length_filter.py`
- `embeddings/run_embeddings.py`
- `embeddings/validate_embeddings.py`
- `model/dataset.py`

Added:

- `evo2_go.cli` top-level command dispatcher.
- `build_database.py` root entry point and `evo2_go.workflows.build_database`.
- Package metadata in `pyproject.toml`.
- Database workflow documentation.
- Shared embedding-root documentation and build config fields.

## Compatibility Rule

Copied scripts should keep their existing command-line flags unless there is a
clear reason to change them. New wrappers may improve ergonomics, but old flags
should remain accepted during migration.

## Next Slices

1. Add smoke-test fixtures for the migrated database commands.
2. Expand `build_database.py` to cover raw Gene2GO processing and RefSeq
   download wrappers as those scripts migrate.
3. Add an embedding queue wrapper around the shared-root extractor for Flux
   wall-time-limited jobs.
4. Move model code with architecture config objects that support variable
   sequence layers, GO GCN layers, dimensions, dropout, logit bias, and loss
   settings without editing source files.
5. Add training run specs and run-matrix generation.
