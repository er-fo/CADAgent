# Add-In Mechanical Skill Runtime Greenlight - 2026-05-10

Status: integration-ready for local/static validation; live Fusion smoke still required.

## Add-In Mechanical Feature Operation Readiness

- Declares `mechanical_feature_operations_v1` only with handlers present.
- Handles `create_countersink_hole`, `apply_draft`, `mirror_entities`, `combine_bodies`, `split_body`, `split_face`, `add_sketch_dimension`, and `add_sketch_constraint`.
- Keeps `create_sweep` on backend `execute_code`; no structured add-in sweep handler is required for this integration.
- Keeps macOS and Windows add-in sources byte-for-byte identical after implementation.
- Fails closed for unsupported boolean semantics and unresolved references instead of approximating unsupported Fusion operations.

## Validation

Validation run on 2026-05-11:

```bash
python3 -m unittest -q tests.test_mechanical_feature_operations_static
python3 -m compileall -q mac/CADAgent win/CADAgent
diff -qr -x __pycache__ mac/CADAgent win/CADAgent
```

Result:

- Static operation coverage: `9 tests OK`
- Python compile: pass
- mac/win parity: pass
- gpt-5.5 high re-audit: no scoped defects found

## Remaining Gate

Live Fusion smoke has not run in this terminal session. Run the backend smoke matrix before claiming real Fusion E2E readiness or releasing the add-in.
