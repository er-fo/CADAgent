"""Tests for sequential face naming in entity_store."""
import sys
import asyncio
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from backend.entity_store import EntityStore, EntityRef


def test_sequential_face_naming():
    """Faces are assigned face_0, face_1, face_2, ... sequentially."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [
            {"entity_token": "bt0", "name": "Box"},
        ])
        refs = await store.register_entities("face", [
            {"entity_token": "f1", "body": "Box", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
            {"entity_token": "f2", "body": "Box", "normal": [0, 0, -1], "centroid": [0, 0, 0], "surface_type": "planar"},
            {"entity_token": "f3", "body": "Box", "normal": [1, 0, 0], "centroid": [5, 0, 5], "surface_type": "planar"},
        ])
        ref_ids = [r.ref_id for r in refs]
        assert ref_ids == ["face_0", "face_1", "face_2"], f"Got {ref_ids}"

    asyncio.run(_run())


def test_mixed_surface_types():
    """Sequential IDs are assigned regardless of surface type (planar, cylindrical, etc.)."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [
            {"entity_token": "bt0", "name": "Part"},
        ])
        refs = await store.register_entities("face", [
            {"entity_token": "f1", "body": "Part", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
            {"entity_token": "f2", "body": "Part", "normal": [1, 0, 0], "centroid": [5, 0, 5], "surface_type": "cylindrical"},
            {"entity_token": "f3", "body": "Part", "normal": [0, 1, 0], "centroid": [0, 5, 5], "surface_type": "toroidal"},
        ])
        ref_ids = [r.ref_id for r in refs]
        assert ref_ids == ["face_0", "face_1", "face_2"], f"Got {ref_ids}"

    asyncio.run(_run())


def test_multi_body_sequential():
    """Multi-body designs get a single global sequential counter (no body prefix)."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [
            {"entity_token": "bt0", "name": "Box"},
            {"entity_token": "bt1", "name": "Cyl"},
        ])
        refs = await store.register_entities("face", [
            {"entity_token": "f1", "body": "Box", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
            {"entity_token": "f2", "body": "Box", "normal": [0, 0, -1], "centroid": [0, 0, 0], "surface_type": "planar"},
            {"entity_token": "f3", "body": "Cyl", "normal": [0, 0, 1], "centroid": [5, 5, 20], "surface_type": "planar"},
            {"entity_token": "f4", "body": "Cyl", "normal": [0, 0, -1], "centroid": [5, 5, 5], "surface_type": "cylindrical"},
        ])
        ref_ids = [r.ref_id for r in refs]
        assert ref_ids == ["face_0", "face_1", "face_2", "face_3"], f"Got {ref_ids}"

    asyncio.run(_run())


def test_spatial_info_preserved():
    """Metadata (normal, centroid, surface_type) is preserved in EntityRef."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [
            {"entity_token": "bt0", "name": "Box"},
        ])
        refs = await store.register_entities("face", [
            {"entity_token": "f1", "body": "Box", "normal": [0, 0, 1], "centroid": [1.0, 2.0, 3.0], "surface_type": "planar"},
        ])
        entry = refs[0]
        assert entry.ref_id == "face_0"
        assert entry.normal == (0.0, 0.0, 1.0)
        assert entry.centroid == (1.0, 2.0, 3.0)
        assert entry.metadata.get("surface_type") == "planar"

    asyncio.run(_run())


def test_dict_centroid_handling():
    """EntityRef.centroid/normal should handle dict format from Fusion."""
    entry = EntityRef(ref_id="face_0", token="t1", kind="face",
                      metadata={"centroid": {"x": 1.0, "y": 2.0, "z": 3.0}})
    assert entry.centroid == (1.0, 2.0, 3.0)

    entry2 = EntityRef(ref_id="face_1", token="t2", kind="face",
                       metadata={"normal": {"x": 0, "y": 0, "z": 1}})
    assert entry2.normal == (0.0, 0.0, 1.0)


def test_cache_reuse():
    """After soft_clear, cached face refs are reused if fingerprint matches."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        r1 = await store.register_entities("face", [
            {"entity_token": "f1", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
            {"entity_token": "f2", "body": "B", "normal": [0, 0, -1], "centroid": [0, 0, 0], "surface_type": "planar"},
        ])
        ref_map_1 = {r.token: r.ref_id for r in r1}

        store.soft_clear()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        r2 = await store.register_entities("face", [
            {"entity_token": "f1", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
            {"entity_token": "f2", "body": "B", "normal": [0, 0, -1], "centroid": [0, 0, 0], "surface_type": "planar"},
        ])
        ref_map_2 = {r.token: r.ref_id for r in r2}

        assert ref_map_1 == ref_map_2, f"Expected same refs, got {ref_map_1} vs {ref_map_2}"

    asyncio.run(_run())


def test_cache_invalidation():
    """After soft_clear, changed geometry gets a new face_N ID."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        r1 = await store.register_entities("face", [
            {"entity_token": "f1", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
        ])
        assert r1[0].ref_id == "face_0"

        store.soft_clear()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        # Same token but different geometry (centroid moved)
        r2 = await store.register_entities("face", [
            {"entity_token": "f1", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 50], "surface_type": "planar"},
        ])
        # Should get a new sequential ID since fingerprint changed
        assert r2[0].ref_id != "face_0", f"Expected new ID, got {r2[0].ref_id}"

    asyncio.run(_run())


def test_resolve_token():
    """resolve_token resolves face_N refs to their underlying tokens."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [
            {"entity_token": "bt0", "name": "A"},
        ])
        await store.register_entities("face", [
            {"entity_token": "ft1", "body": "A", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
            {"entity_token": "ft2", "body": "A", "normal": [0, 0, -1], "centroid": [0, 0, 0], "surface_type": "planar"},
        ])
        tok, err = store.resolve_token("face_0")
        assert tok == "ft1" and err is None, f"Got tok={tok}, err={err}"

        tok2, err2 = store.resolve_token("face_1")
        assert tok2 == "ft2" and err2 is None

        # Unknown ref
        tok3, err3 = store.resolve_token("face_99")
        assert tok3 is None and err3 is not None

    asyncio.run(_run())


def test_dedup_by_token():
    """Duplicate entity_tokens in the same batch are de-duplicated."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        refs = await store.register_entities("face", [
            {"entity_token": "f1", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
            {"entity_token": "f1", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar"},
        ])
        assert len(refs) == 1
        assert refs[0].ref_id == "face_0"

    asyncio.run(_run())


def test_face_ref_reuse_across_token_churn():
    """A new token with matching face fingerprint should reuse previous face_N ref."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        first = await store.register_entities("face", [
            {"entity_token": "tok_a", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar", "area": 100.0},
            {"entity_token": "tok_b", "body": "B", "normal": [0, 0, -1], "centroid": [0, 0, 0], "surface_type": "planar", "area": 100.0},
        ])
        first_refs = [r.ref_id for r in first]
        assert first_refs == ["face_0", "face_1"]

        store.soft_clear()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        second = await store.register_entities("face", [
            {"entity_token": "tok_a2", "body": "B", "normal": [0, 0, 1], "centroid": [0, 0, 10], "surface_type": "planar", "area": 100.0},
            {"entity_token": "tok_b2", "body": "B", "normal": [0, 0, -1], "centroid": [0, 0, 0], "surface_type": "planar", "area": 100.0},
        ])
        second_refs = [r.ref_id for r in second]
        assert second_refs == first_refs, f"Expected stable refs, got {second_refs}"

    asyncio.run(_run())


def test_face_ref_remap_ambiguity_assigns_new_id():
    """If multiple cached faces match, remap is ambiguous and a new face_N is assigned."""
    async def _run():
        store = EntityStore()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        initial = await store.register_entities("face", [
            {"entity_token": "tok_1", "body": "B", "normal": [0, 0, 1], "centroid": [1, 2, 3], "surface_type": "planar", "area": 12.34},
            {"entity_token": "tok_2", "body": "B", "normal": [0, 0, 1], "centroid": [1, 2, 3], "surface_type": "planar", "area": 12.34},
        ])
        assert [r.ref_id for r in initial] == ["face_0", "face_1"]

        store.soft_clear()
        await store.register_entities("body", [{"entity_token": "b0", "name": "B"}])
        remapped = await store.register_entities("face", [
            {"entity_token": "tok_new", "body": "B", "normal": [0, 0, 1], "centroid": [1, 2, 3], "surface_type": "planar", "area": 12.34},
        ])
        assert remapped[0].ref_id == "face_2", f"Expected new ID for ambiguous remap, got {remapped[0].ref_id}"

    asyncio.run(_run())


if __name__ == "__main__":
    tests = [
        test_sequential_face_naming,
        test_mixed_surface_types,
        test_multi_body_sequential,
        test_spatial_info_preserved,
        test_dict_centroid_handling,
        test_cache_reuse,
        test_cache_invalidation,
        test_resolve_token,
        test_dedup_by_token,
        test_face_ref_reuse_across_token_churn,
        test_face_ref_remap_ambiguity_assigns_new_id,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")
