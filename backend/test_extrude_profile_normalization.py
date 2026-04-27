import pytest

try:
    # When collected as part of the `backend` package (pytest default for this repo layout).
    from .code_generator import CodeGenerationError, translate_tool_call
except ImportError:  # pragma: no cover - fallback for alternate import contexts
    from backend.backend.code_generator import CodeGenerationError, translate_tool_call


def test_extrude_profile_empty_profile_indices_is_rejected():
    with pytest.raises(CodeGenerationError) as exc:
        translate_tool_call(
            "extrude_profile",
            {
                "sketch_id": "sk1",
                "distance": 1.0,
                "profile_indices": [],
                "description": "x",
            },
        )
    assert '"profile_indices" cannot be empty.' in str(exc.value)


def test_extrude_profile_profile_index_and_profile_indices_conflict_is_rejected():
    with pytest.raises(CodeGenerationError) as exc:
        translate_tool_call(
            "extrude_profile",
            {
                "sketch_id": "sk1",
                "distance": 1.0,
                "profile_index": 0,
                "profile_indices": [0, 1],
                "description": "x",
            },
        )
    assert 'Provide either "profile_index" or "profile_indices", not both.' in str(exc.value)


def test_extrude_profile_negative_profile_indices_still_rejected():
    with pytest.raises(CodeGenerationError) as exc:
        translate_tool_call(
            "extrude_profile",
            {
                "sketch_id": "sk1",
                "distance": 1.0,
                "profile_indices": [-1],
                "description": "x",
            },
        )
    assert '"profile_indices" entries must be zero or greater.' in str(exc.value)


def test_extrude_profile_nonnumeric_profile_indices_still_rejected():
    with pytest.raises(CodeGenerationError):
        translate_tool_call(
            "extrude_profile",
            {
                "sketch_id": "sk1",
                "distance": 1.0,
                "profile_indices": ["a"],
                "description": "x",
            },
        )


@pytest.mark.parametrize("plane_alias", ["datum_plane", "reference_plane"])
@pytest.mark.parametrize("offset_alias", ["offset", "offset_distance"])
def test_create_construction_plane_offset_aliases_are_normalized(plane_alias, offset_alias):
    code = translate_tool_call(
        "create_construction_plane",
        {
            "plane_id": "p1",
            "mode": "offset_from_datum",
            "description": "offset plane",
            plane_alias: "xy",
            offset_alias: 1.25,
        },
    )

    assert 'base_datum_plane="XY"' in code
    assert "offset_cm=1.25" in code


# --- list_sketch_profiles enricher hint tests ---

def _import_enricher():
    """Import the enricher function, setting a dummy API key if needed."""
    import os
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
    try:
        from .agent_workflow import _enrich_list_sketch_profiles_result
    except ImportError:
        from backend.backend.agent_workflow import _enrich_list_sketch_profiles_result
    return _enrich_list_sketch_profiles_result


def test_list_sketch_profiles_enricher_hint_for_multiple_profiles():
    """When list_sketch_profiles returns 2+ profiles, the enricher should add a HINT suggesting profile_indices."""
    enricher = _import_enricher()
    result = {
        "sketch_id": "sketch_base",
        "profile_count": 3,
        "profiles": [
            {"index": 0, "area_cm2": None, "centroid_world_cm": None, "outer_loop_count": 1, "inner_loop_count": 0},
            {"index": 1, "area_cm2": None, "centroid_world_cm": None, "outer_loop_count": 1, "inner_loop_count": 0},
            {"index": 2, "area_cm2": None, "centroid_world_cm": None, "outer_loop_count": 1, "inner_loop_count": 0},
        ],
    }
    text = enricher("list_sketch_profiles", result, {})
    assert "HINT" in text
    assert "profile_indices=[0, 1, 2]" in text
    assert "omit profile_index" in text


def test_list_sketch_profiles_enricher_no_hint_for_single_profile():
    """When list_sketch_profiles returns only 1 profile, no hint should be added."""
    enricher = _import_enricher()
    result = {
        "sketch_id": "sketch_base",
        "profile_count": 1,
        "profiles": [
            {"index": 0, "area_cm2": 5.0, "centroid_world_cm": [0, 0, 0], "outer_loop_count": 1, "inner_loop_count": 0},
        ],
    }
    text = enricher("list_sketch_profiles", result, {})
    assert "HINT" not in text


def test_extrude_profile_only_profile_indices_no_profile_index():
    """When only profile_indices is provided (profile_index omitted), the template should use them."""
    code = translate_tool_call(
        "extrude_profile",
        {
            "sketch_id": "sk1",
            "distance": 2.0,
            "profile_indices": [0, 1, 2],
            "description": "Extruding all overlapping profiles as one shape",
        },
    )
    assert "_profile_indices = [0, 1, 2]" in code
    # The fallback guard is present but won't execute since _profile_indices is not None
    assert "if _profile_indices is None:" in code


def test_revolve_profile_axis_empty_object_is_rejected():
    with pytest.raises(CodeGenerationError) as exc:
        translate_tool_call(
            "revolve_profile",
            {
                "sketch_id": "sk1",
                "profile_index": 0,
                "axis": {},
                "description": "x",
            },
        )
    assert '"axis" cannot be an empty object.' in str(exc.value)


@pytest.mark.parametrize("mode_value", ["", None, "none"])
def test_revolve_profile_extent_invalid_mode_is_rejected(mode_value):
    with pytest.raises(CodeGenerationError):
        translate_tool_call(
            "revolve_profile",
            {
                "sketch_id": "sk1",
                "profile_index": 0,
                "axis": {"type": "construction", "axis": "z"},
                "extent": {"mode": mode_value},
                "description": "x",
            },
        )


def test_revolve_profile_extent_full_with_extra_fields_is_rejected():
    with pytest.raises(CodeGenerationError) as exc:
        translate_tool_call(
            "revolve_profile",
            {
                "sketch_id": "sk1",
                "profile_index": 0,
                "axis": {"type": "construction", "axis": "z"},
                "extent": {"mode": "full", "angle_degrees": 30},
                "description": "x",
            },
        )
    assert 'Unexpected field(s) for extent mode "full": angle_degrees' in str(exc.value)


def test_revolve_profile_extent_omitted_keeps_full_default():
    code = translate_tool_call(
        "revolve_profile",
        {
            "sketch_id": "sk1",
            "profile_index": 0,
            "axis": {"type": "construction", "axis": "z"},
            "description": "x",
        },
    )
    assert "extent_spec = {'mode': 'full'}" in code
