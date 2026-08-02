"""Tests for parallel face pair computation when normals are inconsistent."""
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

# agent_workflow imports llm_client which instantiates an OpenAI client at import-time.
# Provide a dummy key for unit tests.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from backend.agent_workflow import _compute_parallel_face_pairs


def test_parallel_pairs_with_same_sign_normals():
    entity_context = {
        "spatial_context": {
            "units": "mm",
            "bodies": [
                {
                    "entity_ref": "body_0",
                    "name": "Box",
                    "bbox": {"min": [0, 0, 0], "max": [1, 1, 1]},
                    "faces": [
                        # Both normals point +Z (mis-signed bottom), but centroids differ
                        {"entity_ref": "face_0", "normal": [0, 0, 1], "centroid": [0, 0, 10]},
                        {"entity_ref": "face_1", "normal": [0, 0, 1], "centroid": [0, 0, 0]},
                    ],
                }
            ],
        }
    }

    pairs = _compute_parallel_face_pairs(entity_context)
    assert "body_0" in pairs
    pair_refs = [set(p[:2]) for p in pairs["body_0"]]
    assert {"face_0", "face_1"} in pair_refs


def test_parallel_pairs_excludes_toroidal_faces():
    entity_context = {
        "spatial_context": {
            "units": "mm",
            "bodies": [
                {
                    "entity_ref": "body_0",
                    "name": "Box",
                    "bbox": {"min": [0, 0, 0], "max": [1, 1, 1]},
                    "faces": [
                        {"entity_ref": "face_0", "surface_type": "planar", "normal": [-1, 0, 0], "centroid": [0, 0, 0]},
                        {"entity_ref": "face_1", "surface_type": "planar", "normal": [1, 0, 0], "centroid": [10, 0, 0]},
                        # Non-planar: should be ignored even if axis-ish normal exists
                        {"entity_ref": "face_2", "surface_type": "toroidal", "normal": [1, 0, 0], "centroid": [9, 0, 0]},
                    ],
                }
            ],
        }
    }

    pairs = _compute_parallel_face_pairs(entity_context)
    assert "body_0" in pairs
    for f1, f2, _ in pairs["body_0"]:
        assert "face_2" not in (f1, f2)
