"""Memory tab handlers with real conversation_memory files."""
import pytest

pytestmark = [pytest.mark.real, pytest.mark.app]


def test_memory_refresh_select_approve_export(real_memory_dir, tmp_path):
    import app.gradio_ui as ui
    from app.gradio_ui import (
        _memory_approve,
        _memory_export,
        _memory_refresh,
        _memory_select,
        _parse_interaction_id,
    )

    old_mem, old_train = ui._MEMORY_DIR, ui._TRAINING_DATA_DIR
    ui._MEMORY_DIR = real_memory_dir
    ui._TRAINING_DATA_DIR = tmp_path / "training_data"
    try:
        stats, freq, dd_update = _memory_refresh()
        assert len(stats) > 10
        choices = dd_update.get("choices") or []
        assert len(choices) >= 1
        choice = choices[0]
        rid, detail = _memory_select(choice)
        assert rid == _parse_interaction_id(choice)
        assert "CSUM" in detail
        rid_list, detail_list = _memory_select([choice])
        assert rid_list == rid
        msg, stats2, detail2, dd2 = _memory_approve(rid)
        assert "Approved" in msg
        export_msg, file_up = _memory_export()
        assert "Exported" in export_msg or "approved" in export_msg.lower()
    finally:
        ui._MEMORY_DIR, ui._TRAINING_DATA_DIR = old_mem, old_train
