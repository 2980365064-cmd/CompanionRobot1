from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FW = ROOT / "esp_sparkbot/example/factory_demo_v1"


def test_personal_agent_client_exposes_v2_audio_protocol():
    header = (FW / "components/personal_agent_ws/include/personal_agent_ws.h").read_text()
    source = (FW / "components/personal_agent_ws/src/personal_agent_ws.c").read_text()

    assert "personal_agent_ws_audio_start" in header
    assert "personal_agent_ws_send_audio" in header
    assert "personal_agent_ws_audio_end" in header
    assert '"audio_start"' in source
    assert '"audio_end"' in source
    assert "esp_websocket_client_send_bin" in source
    assert "esp_timer_get_time" in source


def test_recording_streams_pcm_and_uses_400ms_vad_window():
    source = (FW / "main/app/app_audio_record.c").read_text()

    assert "personal_agent_ws_audio_start" in source
    assert "personal_agent_ws_send_audio" in source
    assert "personal_agent_ws_audio_end" in source
    assert "VAD_SILENCE_CHECKS 4" in source


def test_default_backend_uri_uses_audio_v2():
    kconfig = (FW / "main/Kconfig.projbuild").read_text()

    assert 'default "ws://114.55.134.145:8000/ws/v2/audio"' in kconfig
