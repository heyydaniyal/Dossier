from streamlit.testing.v1 import AppTest


def test_app_renders_queue_without_errors():
    at = AppTest.from_file("../src/app/streamlit_app.py").run(timeout=30)
    assert not at.exception
    assert "Dossier" in at.title[0].value
    assert len(at.dataframe) == 1


def test_placeholders_pass_contract():
    from src.app.streamlit_app import placeholder_alerts

    assert all(a.alert_id.startswith("PLACEHOLDER") for a in placeholder_alerts())
