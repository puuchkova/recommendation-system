from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_dashboard():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=90).run()
    assert not app.exception
    app.number_input[0].set_value(0).run()
    assert not app.exception
    assert any('Товаров с такими' in msg.value for msg in app.info)
