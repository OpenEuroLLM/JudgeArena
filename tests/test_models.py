def test_message_text_normalizes_aimessage_and_str():
    from judgearena.models import message_text

    class _Msg:
        content = "hi"

    assert message_text(_Msg()) == "hi"
    assert message_text("plain") == "plain"
