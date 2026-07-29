from backend.inference import predict_sentence


def test_predict_sentence():

    result = predict_sentence(
        "You are an idiot"
    )

    assert isinstance(result, dict)

    assert "prediction" in result
    assert "confidence" in result
    assert "cleaned_text" in result