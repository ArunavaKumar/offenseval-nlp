from backend.preprocessing import clean_tweet


def test_clean_tweet():
    text = "RT @john: You are AWESOME!!! https://abc.com"

    cleaned = clean_tweet(text)

    assert isinstance(cleaned, str)
    assert "http" not in cleaned
    assert "@" not in cleaned