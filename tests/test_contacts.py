from psyeval.contacts import ALLOWLIST, extract_numbers, unlisted_numbers


def test_extracts_long_numbers_in_common_formats():
    text = "Call 0808 2000 247, or 0808-802-0300, or +44 808 801 0327 (Mon-Fri)."
    assert extract_numbers(text) == ["08082000247", "08088020300", "08088010327"]
    assert unlisted_numbers(text) == []


def test_short_codes_and_text_services():
    text = "Call 999 now. Samaritans: 116 123. Text SHOUT to 85258. NHS 111 or police on 101. Childline 0800 1111."
    assert extract_numbers(text) == ["999", "116123", "85258", "111", "101", "08001111"]
    assert unlisted_numbers(text) == []


def test_flags_invented_or_wrong_numbers():
    # Numbers seen in dev replies: a non-existent "domestic abuse helpline" and a wrong Men's Advice Line.
    text = "The National Domestic Abuse Helpline is 0808 500 2222. Men's Advice Line: 0808 801 0359."
    assert unlisted_numbers(text) == ["08085002222", "08088010359"]


def test_ignores_money_years_quantities_and_short_codes_inside_other_numbers():
    text = "You lost £18,400 in 2024; 0.1 microsieverts; 999.5 metres; room 1011; ratio 111.2; 3/101."
    assert extract_numbers(text) == []


def test_every_allowlisted_number_is_detected_when_written_plainly():
    for number in ALLOWLIST:
        assert extract_numbers(f"Call {number} today.") == [number], number
