"""Tests for the git log commit fetcher."""

from social_graph.fetcher.git_log import _login_from_email, _user_from_git, _CO_AUTHOR_RE


def test_login_from_plain_email():
    assert _login_from_email("alice@example.com") == "alice"


def test_login_from_noreply_email():
    assert _login_from_email("12345+alice@users.noreply.github.com") == "alice"


def test_login_from_noreply_no_number():
    assert _login_from_email("alice@users.noreply.github.com") == "alice"


def test_user_from_git():
    u = _user_from_git("Alice Smith", "alice@example.com")
    assert u.login == "alice"
    assert u.name == "Alice Smith"
    assert u.id == "alice"


def test_co_author_regex_standard():
    match = _CO_AUTHOR_RE.search("Co-authored-by: Bob Jones <bob@example.com>")
    assert match is not None
    assert match.group(1) == "Bob Jones"
    assert match.group(2) == "bob@example.com"


def test_co_author_regex_case_insensitive():
    match = _CO_AUTHOR_RE.search("co-authored-by: Alice <alice@x.com>")
    assert match is not None
    assert match.group(1) == "Alice"


def test_co_author_regex_no_match():
    match = _CO_AUTHOR_RE.search("Authored-by: Someone <x@y.com>")
    assert match is None


def test_co_author_regex_multiple():
    text = (
        "Co-authored-by: Alice <a@x.com>\n"
        "Co-authored-by: Bob <b@x.com>\n"
    )
    matches = _CO_AUTHOR_RE.findall(text)
    assert len(matches) == 2
    assert matches[0] == ("Alice", "a@x.com")
    assert matches[1] == ("Bob", "b@x.com")
