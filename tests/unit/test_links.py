from semcache.links import has_disallowed_link, host_allowed, hosts_in


def test_hosts_in_extracts_hosts() -> None:
    text = "see https://www.BazaarNow.in/orders?id=1 and www.example.com/x, http://a.b:8080/p"
    assert hosts_in(text) == ["www.bazaarnow.in", "www.example.com", "a.b"]


def test_host_allowed_subdomains_only() -> None:
    assert host_allowed("bazaarnow.in", ["bazaarnow.in"])
    assert host_allowed("help.bazaarnow.in", ["bazaarnow.in"])
    assert not host_allowed("bazaarnow.in.evil.com", ["bazaarnow.in"])
    assert not host_allowed("notbazaarnow.in", ["bazaarnow.in"])


def test_has_disallowed_link() -> None:
    assert not has_disallowed_link("no links here", [])
    assert has_disallowed_link("go to https://x.example", [])
    assert not has_disallowed_link("go to https://bazaarnow.in", ["BazaarNow.in"])
