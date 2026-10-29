"""Authentication, authorisation and rate limiting (real auth: `auth_client`)."""

PW = "correct horse battery"


def register(c, email="a@example.com", pw=PW):
    r = c.post("/api/v1/auth/register", json={"email": email, "password": pw})
    assert r.status_code == 201, r.text
    return r.json()


def hdr(tok):
    return {"Authorization": f"Bearer {tok['access_token']}"}


def test_register_login_me(auth_client):
    tok = register(auth_client)
    assert auth_client.get("/api/v1/auth/me", headers=hdr(tok)).json()["email"] == "a@example.com"
    login = auth_client.post("/api/v1/auth/login", json={"email": "A@Example.com", "password": PW})
    assert login.status_code == 200 and login.json()["access_token"]


def test_passwords_are_argon2_hashed(auth_client, database):
    register(auth_client)
    h = database.users.find_one({"email": "a@example.com"})["password_hash"]
    assert h.startswith("$argon2") and PW not in h


def test_duplicate_email_weak_password_and_bad_login(auth_client):
    register(auth_client)
    assert auth_client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": PW}).status_code == 409
    assert auth_client.post("/api/v1/auth/register", json={"email": "b@example.com", "password": "short"}).status_code == 422
    wrong = auth_client.post("/api/v1/auth/login", json={"email": "a@example.com", "password": "wrong password!!"})
    unknown = auth_client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "wrong password!!"})
    assert wrong.status_code == unknown.status_code == 401 and wrong.json()["message"] == unknown.json()["message"]
    assert set(wrong.json()) >= {"code", "message"}


def test_protected_routes_need_a_token(auth_client):
    assert auth_client.get("/api/v1/datasets").status_code == 401
    assert auth_client.get("/api/v1/datasets", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_refresh_token_rotates_and_replay_revokes_everything(auth_client):
    tok = register(auth_client)
    new = auth_client.post("/api/v1/auth/refresh", json={"refresh_token": tok["refresh_token"]})
    assert new.status_code == 200
    replay = auth_client.post("/api/v1/auth/refresh", json={"refresh_token": tok["refresh_token"]})
    assert replay.status_code == 401
    again = auth_client.post("/api/v1/auth/refresh", json={"refresh_token": new.json()["refresh_token"]})
    assert again.status_code == 401       # the whole token family was revoked by the replay


def test_access_token_cannot_be_used_as_refresh_token(auth_client):
    tok = register(auth_client)
    assert auth_client.post("/api/v1/auth/refresh", json={"refresh_token": tok["access_token"]}).status_code == 401


def test_workspaces_are_isolated(auth_client):
    a, b = register(auth_client, "a@example.com"), register(auth_client, "b@example.com")
    ds = auth_client.post("/api/v1/datasets/demo", headers=hdr(a)).json()["dataset"]["id"]
    assert auth_client.get(f"/api/v1/datasets/{ds}", headers=hdr(a)).status_code == 200
    for method, path in (("get", f"/api/v1/datasets/{ds}"), ("get", f"/api/v1/datasets/{ds}/profile"), ("delete", f"/api/v1/datasets/{ds}"),
                         ("post", f"/api/v1/datasets/{ds}/analyses")):
        r = getattr(auth_client, method)(path, headers=hdr(b), **({"json": {}} if method == "post" else {}))
        assert r.status_code == 404, (method, path, r.status_code)
    assert auth_client.get("/api/v1/datasets", headers=hdr(b)).json()["total"] == 0


def test_viewer_role_is_read_only(auth_client, database):
    tok = register(auth_client)
    database.workspaces.update_one({}, {"$set": {"members.0.role": "viewer"}})
    assert auth_client.get("/api/v1/datasets", headers=hdr(tok)).status_code == 200
    assert auth_client.post("/api/v1/datasets/demo", headers=hdr(tok)).status_code == 403


def test_guest_session_expires_automatically(auth_client, database):
    r = auth_client.post("/api/v1/auth/guest")
    assert r.status_code == 201 and r.json()["user"]["guest"] is True
    ws = database.workspaces.find_one({})
    assert ws["expires_at"] is not None


def test_auth_endpoints_are_rate_limited(auth_client, settings):
    codes = [auth_client.post("/api/v1/auth/login", json={"email": "x@example.com", "password": "wrong password!!"}).status_code
             for _ in range(settings.auth_rate_limit_per_min * 2 + 5)]
    assert 429 in codes and codes[0] == 401
