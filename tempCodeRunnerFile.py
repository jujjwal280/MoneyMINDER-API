def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):

        # 🔥 FIX: Skip auth if Firebase not initialized (mock mode)
        if db is None:
            print("⚠️ Mock mode: skipping auth")
            return f("mock_user", *args, **kwargs)

        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify(
                {"success": False, "error": "Authorization header required"}
            ), 401

        token = auth_header.split("Bearer ")[1]

        uid = verify_firebase_token(token)

        if not uid:
            return jsonify({"success": False, "error": "Invalid token"}), 401

        return f(uid, *args, **kwargs)

    return wrapper