import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.supabase_client import get_client

MIGRATIONS_DIR = Path(__file__).parent


def get_applied_migrations(client) -> set:
    try:
        resp = client.table("schema_migrations").select("version").execute()
        return {r["version"] for r in resp.data}
    except Exception:
        return set()


def get_pending_migrations(applied: set) -> list[tuple[str, str, Path]]:
    migrations = []
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = f.stem.split("_")[0]
        if version not in applied:
            name = f.stem
            migrations.append((version, name, f))
    return migrations


def mark_applied(client, version: str, name: str):
    client.table("schema_migrations").insert({
        "version": version,
        "name": name,
    }).execute()


def main():
    print("=" * 60)
    print("Quant Bot - Database Migrations")
    print("=" * 60)

    client = get_client()
    applied = get_applied_migrations(client)
    pending = get_pending_migrations(applied)

    if not pending:
        print("  No pending migrations. Database is up to date.")
    else:
        print(f"  {len(pending)} pending migration(s):")
        for version, name, filepath in pending:
            print(f"\n  --- {filepath.name} ---")
            print(f"  SQL file: {filepath}")
            print(f"  To apply, paste the SQL into Supabase SQL Editor, then run:")
            print(f"    python -m migrations.run_migrations --mark {version} {name}")
            print()

    print("=" * 60)


if __name__ == "__main__":
    if "--mark" in sys.argv:
        idx = sys.argv.index("--mark")
        if idx + 2 < len(sys.argv):
            client = get_client()
            mark_applied(client, sys.argv[idx + 1], sys.argv[idx + 2])
            print(f"Marked {sys.argv[idx + 1]} as applied.")
        else:
            print("Usage: python -m migrations.run_migrations --mark <version> <name>")
    else:
        main()
