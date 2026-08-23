# Database Migrations

## How it works

SQL files are numbered sequentially (`001_`, `002_`, etc.). Applied migrations are tracked in a `schema_migrations` table.

## Apply a migration

1. Open **Supabase Dashboard** → **SQL Editor**
2. Paste the contents of the migration `.sql` file
3. Click **Run**
4. Mark it as applied:
   ```bash
   python -m migrations.run_migrations --mark 001 initial_schema
   ```

## Check pending migrations

```bash
python -m migrations.run_migrations
```

## Create a new migration

1. Create a new file: `migrations/002_add_new_table.sql`
2. Write your SQL
3. Follow the steps above to apply and mark

## File naming

```
migrations/
├── 001_initial_schema.sql
├── 002_add_new_feature.sql
└── run_migrations.py
```

Format: `{version}_{description}.sql` where version is a 3-digit zero-padded number.
