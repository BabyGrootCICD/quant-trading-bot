from supabase import create_client, Client
from config.settings import SUPABASE_URL, SUPABASE_KEY


def get_client() -> Client:
    if not SUPABASE_URL:
        raise ValueError(
            "SUPABASE_URL is not set. Set it as a GitHub Actions secret "
            "or in a .env file. Example: https://your-project.supabase.co"
        )
    if not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_KEY is not set. Set it as a GitHub Actions secret "
            "or in a .env file. Find it in Supabase Dashboard > Settings > API."
        )
    url = SUPABASE_URL.rstrip("/")
    if not url.startswith("http"):
        url = "https://" + url
    return create_client(url, SUPABASE_KEY)
