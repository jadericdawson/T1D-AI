#!/usr/bin/env python3
"""Find which user ID has Gluroo configured."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from database.repositories import DataSourceRepository

async def main():
    datasource_repo = DataSourceRepository()

    # Try both possible IDs
    ids_to_try = [
        "05bf0083-5598-43a5-aa7f-bd70b1f1be57",  # Account owner
        "profile_05bf0083-5598-43a5-aa7f-bd70b1f1be57",  # Child profile
        "c4c77958-3e25-42b6-9f9a-f8a6c6aea098",  # Jaderic self profile
    ]

    for user_id in ids_to_try:
        try:
            datasource = await datasource_repo.get(user_id, "gluroo")
            if datasource:
                print(f"✓ FOUND Gluroo connection for: {user_id}")
                print(f"  URL: {datasource.config.get('url')}")
                print(f"  Last sync: {datasource.lastSyncedAt}")
                print(f"  Profile ID: {datasource.profileId if hasattr(datasource, 'profileId') else 'N/A'}")
                return user_id
        except Exception as e:
            print(f"  {user_id}: {e}")

    print("\nNo Gluroo connection found for any ID!")

if __name__ == "__main__":
    asyncio.run(main())
