#!/usr/bin/env python3
"""List all users, especially recent ones."""
import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def main():
    from database.repositories import UserRepository

    user_repo = UserRepository()

    try:
        container = user_repo.container

        # Get all users sorted by creation date
        query = "SELECT c.email, c.displayName, c.emailVerified, c.createdAt, c.id FROM c ORDER BY c.createdAt DESC"
        results = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        print(f"Total users: {len(results)}\n")
        print("Recent users (last 10):")
        print("-" * 80)

        for i, user in enumerate(results[:10], 1):
            created = user.get('createdAt', 'Unknown')
            verified = "✓" if user.get('emailVerified') else "✗"
            print(f"{i}. {verified} {user['email']}")
            print(f"   Name: {user.get('displayName', 'N/A')}")
            print(f"   Created: {created}")
            print(f"   ID: {user['id'][:20]}...")
            print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
