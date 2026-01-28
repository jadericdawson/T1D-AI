#!/usr/bin/env python3
"""Find Denise and mark her email as verified."""
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

async def main():
    from database.repositories import UserRepository

    user_repo = UserRepository()

    print("Searching for users with 'denise' in email...")

    # Try to query CosmosDB directly
    try:
        # Access the container directly to query
        container = user_repo.container

        query = "SELECT * FROM c WHERE CONTAINS(LOWER(c.email), 'denise')"
        results = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        if not results:
            print("❌ No user found with 'denise' in email")
            return

        for user_doc in results:
            print(f"\n✓ Found user: {user_doc['email']}")
            print(f"  ID: {user_doc['id']}")
            print(f"  Display name: {user_doc.get('displayName', 'N/A')}")
            print(f"  Email verified: {user_doc.get('emailVerified', False)}")
            print(f"  Created: {user_doc.get('createdAt', 'N/A')}")

            # Mark as verified if not already
            if not user_doc.get('emailVerified', False):
                print(f"\n  Marking email as verified...")
                user_doc['emailVerified'] = True
                user_doc['emailVerificationToken'] = None
                user_doc['emailVerificationExpires'] = None
                user_doc['updatedAt'] = datetime.utcnow().isoformat()

                # Update in CosmosDB
                container.upsert_item(user_doc)
                print(f"  ✅ Email verified successfully!")
            else:
                print(f"  ℹ Email already verified")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
