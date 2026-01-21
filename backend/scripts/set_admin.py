#!/usr/bin/env python3
"""
Set admin status for a user.

Usage:
    python scripts/set_admin.py jadericdawson@gmail.com
    python scripts/set_admin.py jadericdawson@gmail.com --revoke

Run from the backend directory:
    cd backend && python scripts/set_admin.py jadericdawson@gmail.com
"""
import asyncio
import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.repositories import UserRepository


async def set_admin(email: str, revoke: bool = False):
    """Set or revoke admin status for a user."""
    repo = UserRepository()

    # Find user by email
    user = await repo.get_by_email(email)
    if not user:
        print(f"Error: User with email '{email}' not found")
        return False

    current_status = getattr(user, 'isAdmin', False)
    new_status = not revoke

    if current_status == new_status:
        status_word = "already an admin" if new_status else "already not an admin"
        print(f"User {user.displayName or email} ({user.id}) is {status_word}")
        return True

    # Update admin status
    await repo.update(user.id, {"isAdmin": new_status})

    action = "granted to" if new_status else "revoked from"
    print(f"Admin status {action} {user.displayName or email} ({user.id})")
    return True


def main():
    parser = argparse.ArgumentParser(description='Set admin status for a user')
    parser.add_argument('email', help='Email address of the user')
    parser.add_argument('--revoke', action='store_true', help='Revoke admin status instead of granting')

    args = parser.parse_args()

    # Run the async function
    success = asyncio.run(set_admin(args.email, args.revoke))
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
