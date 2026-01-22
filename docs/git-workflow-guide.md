# Git Workflow Guide: Feature Branch Development

This guide explains how to safely develop and deploy features using Git branches.

---

## The Concept

```
main branch (stable, production)
    │
    └── feature/your-feature (experimental)
            │
            ▼
        Deploy & Test
            │
            ├── Works? → Merge to main
            │
            └── Broken? → Checkout main, redeploy
```

**Key principle:** `main` stays stable. All experimental work happens on feature branches.

---

## Step-by-Step Workflow

### 1. Start from Stable Main

```bash
git checkout main
git pull origin main    # Get latest (if using remote)
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Your local files are now on the feature branch. Any changes you make are isolated from main.

### 3. Make Your Changes

Edit code, add files, etc. Your changes only exist on this branch.

### 4. Commit Changes

```bash
git add -A
git commit -m "feat: Description of your changes"
```

### 5. Deploy Feature Branch to Test

```bash
./deploy.sh    # Deploys current branch (feature) to Azure
```

### 6a. If It Works → Merge to Main

```bash
git checkout main
git merge feature/your-feature-name
git push origin main    # Push to remote (optional)
```

### 6b. If It Breaks → Rollback to Main

```bash
git checkout main       # Local files instantly revert to stable
./deploy.sh             # Redeploy stable main to Azure
```

Your feature branch still exists - you can fix it and try again later.

---

## Quick Reference

| Command | What it does |
|---------|--------------|
| `git branch` | List branches (* = current) |
| `git branch --show-current` | Show current branch name |
| `git checkout main` | Switch to main (files change!) |
| `git checkout feature/x` | Switch to feature branch |
| `git checkout -b feature/x` | Create AND switch to new branch |
| `git merge feature/x` | Merge feature into current branch |
| `git branch -d feature/x` | Delete branch (after merge) |
| `git status` | Show uncommitted changes |

---

## Important Concepts

### Your Local Files = Mirror of Current Branch

When you `git checkout <branch>`, your local files **instantly become** that branch's version.

```
git checkout main
# → Your files are now main's version

git checkout feature/new-thing
# → Your files are now feature/new-thing's version
```

Nothing is lost - Git stores everything.

### Uncommitted Changes

If you have uncommitted changes when switching branches:
- Git will warn you
- Either commit them first, or stash them (`git stash`)

### Branches Are Cheap

Create branches freely. They're just pointers - they don't duplicate files.

---

## Example: Adding a New Feature

```bash
# 1. Start fresh from main
git checkout main
git pull

# 2. Create feature branch
git checkout -b feature/add-dark-mode

# 3. Make changes (edit files, add code, etc.)
# ... work work work ...

# 4. Commit
git add -A
git commit -m "feat: Add dark mode toggle"

# 5. Deploy to test
./deploy.sh

# 6. Test on live site...

# 7a. SUCCESS - merge to main
git checkout main
git merge feature/add-dark-mode
./deploy.sh    # Deploy merged main

# 7b. FAILURE - rollback
git checkout main
./deploy.sh    # Redeploy stable main
# Fix the feature branch later
```

---

## Recovery Scenarios

### "I deployed a broken feature"

```bash
git checkout main
./deploy.sh
```

### "I want to abandon this feature entirely"

```bash
git checkout main
git branch -D feature/broken-thing    # -D force deletes
```

### "I want to see what changed between branches"

```bash
git diff main feature/your-feature
```

### "I want to save my work but not commit yet"

```bash
git stash                    # Save uncommitted changes
git checkout other-branch    # Switch freely
git checkout original-branch
git stash pop                # Restore saved changes
```

---

## Summary

1. **main** = stable, always deployable
2. **feature branches** = experimental work
3. **deploy.sh** = deploys whatever branch you're on
4. **git checkout main** = instant rollback of local files
5. Branches are free - use them liberally
