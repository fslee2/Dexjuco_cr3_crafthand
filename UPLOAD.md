# Upload To GitHub

This local showcase repository is already initialized and committed.

Current blocker:

```text
gh CLI is not installed on this machine, and the Codex GitHub App has no
authorized repositories/accounts available in this session.
```

## Option A: GitHub CLI

Install GitHub CLI:

```powershell
winget install --id GitHub.cli
```

Restart PowerShell, then login:

```powershell
gh auth login
```

Publish:

```powershell
cd E:\material_for_uci_experiment\cr3-craft-teleop-showcase
.\scripts\publish_to_github.ps1
```

This will create a public repository named:

```text
cr3-craft-teleop-showcase
```

## Option B: Existing Empty GitHub Repository

If you already created an empty GitHub repo in the browser:

```powershell
cd E:\material_for_uci_experiment\cr3-craft-teleop-showcase
git remote add origin https://github.com/<your-user>/<repo-name>.git
git push -u origin main
```

## Option C: GitHub Desktop

1. Open GitHub Desktop.
2. File -> Add local repository.
3. Select:

```text
E:\material_for_uci_experiment\cr3-craft-teleop-showcase
```

4. Publish repository.

