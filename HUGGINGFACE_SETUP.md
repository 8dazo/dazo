# Hugging Face Trusted Publishing setup

Dazo publishes from GitHub Actions to `d2v1shx/dazo` without a long-lived `HF_TOKEN`.

## One-time Hugging Face setup

1. Sign in to Hugging Face as `d2v1shx`.
2. Create a **Model** repository named `dazo` under the `d2v1shx` account.
   - Target repo: `d2v1shx/dazo`
   - Public is recommended for the research project.
3. Open the new model repository's **Settings** page.
4. Open **Trusted Publishers** and add a publisher with:
   - Provider: `GitHub Actions`
   - Repository: `8dazo/dazo`
   - Branch: `main`
   - Workflow: `publish-hf.yml`
5. Save the publisher.

All claims are exact-match. Keep the GitHub workflow filename exactly `.github/workflows/publish-hf.yml` unless you also update the Trusted Publisher entry.

## First publish

In GitHub, open:

`8dazo/dazo` → **Actions** → **Publish Dazo to Hugging Face** → **Run workflow** → `main` → **Run workflow**.

The workflow requests a short-lived OIDC identity token from GitHub, exchanges it for a one-hour Hugging Face token scoped only to `d2v1shx/dazo`, builds `.hf_publish`, and uploads it.

No `HF_TOKEN` GitHub secret is required.

## After the first successful publish

The workflow can be changed from manual-only to automatic publishing on pushes to `main` by adding:

```yaml
on:
  push:
    branches: [main]
  workflow_dispatch:
```

## Security

The GitHub workflow has only:

```yaml
permissions:
  contents: read
  id-token: write
```

The Hugging Face token minted by Trusted Publishing is short-lived and repository-scoped. It cannot write to other Hub repositories.
