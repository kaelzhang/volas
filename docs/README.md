# volas docs source

The rendered documentation starts at [index.md](index.md). The remaining
Markdown files are focused guides included in that page's Sphinx navigation.

Build the site from the repository root:

```sh
python -m pip install --requirement docs/requirements.txt
make docs
```

## Read the Docs publication

`.readthedocs.yaml` is the production build contract. After the repository is
imported through the Read the Docs GitHub integration, a push to `main`
automatically rebuilds the `latest` version. The `docs` GitHub workflow runs
the same warning-as-error build on pull requests and pushes; it does not store
publication credentials or upload a second copy of the site.
