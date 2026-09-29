# Contributing

Mantle is a community-driven project and contributions of all sizes are
welcome.

If you discover a bug, have an idea for a feature, or would like to
contribute code, please open an issue or discussion first for larger
changes.

Before opening a new issue:

- Check whether the issue has already been reported.
- Provide steps to reproduce bugs whenever possible.
- Include sufficient context for feature requests.

## Creating a pull request

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a pull request.

Before submitting:

- Ensure existing tests pass (`pixi run -e test test`).
- Add tests for significant new functionality.
- Provide a clear explanation of the changes.
- Link any relevant issues or discussions.
- Include any special testing instructions if applicable.

Example of running a single test file while iterating:

```bash
pixi run mojo run -I . tests/mojo/test_ops.mojo
```

See the [style guide](style-guide.md) for docstring and code-comment
conventions used across the codebase.
