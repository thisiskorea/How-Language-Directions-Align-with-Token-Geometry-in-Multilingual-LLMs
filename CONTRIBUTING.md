# Contributing to Multilingual LLM Analysis

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing to the codebase.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU, etc.)
   - Relevant error messages or logs

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**:
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed
4. **Test your changes**:
   - Ensure existing experiments still run
   - Add new tests if applicable
5. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference related issues (e.g., "Fixes #123")
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Submit a Pull Request**

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions:
  ```python
  def analyze_probe_geometry(probe, layer_idx, model_name):
      """
      Analyze geometric properties of language probe directions

      Args:
          probe: Trained probe model
          layer_idx: Layer index
          model_name: Model name string

      Returns:
          dict with geometry metrics
      """
  ```

### Comments

- Use English for docstrings and main comments
- Korean comments are acceptable for implementation details
- Explain *why*, not just *what*

### Naming Conventions

- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Variables: `snake_case`

## Adding New Features

### Adding a New Model

1. Add model name to `MODEL_LIST` in scripts
2. Ensure model is compatible with Hugging Face Transformers
3. Test feature extraction and probe training
4. Update documentation

### Adding a New Language

1. Ensure XNLI dataset supports the language
2. Update language lists in scripts
3. Add language-specific token detection rules in `detect_token_language()`
4. Test with all analysis pipelines

### Adding New Analysis

1. Create a new function following existing patterns
2. Document inputs, outputs, and methodology
3. Add to main experiment pipeline
4. Update README with new outputs

## Testing

Before submitting changes:

```bash
# Test data preparation
python prepare_data.py --output_dir ./test_data

# Test main analysis (on single layer for speed)
# Modify script to run only Layer 0 for testing

# Verify outputs
ls -la ./results/
```

## Documentation

When adding features:

1. **Update README.md**: Add usage examples
2. **Update CLAUDE.md**: Add technical details for AI assistants
3. **Add inline comments**: Explain complex logic
4. **Create examples**: Show how to use new features

## Research Reproducibility

If your changes affect experiment results:

1. **Document changes** in commit messages
2. **Provide comparison** with original results
3. **Update seed settings** if needed
4. **Test with multiple random seeds** for statistical robustness

## Questions?

- Open an issue for discussion
- Email the authors:
  - JaeSeong Kim: mmmqp1010@gmail.com
  - Suan Lee: suanlee@semyung.ac.kr

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping improve this research!
