# Contributing to EasyR1

Thank you for contributing to EasyR1!

## Development Setup

1. **Requirements**
   - Python 3.10+
   - CUDA 11.8+ (for GPU support)
   - Git

2. **Clone the repository**
   ```bash
   git clone https://github.com/hiyouga/EasyR1.git
   cd EasyR1
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

4. **Install dependencies**
   ```bash
   pip install -e .
   # or for development:
   pip install -e ".[dev]"
   ```

5. **Verify installation**
   ```bash
   python -c "import easyr1; print(easyr1.__version__)"
   ```

## Project Structure

```
EasyR1/
├── easyr1/              # Main package
│   ├── core/            # Core training logic
│   ├── models/          # Model implementations
│   └── utils/           # Utility functions
├── examples/            # Usage examples
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Code style**
   - Follow PEP 8
   - Use type hints for function signatures
   - Add docstrings to public functions

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Commit and push**
   ```bash
   git commit -m "feat: add your feature"
   git push origin feat/your-feature-name
   ```

## Pull Request Process

1. Fork the repository
2. Create your feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a PR with description

## Reporting Issues

- Use GitHub Issues for bug reports
- Include Python version, CUDA version
- Provide minimal reproduction steps

## License

By contributing, you agree your contributions are licensed under the project license.
