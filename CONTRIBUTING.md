# Contributing to JSAC LLM Project

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing to the JSAC (Joint Sensing and Communication) implementation with LLM-based actors.

## Quick Contribution TL;DR
1. Fork + branch
2. Make atomic change (one concern)
3. Regenerate any affected plots (keep artifacts synced)
4. Add / update docstrings
5. Open PR with rationale + sample figure (if visual)

## Getting Started

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Git

### Setup Development Environment
```bash
git clone https://github.com/Utkarsh-upadhyay9/jsac_llm.git
cd jsac_llm
pip install -r requirements.txt
```

## How to Contribute

### Types of Contributions
- 🐛 **Bug fixes**: Fix issues in the codebase
- ✨ **New features**: Add new algorithms or functionality
- 📊 **Experiments**: Share experimental results and improvements
- 📚 **Documentation**: Improve documentation and examples
- 🔧 **Performance**: Optimize existing code

### Contribution Process
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** (if applicable)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

## Code Standards

### Python Style Guide
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Maximum line length: 127 characters

### Commit Message Format
```
type(scope): description

Examples:
feat(plots): add figure 3 with hybrid superiority
fix(training): resolve convergence issues in LLM actor
docs(readme): update installation instructions
test(jsac): add unit tests for reward calculation
```

### Code Review Checklist
- [ ] Code is readable and well-documented
- [ ] No hardcoded paths or values
- [ ] Error handling is implemented
- [ ] Performance considerations addressed
- [ ] Experimental results are reproducible

## Research-Specific Guidelines

### Experimental Contributions
- **Document hyperparameters**: Include all configuration details
- **Reproducible results**: Provide seeds and environment setup
- **Performance metrics**: Include quantitative comparisons
- **Figures**: Generate publication-quality plots

### Algorithm Implementations
- **Mathematical notation**: Match paper notation where possible
- **Comments**: Explain complex algorithms step-by-step
- **Validation**: Compare against reference implementations
- **Efficiency**: Consider computational complexity

### Data and Results
- **Version control**: Use Git LFS for large data files
- **Naming conventions**: Use descriptive names for result files
- **Metadata**: Include experimental configuration with results
- **Comparison**: Provide baseline comparisons

## Testing

### Running Tests
```bash
# Code quality checks
flake8 .
black --check .
isort --check .

# Basic functionality tests
python -c "from jsac_active_ris_dam import *"
python comprehensive_plots.py
```

### Experimental Validation
- Verify scripts run without errors
- Check that figures are generated correctly
- Ensure results are reproducible
- Test with different random seeds

## Documentation

### Code Documentation
- Add docstrings to all public functions
- Include parameter descriptions and return values
- Provide usage examples
- Document any assumptions or limitations

### Research Documentation
- Update README with new features
- Add references to relevant papers
- Include performance benchmarks
- Provide usage examples

## Issue Reporting

### Bug Reports
Use the bug report template and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Error messages/logs

### Feature Requests
Use the feature request template and include:
- Problem statement
- Proposed solution
- Use case description
- Implementation ideas

### Experimental Results
Use the experiment results template and include:
- Configuration details
- Performance metrics
- Generated figures
- Key findings
- Comparison with baselines

## Questions and Support

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Email**: Contact repository maintainers for sensitive issues

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be acknowledged in:
- README contributors section
- Research paper acknowledgments (for significant contributions)
- Release notes

Thank you for contributing to advancing research in joint sensing and communication!
