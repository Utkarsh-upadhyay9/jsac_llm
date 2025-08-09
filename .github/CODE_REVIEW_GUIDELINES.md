# Code Review Guidelines for JSAC LLM Project

## Overview
This document provides guidelines for conducting effective code reviews in our research project. Good code reviews improve code quality, share knowledge, and ensure reproducible research.

## Review Process

### For Authors (Creating PRs)
1. **Self-review first**: Review your own code before requesting review
2. **Small, focused PRs**: Keep changes focused on a single feature/fix
3. **Clear description**: Explain what, why, and how
4. **Include tests**: Add or update tests for your changes
5. **Update documentation**: Keep docs in sync with code changes

### For Reviewers
1. **Timely reviews**: Aim to review within 24-48 hours
2. **Thorough examination**: Check functionality, style, and research validity
3. **Constructive feedback**: Focus on improvement, not criticism
4. **Ask questions**: Seek clarification when needed
5. **Approve explicitly**: Use GitHub's review features

## Review Checklist

### ✅ Functionality
- [ ] Code does what it's supposed to do
- [ ] Edge cases are handled appropriately
- [ ] Error handling is implemented
- [ ] No obvious bugs or logical errors
- [ ] Performance is reasonable

### ✅ Code Quality
- [ ] Code is readable and maintainable
- [ ] Variable and function names are descriptive
- [ ] Functions are appropriately sized
- [ ] No code duplication
- [ ] Follows project coding standards

### ✅ Research Validity
- [ ] Mathematical implementations are correct
- [ ] Algorithms match paper descriptions
- [ ] Hyperparameters are appropriate
- [ ] Experimental setup is sound
- [ ] Results are reproducible

### ✅ Documentation
- [ ] Code is well-commented
- [ ] Docstrings are present and accurate
- [ ] README is updated if needed
- [ ] Complex algorithms are explained

### ✅ Testing
- [ ] Adequate test coverage
- [ ] Tests are meaningful and correct
- [ ] All tests pass
- [ ] No regression in existing functionality

### ✅ Experimental
- [ ] Figures are generated correctly
- [ ] Results files are properly saved
- [ ] Experimental configuration is documented
- [ ] Comparison with baselines is fair

## Review Comments

### Effective Comment Types

#### 🐛 **Bug/Issue**
```
There's a potential division by zero here when `sigma2` is 0.
Consider adding a check: `if sigma2 == 0: sigma2 = 1e-12`
```

#### 💡 **Suggestion**
```
Consider using a more descriptive variable name here.
`snr_values` would be clearer than `vals`.
```

#### ❓ **Question**
```
Could you explain the reasoning behind this hyperparameter choice?
Is this value from the paper or empirically determined?
```

#### 🎉 **Praise**
```
Nice optimization! This vectorized approach should be much faster
than the previous loop-based implementation.
```

#### 📚 **Knowledge Sharing**
```
For future reference, this is implementing the TZF beamforming
from equation (12) in the paper.
```

### Comment Guidelines
- **Be specific**: Reference exact lines or functions
- **Be constructive**: Suggest improvements, don't just point out problems
- **Be respectful**: Remember there's a person behind the code
- **Be clear**: Explain the reasoning behind your feedback
- **Be timely**: Don't let PRs sit without feedback

## Research-Specific Review Points

### Algorithm Implementation
- Mathematical correctness
- Consistency with published papers
- Appropriate use of numerical libraries
- Handling of edge cases in simulations

### Experimental Code
- Reproducibility (seeds, environment)
- Fair comparison with baselines
- Appropriate statistical analysis
- Clear documentation of setup

### Performance Analysis
- Computational complexity considerations
- Memory usage optimization
- Scalability concerns
- GPU utilization (if applicable)

### Plotting and Visualization
- Figure quality and clarity
- Appropriate choice of visualizations
- Consistent styling across figures
- Publication-ready output

## Common Issues to Watch For

### 🚨 **Critical Issues**
- Incorrect mathematical implementations
- Race conditions or concurrency issues
- Memory leaks or excessive resource usage
- Security vulnerabilities (hardcoded credentials, etc.)

### ⚠️ **Important Issues**
- Poor error handling
- Inefficient algorithms
- Code duplication
- Missing documentation for complex logic

### 💅 **Style Issues**
- Inconsistent naming conventions
- Long, complex functions
- Poor code organization
- Missing or inadequate comments

## Review Approval Guidelines

### ✅ **Approve** when:
- All checklist items are satisfied
- No critical or important issues remain
- Minor style issues can be addressed in follow-up
- Research contributions are valid and valuable

### 📝 **Request Changes** when:
- Critical bugs or issues are present
- Research methodology is flawed
- Code doesn't meet quality standards
- Documentation is severely lacking

### 💬 **Comment** when:
- You have suggestions but no blocking issues
- You want to share knowledge or ask questions
- Minor improvements could be made
- You're not the primary reviewer but have input

## Tools and Automation

### Automated Checks
- Code formatting (Black, isort)
- Linting (flake8)
- Basic import/syntax validation
- Simple plotting tests

### Manual Review Focus
- Research validity
- Algorithm correctness
- Experimental design
- Code logic and architecture

## Examples

### Good Review Comment
```
In the `compute_snr_delayed` function, I notice the interference 
calculation doesn't account for the case where V=1. When there's 
only one user, the interference term should be zero, but the current 
loop still runs. Consider adding:

if V == 1:
    interference = 0
else:
    # existing loop

This also matches the mathematical definition in equation (14).
```

### Poor Review Comment
```
This is wrong.
```

## Follow-up Actions

### After Review
1. **Author**: Address feedback promptly and thoroughly
2. **Reviewer**: Re-review changed code
3. **Team**: Merge when approved
4. **All**: Learn from feedback for future contributions

### Continuous Improvement
- Regularly review and update these guidelines
- Share lessons learned from reviews
- Celebrate good examples of code and reviews
- Address recurring issues through automation or training

Remember: The goal is to improve code quality, advance research, and help each other grow as researchers and developers!
