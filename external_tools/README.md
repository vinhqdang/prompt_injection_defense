# External Tools

This directory contains external repositories used for comparison and research:

## Included Repositories

### 1. malicious-prompt-detection
- **Source**: https://github.com/AhsanAyub/malicious-prompt-detection
- **Purpose**: ML-based approach for detecting malicious prompts using various classifiers
- **Key Features**: Machine learning models, feature extraction, comprehensive evaluation

### 2. promptmap
- **Source**: https://github.com/utkusen/promptmap
- **Purpose**: Automated prompt injection testing tool
- **Key Features**: Pattern-based detection, multiple injection techniques, API testing

### 3. Open-Prompt-Injection
- **Source**: https://github.com/liu00222/Open-Prompt-Injection
- **Purpose**: Comprehensive toolkit for prompt injection attacks and defenses
- **Key Features**: DataSentinel detection, multiple attack strategies, formal benchmarking

## Usage

These repositories are cloned as subprojects for research and comparison purposes. Each maintains its own git history and can be updated independently.

To update external tools:
```bash
cd external_tools/malicious-prompt-detection
git pull origin main

cd ../promptmap
git pull origin main

cd ../Open-Prompt-Injection
git pull origin main
```

## Integration

See the main project's comparison scripts for how these tools are integrated into the prompt injection defense analysis.