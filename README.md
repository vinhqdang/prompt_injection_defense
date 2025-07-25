# Homomorphic Encryption for Intrusion Detection System

This project implements a privacy-preserving intrusion detection system using homomorphic encryption, evaluated on the CIC-IDS2017 dataset and compared with state-of-the-art machine learning models.

## Features

- **Homomorphic Encryption**: Privacy-preserving machine learning using TenSEAL (CKKS scheme)
- **Baseline Models**: XGBoost, Random Forest, and Deep Learning models for comparison
- **CIC-IDS2017 Dataset**: Industry-standard dataset for intrusion detection evaluation
- **GPU Support**: CUDA acceleration for deep learning and XGBoost
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations

## Requirements

- Python 3.10+
- CUDA-compatible GPU (optional, for acceleration)
- Conda package manager

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd homomorphic-encryption-ids
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate homomorphic-ids
```

## Dataset Setup

1. Download the CIC-IDS2017 dataset from:
   https://www.unb.ca/cic/datasets/ids-2017.html

2. Extract the CSV files to the `data/raw/` directory:
```
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

## Usage

### Basic Usage

Run the complete pipeline:
```bash
python main.py
```

### Advanced Options

```bash
python main.py --help
```

Options:
- `--data-path`: Path to raw dataset (default: `data/raw`)
- `--skip-baseline`: Skip baseline model training
- `--skip-homomorphic`: Skip homomorphic model training
- `--test-size`: Test set size ratio (default: 0.2)
- `--sample-size`: Use subset of data for testing

### Examples

1. **Quick test with small sample**:
```bash
python main.py --sample-size 5000
```

2. **Train only homomorphic models**:
```bash
python main.py --skip-baseline
```

3. **Train only baseline models**:
```bash
python main.py --skip-homomorphic
```

## Project Structure

```
├── data/
│   ├── raw/                    # Raw CIC-IDS2017 dataset
│   └── processed/              # Preprocessed data
├── src/
│   ├── data/
│   │   └── cic_ids_loader.py   # Dataset loading and preprocessing
│   ├── models/
│   │   └── baseline_models.py  # XGBoost, Random Forest, Deep Learning
│   ├── homomorphic/
│   │   └── he_ids.py          # Homomorphic encryption IDS
│   └── utils/
│       └── evaluation.py      # Evaluation metrics and visualization
├── models/saved/               # Saved trained models
├── results/                    # Evaluation results and plots
├── notebooks/                  # Jupyter notebooks (optional)
├── environment.yml             # Conda environment
├── main.py                     # Main execution script
└── README.md                   # This file
```

## Models

### Baseline Models

1. **Random Forest**: Ensemble method with 100 trees
2. **XGBoost**: Gradient boosting with GPU acceleration
3. **Deep Learning**: 5-layer neural network with dropout

### Homomorphic Encryption Model

- **Encryption Scheme**: CKKS (Complex numbers)
- **Library**: TenSEAL
- **Model**: Linear classifier with homomorphic evaluation
- **Approximation**: Polynomial approximation of sigmoid function

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Inference Time
- ROC Curves
- Confusion Matrices

## Results

Results are saved in the `results/` directory:
- `confusion_matrices.png`: Confusion matrices for all models
- `roc_curves.png`: ROC curves comparison
- `performance_comparison.png`: Bar charts of performance metrics
- `inference_time_comparison.png`: Inference time comparison
- `interactive_dashboard.html`: Interactive Plotly dashboard
- `detailed_report.txt`: Comprehensive evaluation report

## Performance Considerations

### Homomorphic Encryption Limitations

- **Computational Overhead**: 10-1000x slower than plaintext computation
- **Memory Usage**: Higher memory requirements for encrypted data
- **Precision Loss**: CKKS scheme introduces small numerical errors
- **Model Complexity**: Limited to simple models (linear, polynomial)

### Optimization Strategies

1. **Batch Processing**: Process multiple samples together
2. **Model Simplification**: Use linear models instead of complex ones
3. **Polynomial Approximation**: Replace non-polynomial functions
4. **Parameter Tuning**: Optimize encryption parameters

## GPU Acceleration

The system automatically detects and uses CUDA-compatible GPUs for:
- XGBoost training (`tree_method='gpu_hist'`)
- Deep learning model training (PyTorch CUDA)
- TenSEAL may benefit from GPU in some operations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{homomorphic-encryption-ids,
  title={Homomorphic Encryption for Intrusion Detection System},
  author={Quang-Vinh Dang},
  year={2025},
  url={https://github.com/vinhqdang/Homomorphic_Encryption_IDS}
}
```

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP.
2. Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). Homomorphic encryption for arithmetic of approximate numbers. ASIACRYPT.
3. TenSEAL: A Library for Encrypted Tensor Operations Using Homomorphic Encryption. https://github.com/OpenMined/TenSEAL

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU-only mode
2. **Dataset Not Found**: Ensure CIC-IDS2017 files are in `data/raw/`
3. **TenSEAL Installation**: May require specific compiler versions
4. **Memory Issues**: Use smaller sample sizes for homomorphic evaluation

### Performance Tips

1. Use `--sample-size` for quick testing
2. Skip baseline models if only testing homomorphic encryption
3. Monitor memory usage during homomorphic operations
4. Use GPU acceleration when available