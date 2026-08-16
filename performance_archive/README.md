# Performance Archive

This folder contains the data and scripts necessary to generate the performance and comparison charts for the evaluation framework.

## Prerequisites

To generate the charts, you need to install the required Python libraries. You can install them using `pip`:

```bash
pip install pandas matplotlib seaborn numpy
```

*(Note: These dependencies are specific to chart generation and may not be fully covered by the main `requirements.txt` file.)*

## Generating the Charts

There are two main scripts in this directory that read the CSV files and output PNG images of the charts.

### 1. Combined Performance Metrics Chart
This script generates a 3-panel chart comparing global architecture performance, SLM pairwise performance (Static vs EPD), and relative memory usage.

```bash
python combined_charts.py
```
**Output:** The chart will be saved to `charts/04_combined_performance_metrics.png`.

### 2. Deployment Configuration Comparison Chart
This script generates a horizontal bar chart comparing the win rates across different deployment configurations.

```bash
python config_comparison_chart.py
```
**Output:** The chart will be saved to `charts/deployment_configuration_comparison.png`.

## Folder Contents
- `*.csv`: The raw metrics data derived from the evaluation runs.
- `*.py`: The chart generation scripts.
- `charts/`: The output directory where generated images are saved.
