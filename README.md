# Computer Vision Libraries Benchmark

This project is a benchmarking application for evaluating the performance of various Python libraries in reading and processing images. Streamlit provides an interactive interface that allows users to compare different libraries based on their image reading times.

## Features

- Benchmark performance of multiple libraries:
  - `scikit-image`
  - `imageio`
  - `pillow`
  - `opencv`
  - `torchvision`
- Visualize results with histograms and bar plots.
- Detailed CPU information display.

## Prerequisites

Please make sure you have Conda installed. This project requires several Python libraries managed through Conda to avoid environment conflicts.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/computer-vision-library-benchmark.git
   cd computer-vision-library-benchmark

2. **Create a Conda environment:**
   ```bash
   conda create --name cv_benchmark python=3.9
