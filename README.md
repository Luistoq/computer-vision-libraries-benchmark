
# Computer Vision Libraries Benchmark

This project is a benchmarking application for evaluating the performance of various Python libraries in reading and processing images. Using Streamlit, it provides an interactive interface to compare different libraries based on their image reading times.

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

Ensure you have Conda installed. This project requires several Python libraries, which are managed through Conda to avoid environment conflicts.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/computer-vision-libraries-benchmark.git
   cd computer-vision-libraries-benchmark
   ```

2. **Create a Conda environment:**

   ```bash
   conda create --name cv_benchmark python=3.10
   ```

3. **Activate the Conda environment:**

   ```bash
   conda activate cv_benchmark
   ```

4. **Install dependencies:**

   ```bash
   conda install -c conda-forge streamlit numpy pandas matplotlib seaborn scikit-image imageio pillow opencv-python-headless torchvision psutil
   ```

   This command installs all required libraries from the Conda-Forge channel.

## Usage

1. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

2. **Open your web browser and navigate to the local Streamlit server, typically `http://localhost:8501`.**

3. **Enter the path to your folder containing PNG images in the input field.**

4. **View the benchmarking results, including average read times per library, distribution plots, and CPU information.

## Example

Upon running the app and providing a folder path, you will see:

- A comparison of average read times for each library.
- Distribution plots for reading times.
- Additional KPIs like total images processed, average file size, and image dimensions.
- A bar plot showing the average image reading performance in images per second.
- Detailed CPU information.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [scikit-image](https://scikit-image.org/)
- [imageio](https://imageio.github.io/)
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [OpenCV](https://opencv.org/)
- [torchvision](https://pytorch.org/vision/)
- [seaborn](https://seaborn.pydata.org/)
- [psutil](https://psutil.readthedocs.io/)

---
