import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread as skimage_imread
import imageio
from PIL import Image
import cv2
import torchvision.transforms as transforms
import logging
import psutil

# Set page configuration for Streamlit
st.set_page_config(layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Title and description of the app
st.title('Benchmarking PNG Image Reading Libraries')
st.write("""
This app benchmarks the performance of various Python libraries for reading PNG images. Simply input the folder path to your image dataset to generate a comprehensive performance comparison. Please ensure that all your images have the same dimensions before starting.
""")

# Function to get the dimensions of the first image in the dataset
def get_first_image_dimensions(image_file):
    image = Image.open(image_file)
    return image.width, image.height

# Function to get list of image files from the specified folder
def get_image_files(folder_path):
    extensions = ['.png']
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in extensions]

# Initialize session state if not already done
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = None
if 'detailed_read_times' not in st.session_state:
    st.session_state.detailed_read_times = {}

# Fragment to display stats per library
@st.experimental_fragment
def stats_per_library_fragment():
    # User selection for specific library stats
    selected_library = st.selectbox("Select a library to view specific stats", st.session_state.benchmark_results['Library'])
    selected_data = st.session_state.benchmark_results[st.session_state.benchmark_results['Library'] == selected_library].squeeze()

    st.write(f"### Stats for {selected_library}")
    st.info(f"**Average Read Time per Image:** {selected_data['read_time']:.4f} seconds")

    # Distribution plots for read times
    st.write(f"### Distribution of Read Times for {selected_library}")
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    sns.histplot(st.session_state.detailed_read_times[selected_library], ax=ax, kde=True, bins=20)
    ax.set_title(f'{selected_library} Read Time Distribution')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Benchmarking functions for each library
def benchmark_function(func, image_files):
    read_times = []
    for image_file in image_files:
        start_time = time.time()
        image = func(image_file)
        read_time = time.time() - start_time
        read_times.append(read_time)
    return read_times

def skimage_benchmark(image_file):
    image = skimage_imread(image_file)
    return image

def imageio_benchmark(image_file):
    image = imageio.imread(image_file)
    return image

def pillow_benchmark(image_file):
    image = Image.open(image_file)
    image.load()
    return image

def opencv_benchmark(image_file):
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    return image

def torchvision_benchmark(image_file):
    image = Image.open(image_file)
    tensor_image = transforms.ToTensor()(image)
    return tensor_image

def main():
    # User input for folder path
    folder_path = st.text_input('Enter the folder path of the image dataset:')

    if not folder_path or not os.path.isdir(folder_path):
        st.write('Please enter a valid folder path.')
        return

    # Run benchmark and display results
    if folder_path and os.path.isdir(folder_path) and st.session_state.benchmark_results is None:
        image_files = get_image_files(folder_path)
        st.write(f'Found {len(image_files)} PNG images in the folder.')
        logger.info(f'Found {len(image_files)} PNG images in the folder.')

        # Get dimensions of the first image
        st.session_state.width, st.session_state.height = get_first_image_dimensions(image_files[0])
       
        # Initialize results dictionary
        benchmarks = {
            'scikit-image 0.24.0': skimage_benchmark,
            'imageio 2.34.2': imageio_benchmark,
            'pillow 10.4.0': pillow_benchmark,
            'opencv-python-headless 4.10.0.84': opencv_benchmark,
            'torchvision 0.18.1': torchvision_benchmark,
        }

        results = {}
        detailed_read_times = {}
        with st.spinner('Running PNG benchmark...'):
            for name, func in benchmarks.items():
                logger.info(f'Benchmarking {name}...')
                read_times = benchmark_function(func, image_files)
                results[name] = {
                    'read_time': np.mean(read_times)
                }
                detailed_read_times[name] = read_times
                logger.info(f'{name} - Read: {results[name]["read_time"]:.4f} seconds per image')
        
        # Convert results to DataFrame for visualization
        df_results = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'Library'})

        # Save the results in session state
        st.session_state.benchmark_results = df_results
        st.session_state.detailed_read_times = detailed_read_times
        st.session_state.total_images = len(image_files)
        st.session_state.avg_file_size = np.mean([os.path.getsize(img) for img in image_files]) / 1024  # in KB
           
    else:
        df_results = st.session_state.benchmark_results

    if st.session_state.benchmark_results is not None:
        df_results = st.session_state.benchmark_results

        # Show the results in a dataframe
        st.dataframe(df_results, hide_index=True)

        # Display additional KPIs
        st.write("### Additional KPIs")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total images processed", st.session_state.total_images)
        col2.metric("Average file size per image", f"{st.session_state.avg_file_size:.2f} KB")
        col3.metric("Average Image Dimension", f"{st.session_state.width} x {st.session_state.height}") 
        
        stats_per_library_fragment()

        # Combined plot for read times
        st.write("### Combined Performance Metrics")
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Adjust the spacing between the subplots
        fig.subplots_adjust(wspace=0.4)

        # Create a list of simplified library names
        simple_library_names = ['scikit-image', 'imageio', 'pillow', 'opencv', 'torchvision']

        # Create a copy of df_results and replace the 'Library' column
        df_simplified = df_results.copy()
        df_simplified['Library'] = simple_library_names

        # Calculate images per second
        df_simplified['images_per_sec'] = 1 / df_simplified['read_time']

        sns.barplot(y='Library', x='images_per_sec', data=df_simplified, ax=ax, palette="viridis")
        ax.set_title('PNG Average Image Reading Performance')
        ax.set_ylabel('Library')
        ax.set_xlabel('Images per Second')

        st.pyplot(fig)

        st.subheader("CPU Info")
        # Get detailed CPU information
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_stats = psutil.cpu_stats()

        # Print the detailed CPU information
        st.write("### Processor Information")
        st.write(f"Physical cores: {cpu_count}")
        st.write(f"Total threads: {cpu_count_logical}")
        st.write(f"Max frequency: {cpu_freq.max:.2f}Mhz")
        st.write(f"Min frequency: {cpu_freq.min:.2f}Mhz")
        st.write(f"Current frequency: {cpu_freq.current:.2f}Mhz")
    else:
        st.write('Please enter a valid folder path.')

if __name__ == "__main__":
    main()
