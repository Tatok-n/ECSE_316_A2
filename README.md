# ECSE_316_A2

Implemented by: Mathis Bélanger (261049961) & Tarek Namani (261085655)

## Running the Project

Activate venv:

```bash
source venv/bin/activate
```

Windows:

```bash
.\venv\Scripts\Activate.ps1
```

### Mode 1: Visualization [Default]

Displays the original image next to its Log-Magnitude FFT heatmap.

```bash
python3 fft.py -m 1 -i moonlanding.png
```

### Mode 2: Denoising

Use various frequency filters to remove noise from an image

```bash
python3 fft.py -m 2 -i moonlanding.png
```

Configuration: Edit DENOISE_METHOD and DENOISE_VALUE in fft.py to change filter types.

### Mode 3: Compression

Shows the image reconstructed at 5 different compression levels (up to 99.9%).

```bash
python3 fft.py -m 3 -i moonlanding.png
```

### Mode 4: Runtime Analysis

Plots the performance of your recursive FFT vs. the Naïve DFT over various matrix sizes.

```bash
python3 fft.py -m 4
```

### Run Unit Tests

```bash
python3 tests.py
```

## Setting up the Python Environment

This project uses a Python virtual environment to manage dependencies.

### 1. Create a virtual environment

Run the following command in the project folder:

```bash
python3 -m venv venv
```

This will create a folder called venv containing a self-contained Python environment.

### 2. Activate the virtual environment

mac/linux:

```bash
source venv/bin/activate
```

Windows:

```bash
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

You now have a valid python environment to run the project

### 4. Deactivate the virtual environment

Once you are done:

```bash
deactivate
```

### 5. Notes

- See requirement.txt for detailed dependencies

- If you update dependencies, remember to regenerate requirements.txt:

```bash
pip freeze > requirements.txt
```
