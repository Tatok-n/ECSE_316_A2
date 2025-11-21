# ECSE_316_A2

## Running the Project

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
