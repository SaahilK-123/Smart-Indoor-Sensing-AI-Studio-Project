
# Smart IoT Indoor Sensing Project
### Project Description
#### *School of Data Engineering (SEDE) Studio Project*

The following project focuses on developing an AI that can track the movement of objects across a wide range of environments ranging from calm to active to chaotic with the primary focus being on room-level localisation activities.

## Table of Contents

1. [Project Description](#School-of-Data-Engineering-(SEDE)-Studio-Project)

2. [Software Technologies](#Technologies-(Software))

3. [Hardware Technologies](#Technologies-(Hardware))

4. [Industry Applications](#Used-By)

5. [Run Locally](#Run-Locally)

6. [Demo](#Demo)

7. [Authors](#Authors)

## Technologies (Software)

Scipy: 1.17.1

Openpyxl: 3.1.5

Torch: 2.10.0

Matplotlib: 3.10.8

Scikit-learn: 1.8.0

Torchvision: 0.25.0

## Technologies (Hardware)

1 x Router

2 x ESP32 Microcontrollers

1 x ESP32 Microcontroller w/ Antenna

3 x Laptops

## Used By

This project can be applied to a wide range of cases. Some examples of usages for this project include:

- Smart Homes / Environments
- Hospital Intensive Care Units
- Aged Care

## Run Locally

1. Clone the project:

```bash
  git clone https://github.com/SaahilK-123/Smart-Indoor-Sensing-AI-Studio-Project
```

2. Create a Python Virtual Environment:

```bash
  macOS / Linux: python3 -m venv <venv name>
  Windows: python -m venv <venv name>
```

3. Activate the Virtual Environment:

```bash
  macOS / Linux: source <venv name>/bin/activate
  Windows (Command Promot): <venv name>\Scripts\activate.bat
  Windows (PowerShell): .\\<venv name>\\Scripts\\Activate.ps1
```
4. Select interpreter and choose newly created Virtual Environment

5. Install all pre-requisites through the Virtual Environment Terminal:

```bash
  pip install -r requirements.txt
```

6. Flash the ESP32 Firmware using the tutorial provided

```bash
  File: 'ESP32 Firmware Flashing Tutorial.pdf'
```

7. Ensure that all correct drivers are installed using the provided guide

```bash
  File: 'ESP32 USB Driver Guide.pdf'
```

8. Ensure all variables are correctly assigned in python files

```bash
  File: 'csi_data_collection.py' requies variable 'serial_port' to be assigned a COM number. Refer to 'Step 7' to identify COM number
```

9. Space out the devices in a triangular shape and ensure they remain ~ 1-2 metres apart ensuring a wireless connection with the router for all devices and attach 1x ESP32 microcontroller per device

```bash
  The router should be kept at a moderate distance away from the devices so as to avoid affecting results
```

10. All devices are to now run the 'csi_data_collection.py' file simultaneously for an agreed upon time

```bash
  NOTE 1: The longer the experiment is run, the larger your csi files will become
  NOTE 2: Press 'CTRL + C' to end frame data collection and save results
```

11. Run 'read_h5.py' to convert frames to processable data.

```bash
  NOTE: The 'read_h5.py' file mentioned IS NOT the one under Feature Extraction and ML and is referring to the one visible in the main folder of the project (the one visible on the home screen of the project)
```

12. Manually label the data (using relevant time intervals [e.g. 1-min intervals for a 15-min experiment])

13. Repeat experiment (Steps 9-12) for all environment types

14. Change directory to 'Feature Extraction and ML' folder

```bash
  cd <project-folder-file-path>\Feature Extraction and ML
```

15. Run the 'read_h5.py' file to complete Feature Extraction

```bash
  python read_h5.py
```

16. Run the 'train_rd.py' file to Train and Evaluate Model

```bash
  python train_rd.py
```

## Authors

- [@ivany-sys](https://github.com/ivany-sys)
- [@jennifernguyen0512-cell](https://github.com/jennifernguyen0512-cell)
- [@ChhoungBun27](https://github.com/ChhoungBun27)