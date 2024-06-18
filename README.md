# Millimeter Wave Data Capture Scripts

## Directory Structure
```
mmHER
└── data_collection_pipeline
    └── configuration.py
    └── dataCaptureScript.lua
    └── data_capture.py
    └── pcd_generation.py
    └── steaming.py
```

## Description

### 1. Real-time mmWave Data Capturing

To enable real-time data capturing, the first step is to start the radar in `Windows OS`. The softwave used in the work is `mmWave Studio 2.0.0.2`. Just copy `dataCaptureScript.lua` script to `C:\\ti\\mmwave_studio_02_01_00_00\\mmWaveStudio\\Scripts` directory and load the lua script and click the `Run!` button on the left bottom corner of the softwave as show in the image.
![mmWave Studio](https://raw.githubusercontent.com/HavocFiXer/mmMesh/master/mmwave_studio.png)
Then the radar will keep sending the chirps and steaming the data through networking cables.

The methods provided in `steaming.py` will automatically collect the packets from the radar and assemble them into frames. It also allows you to access these frames using `getFrame` method. As an example, if you want to capture the data from the mmWave radar for 5 mins and store them, just run:
``` bash
python data_capture.py 5
```

### 2. Point Cloud Generation from Binary mmWave Data

The methods in `pcd_generation.py` allows you to generate the point cloud from the binary mmWave radar. This method also computes `range-FFT` and `doppler-FFT` parameters in the intermediate steps. Suppose you want to generate the point cloud data from `test.bin` for 10 frames, try:
``` bash
python pcd_generation.py test.bin 10
```

**Note**: to successfully read the data from the binary file, you should change the radar configuration to generate the binary file without the packet head.
