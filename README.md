# hand_detection_SDNN_inference
Perform inference on trained hand detection SDNN on both CPU and Loihi 2 neurocore.

### Requirements
```bash
pip list | grep lava
pip list | grep nx
``` 
- lava-dl                   0.4.0
- lava-dnf                  0.1.4
- lava-loihi                0.5.0
- lava-nc                   0.8.0
- lava-optimization         0.3.0
- nxcore                    2.4.0
- on `ncl-edu.research.intel-research.net`

### Infernece on Pilotnet SDNN tutorial
`pilotnet_tutorial/pilot_net_tutorial.py`
