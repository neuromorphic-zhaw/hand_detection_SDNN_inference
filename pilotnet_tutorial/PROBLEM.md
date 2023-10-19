### Pilotnet SDNN inference tutorial 
https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/netx/pilotnet_snn/run.ipynb

### Requirements
- `lava_loihi-0.5.0`
- on `ncl-edu.research.intel-research.net`

### Problem
When running the pilot net for inference (`pilot_net_tutorial.py`) it fails with the following error message.

Output when running on Loihi2:
```bash
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_7635, shape : (99, 32, 24)
Conv  : Process_7638, shape : (49, 15, 36)
Conv  : Process_7641, shape : (24, 7, 48)
Conv  : Process_7644, shape : (22, 4, 64)
Conv  : Process_7647, shape : (20, 2, 64)
Dense : Process_7650, shape : (100,)
Dense : Process_7653, shape : (50,)
Dense : Process_7656, shape : (10,)
Dense : Process_7659, shape : (1,)

...

Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_8931, shape : (99, 32, 24)
Conv  : Process_8934, shape : (49, 15, 36)
Conv  : Process_8937, shape : (24, 7, 48)
Conv  : Process_8940, shape : (22, 4, 64)
Conv  : Process_8943, shape : (20, 2, 64)
Dense : Process_8946, shape : (100,)
Dense : Process_8949, shape : (50,)
Dense : Process_8952, shape : (10,)
Dense : Process_8955, shape : (1,)
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'pilot_net_tutorial' from '/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py'>' when searching ProcessModels for Process 'PilotNetEncoder'.
  warnings.warn(
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_8969, shape : (99, 32, 24)
Conv  : Process_8972, shape : (49, 15, 36)
Conv  : Process_8975, shape : (24, 7, 48)
Conv  : Process_8978, shape : (22, 4, 64)
Conv  : Process_8981, shape : (20, 2, 64)
Dense : Process_8984, shape : (100,)
Dense : Process_8987, shape : (50,)
Dense : Process_8990, shape : (10,)
Dense : Process_8993, shape : (1,)
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9005, shape : (99, 32, 24)
Conv  : Process_9008, shape : (49, 15, 36)
Conv  : Process_9011, shape : (24, 7, 48)
Conv  : Process_9014, shape : (22, 4, 64)
Conv  : Process_9017, shape : (20, 2, 64)
Dense : Process_9020, shape : (100,)
Dense : Process_9023, shape : (50,)
Dense : Process_9026, shape : (10,)
Dense : Process_9029, shape : (1,)
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'utils' from '/homes/glue/lava-env/lib/python3.8/site-packages/lava/proc/io/utils.py'>' when searching ProcessModels for Process 'SpikeDataloader'.
  warnings.warn(
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9041, shape : (99, 32, 24)
Conv  : Process_9044, shape : (49, 15, 36)
Conv  : Process_9047, shape : (24, 7, 48)
Conv  : Process_9050, shape : (22, 4, 64)
Conv  : Process_9053, shape : (20, 2, 64)
Dense : Process_9056, shape : (100,)
Dense : Process_9059, shape : (50,)
Dense : Process_9062, shape : (10,)
Dense : Process_9065, shape : (1,)
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9077, shape : (99, 32, 24)
Conv  : Process_9080, shape : (49, 15, 36)
Conv  : Process_9083, shape : (24, 7, 48)
Conv  : Process_9086, shape : (22, 4, 64)
Conv  : Process_9089, shape : (20, 2, 64)
Dense : Process_9092, shape : (100,)
Dense : Process_9095, shape : (50,)
Dense : Process_9098, shape : (10,)
Dense : Process_9101, shape : (1,)
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'utils' from '/homes/glue/lava-env/lib/python3.8/site-packages/lava/proc/io/utils.py'>' when searching ProcessModels for Process 'RingBuffer'.
  warnings.warn(
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9113, shape : (99, 32, 24)
Conv  : Process_9116, shape : (49, 15, 36)
Conv  : Process_9119, shape : (24, 7, 48)
Conv  : Process_9122, shape : (22, 4, 64)
Conv  : Process_9125, shape : (20, 2, 64)
Dense : Process_9128, shape : (100,)
Dense : Process_9131, shape : (50,)
Dense : Process_9134, shape : (10,)
Dense : Process_9137, shape : (1,)
Fatal Python error: Cannot recover from stack overflow.
Python runtime state: initialized

Current thread 0x00007fd46f39c740 (most recent call first):
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 162 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 170 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 170 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 170 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 143 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 608 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 107 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  ...
Aborted (core dumped)
```

Output when running on the CPU:
```bash
Conv  : Process_7563, shape : (99, 32, 24)
Conv  : Process_7566, shape : (49, 15, 36)
Conv  : Process_7569, shape : (24, 7, 48)
Conv  : Process_7572, shape : (22, 4, 64)
Conv  : Process_7575, shape : (20, 2, 64)
Dense : Process_7578, shape : (100,)
Dense : Process_7581, shape : (50,)
Dense : Process_7584, shape : (10,)
Dense : Process_7587, shape : (1,)
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.

...

Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_8931, shape : (99, 32, 24)
Conv  : Process_8934, shape : (49, 15, 36)
Conv  : Process_8937, shape : (24, 7, 48)
Conv  : Process_8940, shape : (22, 4, 64)
Conv  : Process_8943, shape : (20, 2, 64)
Dense : Process_8946, shape : (100,)
Dense : Process_8949, shape : (50,)
Dense : Process_8952, shape : (10,)
Dense : Process_8955, shape : (1,)
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'pilot_net_tutorial' from '/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py'>' when searching ProcessModels for Process 'PilotNetDecoder'.
  warnings.warn(
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_8967, shape : (99, 32, 24)
Conv  : Process_8970, shape : (49, 15, 36)
Conv  : Process_8973, shape : (24, 7, 48)
Conv  : Process_8976, shape : (22, 4, 64)
Conv  : Process_8979, shape : (20, 2, 64)
Dense : Process_8982, shape : (100,)
Dense : Process_8985, shape : (50,)
Dense : Process_8988, shape : (10,)
Dense : Process_8991, shape : (1,)
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9003, shape : (99, 32, 24)
Conv  : Process_9006, shape : (49, 15, 36)
Conv  : Process_9009, shape : (24, 7, 48)
Conv  : Process_9012, shape : (22, 4, 64)
Conv  : Process_9015, shape : (20, 2, 64)
Dense : Process_9018, shape : (100,)
Dense : Process_9021, shape : (50,)
Dense : Process_9024, shape : (10,)
Dense : Process_9027, shape : (1,)
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9039, shape : (99, 32, 24)
Conv  : Process_9042, shape : (49, 15, 36)
Conv  : Process_9045, shape : (24, 7, 48)
Conv  : Process_9048, shape : (22, 4, 64)
Conv  : Process_9051, shape : (20, 2, 64)
Dense : Process_9054, shape : (100,)
Dense : Process_9057, shape : (50,)
Dense : Process_9060, shape : (10,)
Dense : Process_9063, shape : (1,)
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'utils' from '/homes/glue/lava-env/lib/python3.8/site-packages/lava/proc/io/utils.py'>' when searching ProcessModels for Process 'RingBuffer'.
  warnings.warn(
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9075, shape : (99, 32, 24)
Conv  : Process_9078, shape : (49, 15, 36)
Conv  : Process_9081, shape : (24, 7, 48)
Conv  : Process_9084, shape : (22, 4, 64)
Conv  : Process_9087, shape : (20, 2, 64)
Dense : Process_9090, shape : (100,)
Dense : Process_9093, shape : (50,)
Dense : Process_9096, shape : (10,)
Dense : Process_9099, shape : (1,)
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
There are 9 layers in the network:
Conv  : Process_9111, shape : (99, 32, 24)
Conv  : Process_9114, shape : (49, 15, 36)
Conv  : Process_9117, shape : (24, 7, 48)
Conv  : Process_9120, shape : (22, 4, 64)
Conv  : Process_9123, shape : (20, 2, 64)
Dense : Process_9126, shape : (100,)
Dense : Process_9129, shape : (50,)
Dense : Process_9132, shape : (10,)
Dense : Process_9135, shape : (1,)
Fatal Python error: Cannot recover from stack overflow.
Python runtime state: initialized

Current thread 0x00007f1adcf3d740 (most recent call first):
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 162 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 170 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 170 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/ports/ports.py", line 170 in get_src_ports
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 143 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 158 in find_processes
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 608 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 109 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  ...
Aborted (core dumped)
```
