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
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'pilot_net_tutorial' from '/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py'>' when searching ProcessModels for Process 'PilotNetDecoder'.
  warnings.warn(
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'utils' from '/homes/glue/lava-env/lib/python3.8/site-packages/lava/proc/io/utils.py'>' when searching ProcessModels for Process 'RingBuffer'.
  warnings.warn(
Running on oheogulch_20m
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Fatal Python error: Cannot recover from stack overflow.
Python runtime state: initialized

Current thread 0x00007f049e7e4740 (most recent call first):
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
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 123 in <module>
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
...
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py:868: UserWarning: Cannot import module '<module 'pilot_net_tutorial' from '/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py'>' when searching ProcessModels for Process 'PilotNetEncoder'.
  warnings.warn(
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Input     |  200|   66|    3|     |     |     |     |     |False|
|Conv      |   99|   32|   24| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   49|   15|   36| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   24|    7|   48| 3, 3| 2, 2| 0, 0| 1, 1|    1|False|
|Conv      |   22|    4|   64| 3, 3| 1, 2| 0, 1| 1, 1|    1|False|
|Conv      |   20|    2|   64| 3, 3| 1, 1| 0, 0| 1, 1|    1|False|
|Dense     |    1|    1|  100|     |     |     |     |     |False|
|Dense     |    1|    1|   50|     |     |     |     |     |False|
|Dense     |    1|    1|   10|     |     |     |     |     |False|
|Dense     |    1|    1|    1|     |     |     |     |     |False|
Fatal Python error: Cannot recover from stack overflow.
Python runtime state: initialized

Current thread 0x00007f4f11649740 (most recent call first):
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
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 124 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 124 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 124 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 124 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 971 in _expand_sub_proc_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1011 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 124 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 124 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 1005 in _map_proc_to_model
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 621 in __init__
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler.py", line 129 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 398 in compile
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 374 in create_runtime
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/core/process/process.py", line 350 in run
  File "/homes/glue/oasis_test/hand_detection_SDNN_inference/pilotnet_tutorial/pilot_net_tutorial.py", line 124 in <module>
  File "<frozen importlib._bootstrap>", line 219 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 848 in exec_module
  File "/homes/glue/lava-env/lib/python3.8/site-packages/lava/magma/compiler/compiler_graphs.py", line 853 in _find_proc_models
  ...
Aborted (core dumped)
```
