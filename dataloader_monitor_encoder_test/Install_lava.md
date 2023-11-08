# Install lava from source
- on `ncl-edu.research.intel-research.net`
## 1. Install lava-nc from source
### Create new venv
```bash
~/$ python3 -m venv lava_env2
~/$ source lava_env2/bin/activate
```

### Clone repo
```bash
(lava_env2) ~/$ mkdir lava_git_repos
(lava_env2) ~/$ cd lava_git_repos/
(lava_env2) ~/lava_git_repos$ git clone git@github.com:lava-nc/lava.git
(lava_env2) ~/lava_git_repos$ cd lava/
(lava_env2) ~/lava_git_repos/lava$ git checkout main
Already on 'main'
Your branch is up to date with 'origin/main'.
```

### Install
```bash
(lava_env2) ~/lava_git_repos/lava$ ./utils/githook/install-hook.sh
(lava_env2) ~/lava_git_repos/lava$ poetry install
Installing dependencies from lock file

Package operations: 165 installs, 0 updates, 0 removals

  • Installing attrs (23.1.0)
  • Installing rpds-py (0.9.2)
  • Installing zipp (3.16.2)
  • Installing importlib-resources (6.0.0)
  • Installing pycparser (2.21)
  • Installing referencing (0.30.0)
  • Installing certifi (2023.7.22)
  • Installing cffi (1.15.1)
  • Installing charset-normalizer (3.2.0)
  • Installing idna (3.4)
  • Installing jsonschema-specifications (2023.7.1)
  • Installing markupsafe (2.1.3)
  • Installing pkgutil-resolve-name (1.3.10)
  • Installing platformdirs (3.9.1)
  • Installing pytz (2023.3)
  • Installing six (1.16.0)
  • Installing smmap (5.0.0)
  • Installing traitlets (5.9.0)
  • Installing urllib3 (1.26.18)
  • Installing alabaster (0.7.13)
  • Installing asttokens (2.2.1)
  • Installing babel (2.12.1)
  • Installing cryptography (41.0.4)
  • Installing docutils (0.17.1)
  • Installing executing (1.2.0)
  • Installing fastjsonschema (2.18.0)
  • Installing gitdb (4.0.10)
  • Installing imagesize (1.4.1)
  • Installing importlib-metadata (6.8.0)
  • Installing jeepney (0.8.0)
  • Installing jinja2 (3.0.3)
  • Installing jsonschema (4.18.4)
  • Installing jupyter-core (5.3.1)
  • Installing mccabe (0.6.1)
  • Installing more-itertools (10.0.0)
  • Installing packaging (23.1)
  • Installing parso (0.8.3)
  • Installing pbr (5.11.1)
  • Installing ptyprocess (0.7.0)
  • Installing pure-eval (0.2.2)
  • Installing pycodestyle (2.8.0)
  • Installing pyflakes (2.4.0)
  • Installing pygments (2.15.1)
  • Installing python-dateutil (2.8.2)
  • Installing pyzmq (25.1.0)
  • Installing requests (2.31.0)
  • Installing snowballstemmer (2.2.0)
  • Installing sphinxcontrib-applehelp (1.0.4)
  • Installing sphinxcontrib-devhelp (1.0.2)
  • Installing sphinxcontrib-htmlhelp (2.0.1)
  • Installing sphinxcontrib-jsmath (1.0.1)
  • Installing sphinxcontrib-qthelp (1.0.3)
  • Installing sphinxcontrib-serializinghtml (1.1.5)
  • Installing tomli (2.0.1)
  • Installing tornado (6.3.3)
  • Installing wcwidth (0.2.6)
  • Installing backcall (0.2.0)
  • Installing crashtest (0.4.1)
  • Installing decorator (5.1.1)
  • Installing distlib (0.3.7)
  • Installing exceptiongroup (1.1.2)
  • Installing filelock (3.12.2)
  • Installing flake8 (4.0.1)
  • Installing gitpython (3.1.37)
  • Installing iniconfig (2.0.0)
  • Installing jaraco-classes (3.3.0)
  • Installing jedi (0.18.2)
  • Installing jupyter-client (8.3.0)
  • Installing linecache2 (1.0.0)
  • Installing lockfile (0.12.2)
  • Installing matplotlib-inline (0.1.6)
  • Installing msgpack (1.0.5)
  • Installing nbformat (5.9.1)
  • Installing numpy (1.24.4)
  • Installing pexpect (4.8.0)
  • Installing pickleshare (0.7.5)
  • Installing pluggy (1.2.0)
  • Installing poetry-core (1.6.1)
  • Installing pyproject-hooks (1.0.0)
  • Installing prompt-toolkit (3.0.39)
  • Installing rapidfuzz (2.15.1)
  • Installing secretstorage (3.3.3)
  • Installing pyyaml (6.0.1)
  • Installing soupsieve (2.4.1)
  • Installing sphinx (4.5.0)
  • Installing stack-data (0.6.2)
  • Installing stevedore (5.1.0)
  • Installing typing-extensions (4.7.1)
  • Installing webencodings (0.5.1)
  • Installing argparse (1.4.0)
  • Installing bandit (1.7.4)
  • Installing beautifulsoup4 (4.12.2)
  • Installing bleach (6.0.0)
  • Installing build (0.10.0)
  • Installing cachecontrol (0.12.14)
  • Installing cleo (2.0.1)
  • Installing colorama (0.4.6)
  • Installing comm (0.1.3)
  • Installing contourpy (1.1.0)
  • Installing coverage (6.5.0)
  • Installing cycler (0.11.0)
  • Installing debugpy (1.6.7)
  • Installing defusedxml (0.7.1)
  • Installing dulwich (0.21.5)
  • Installing entrypoints (0.4)
  • Installing eradicate (2.3.0)
  • Installing flake8-plugin-utils (1.3.3)
  • Installing flake8-polyfill (1.0.2)
  • Installing fonttools (4.41.1)
  • Installing html5lib (1.1)
  • Installing installer (0.7.0)
  • Installing ipython (8.12.2)
  • Installing isort (5.12.0)
  • Installing joblib (1.3.2)
  • Installing jupyterlab-pygments (0.2.2)
  • Installing keyring (23.13.1)
  • Installing kiwisolver (1.4.4)
  • Installing mistune (2.0.5)
  • Installing nbclient (0.8.0)
  • Installing nest-asyncio (1.5.6)
  • Installing pandocfilters (1.5.0)
  • Installing pillow (10.0.1)
  • Installing pkginfo (1.9.6)
  • Installing poetry-plugin-export (1.4.0)
  • Installing psutil (5.9.5)
  • Installing pydocstyle (6.3.0)
  • Installing pyparsing (3.0.9)
  • Installing pytest (7.4.0)
  • Installing requests-toolbelt (1.0.0)
  • Installing scipy (1.10.1)
  • Installing shellingham (1.5.0.post1)
  • Installing sphinxcontrib-jquery (4.1)
  • Installing threadpoolctl (3.2.0)
  • Installing tinycss2 (1.2.1)
  • Installing toml (0.10.2)
  • Installing tomlkit (0.11.8)
  • Installing traceback2 (1.4.0)
  • Installing trove-classifiers (2023.7.6)
  • Installing virtualenv (20.24.2)
  • Installing asteval (0.9.31)
  • Installing autopep8 (1.6.0)
  • Installing darglint (1.8.1)
  • Installing flake8-bandit (3.0.0)
  • Installing flake8-bugbear (22.12.6)
  • Installing flake8-builtins (1.5.3)
  • Installing flake8-comprehensions (3.14.0)
  • Installing flake8-docstrings (1.7.0)
  • Installing flake8-eradicate (1.4.0)
  • Installing flake8-isort (4.2.0)
  • Installing flake8-mutable (1.2.0)
  • Installing flake8-pytest-style (1.7.2)
  • Installing flake8-spellcheck (0.25.0)
  • Installing flakeheaven (3.3.0)
  • Installing ipykernel (6.24.0)
  • Installing matplotlib (3.7.2)
  • Installing nbconvert (7.2.10)
  • Installing networkx (2.8.7)
  • Installing pandas (1.5.3)
  • Installing pep8-naming (0.12.1)
  • Installing poetry (1.5.1)
  • Installing pytest-cov (3.0.0)
  • Installing scikit-learn (1.3.1)
  • Installing sphinx-rtd-theme (1.2.2)
  • Installing sphinx-tabs (3.4.0)
  • Installing unittest2 (1.1.0)

Installing the current project: lava-nc (0.8.0.dev0)
```

```bash
(lava_env2) ~/lava_git_repos/lava$ pip list | grep lava
lava-nc                       0.8.0.dev0 
```
## Run test
```bash
(lava_env2) ~/lava_git_repos/lava$ pytest

...

=============================================================================================================== warnings summary ===============================================================================================================
../../lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:256
  /homes/glue/lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:256: PytestDeprecationWarning: The hookimpl CovPlugin.pytest_configure_node uses old-style configuration options (marks or attributes).
  Please use the pytest.hookimpl(optionalhook=True) decorator instead
   to configure the hooks.
   See https://docs.pytest.org/en/latest/deprecations.html#configuring-hook-specs-impls-using-markers
    def pytest_configure_node(self, node):

../../lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:265
  /homes/glue/lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:265: PytestDeprecationWarning: The hookimpl CovPlugin.pytest_testnodedown uses old-style configuration options (marks or attributes).
  Please use the pytest.hookimpl(optionalhook=True) decorator instead
   to configure the hooks.
   See https://docs.pytest.org/en/latest/deprecations.html#configuring-hook-specs-impls-using-markers
    def pytest_testnodedown(self, node, error):

tests/lava/proc/io/test_dataloader.py:26
  /homes/glue/lava_git_repos/lava/tests/lava/proc/io/test_dataloader.py:26: PytestCollectionWarning: cannot collect test class 'TestRunConfig' because it has a __init__ constructor (from: tests/lava/proc/io/test_dataloader.py)
    class TestRunConfig(RunConfig):

tests/lava/proc/io/test_source_sink.py:18
  /homes/glue/lava_git_repos/lava/tests/lava/proc/io/test_source_sink.py:18: PytestCollectionWarning: cannot collect test class 'TestRunConfig' because it has a __init__ constructor (from: tests/lava/proc/io/test_source_sink.py)
    class TestRunConfig(RunConfig):

tests/lava/proc/io/test_extractor.py::TestPyLoihiVarWireModel::test_receive_data_receive_empty_blocking
  /homes/glue/lava_env2/lib/python3.8/site-packages/_pytest/threadexception.py:73: PytestUnhandledThreadExceptionWarning: Exception in thread src.send
  
  Traceback (most recent call last):
    File "/usr/lib/python3.8/threading.py", line 932, in _bootstrap_inner
      self.run()
    File "/usr/lib/python3.8/threading.py", line 870, in run
      self._target(*self._args, **self._kwargs)
    File "/homes/glue/lava_git_repos/lava/src/lava/magma/compiler/channels/pypychannel.py", line 114, in _ack_callback
      self._semaphore.release()
    File "/usr/lib/python3.8/threading.py", line 489, in release
      raise ValueError("Semaphore released too many times")
  ValueError: Semaphore released too many times
  
    warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))

tests/lava/proc/io/test_injector.py::TestPyLoihiInjectorModel::test_send_data_send_full_non_blocking_drop
  /homes/glue/lava_git_repos/lava/src/lava/proc/io/utils.py:57: UserWarning: Send buffer is full. Dropping items ...
    warnings.warn("Send buffer is full. Dropping items ...")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

---------- coverage: platform linux, python 3.8.10-final-0 -----------
Name                                                                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------------------------------------------------
src/lava/magma/compiler/builders/channel_builder.py                                    94      4    96%   24-25, 314-318
src/lava/magma/compiler/builders/interfaces.py                                         44      2    95%   47, 127
src/lava/magma/compiler/builders/py_builder.py                                        148     10    93%   63, 115-123, 197, 285, 300
src/lava/magma/compiler/builders/runtimeservice_builder.py                             70      4    94%   63, 105-113
src/lava/magma/compiler/channel_map.py                                                 66      2    97%   61, 127
src/lava/magma/compiler/channels/interfaces.py                                         56     11    80%   18, 23, 28, 33, 37, 41, 44, 50, 56, 63, 68
src/lava/magma/compiler/channels/pypychannel.py                                       206     19    91%   24, 79, 87, 116, 118, 136, 139, 168-169, 173-177, 233, 237, 241, 247, 274-275
src/lava/magma/compiler/channels/watchdog.py                                          131     56    57%   31-43, 48-54, 79-83, 86-87, 94, 98, 106, 109, 112, 119-126, 130, 134, 141-145, 148-159, 162-170, 214-215, 219
src/lava/magma/compiler/compiler.py                                                   192     33    83%   17-22, 32, 36, 281, 283, 327, 329, 340, 345-346, 495-497, 509-532, 677
src/lava/magma/compiler/compiler_graphs.py                                            380     77    80%   27, 103, 146, 151, 170-180, 223, 256-257, 260-265, 291, 298, 304, 324-335, 357-364, 433, 459, 540-553, 568, 632, 647-648, 692, 704, 712, 724, 730, 742, 755, 829, 834-841, 894-895, 902, 933, 935, 1033
src/lava/magma/compiler/compiler_utils.py                                              26      6    77%   12, 34-39
src/lava/magma/compiler/exceptions.py                                                   8      4    50%   8-9, 14-18
src/lava/magma/compiler/executable.py                                                  21      3    86%   15-18
src/lava/magma/compiler/mappable_interface.py                                           6      0   100%
src/lava/magma/compiler/mapper.py                                                      91     66    27%   21, 53-66, 83-115, 119-169
src/lava/magma/compiler/node.py                                                        38      6    84%   11, 36, 54-57
src/lava/magma/compiler/subcompilers/address.py                                        10      0   100%
src/lava/magma/compiler/subcompilers/channel_builders_factory.py                      139     68    51%   21, 85-87, 90, 92-105, 107-123, 151-182, 195-198, 203-206, 225, 259-290
src/lava/magma/compiler/subcompilers/channel_map_updater.py                            29      1    97%   62
src/lava/magma/compiler/subcompilers/constants.py                                      11      0   100%
src/lava/magma/compiler/subcompilers/interfaces.py                                     22      2    91%   53, 59
src/lava/magma/compiler/subcompilers/py/pyproc_compiler.py                            150     30    80%   48, 52, 56, 64-66, 69-72, 75-77, 120, 161, 163, 166-194
src/lava/magma/compiler/utils.py                                                       81     11    86%   67-70, 86-96
src/lava/magma/compiler/var_model.py                                                  151     12    92%   107, 122-128, 179, 194-200
src/lava/magma/core/callback_fx.py                                                     39     17    56%   27, 30, 33, 36, 50, 56, 66-72, 78-79, 85-86
src/lava/magma/core/decorator.py                                                       57      6    89%   29, 32, 42, 85, 90, 165
src/lava/magma/core/learning/constants.py                                              25      0   100%
src/lava/magma/core/learning/learning_rule.py                                         205      3    99%   218, 383, 480
src/lava/magma/core/learning/learning_rule_applier.py                                  87      7    92%   32, 119-120, 132, 214, 240-241
src/lava/magma/core/learning/product_series.py                                        235     32    86%   163, 351-363, 596, 605-606, 637-638, 653-662, 688-689, 693, 697, 704, 712, 717-718, 721, 724, 730, 746-750
src/lava/magma/core/learning/random.py                                                 38      1    97%   16
src/lava/magma/core/learning/string_symbols.py                                         24      0   100%
src/lava/magma/core/learning/symbolic_equation.py                                     269      3    99%   50, 491, 674
src/lava/magma/core/learning/utils.py                                                  14      2    86%   47-48
src/lava/magma/core/model/interfaces.py                                                20      0   100%
src/lava/magma/core/model/model.py                                                     23      6    74%   12, 91-95
src/lava/magma/core/model/py/connection.py                                            529     20    96%   213, 252, 266, 307, 311, 418, 422, 449, 472, 1052-1055, 1066, 1168, 1546-1549, 1559
src/lava/magma/core/model/py/model.py                                                 331     29    91%   153-160, 168, 197-198, 222, 346, 352, 358, 393, 413-419, 594, 610-611, 613-614, 709, 757, 760
src/lava/magma/core/model/py/neuron.py                                                 39      0   100%
src/lava/magma/core/model/py/ports.py                                                 247     50    80%   297, 310, 364, 377-381, 385, 393, 397, 405, 409, 471, 475, 480, 505-518, 530, 538, 660, 679, 746, 750, 758, 762, 770, 774, 879, 889, 920, 928, 932, 936, 944, 948, 952, 960, 964, 968
src/lava/magma/core/model/py/type.py                                                    8      0   100%
src/lava/magma/core/model/spike_type.py                                                 9      0   100%
src/lava/magma/core/model/sub/model.py                                                 21      4    81%   67-70
src/lava/magma/core/process/connection.py                                              41      0   100%
src/lava/magma/core/process/interfaces.py                                              47      5    89%   11, 20, 40, 54, 64
src/lava/magma/core/process/message_interface_enum.py                                   3      0   100%
src/lava/magma/core/process/neuron.py                                                  17      0   100%
src/lava/magma/core/process/ports/connection_config.py                                 29      0   100%
src/lava/magma/core/process/ports/exceptions.py                                        28      0   100%
src/lava/magma/core/process/ports/ports.py                                            312     18    94%   112, 352, 571, 607, 619, 652, 659, 666, 722, 747, 783, 821, 842, 846-850, 865, 876
src/lava/magma/core/process/ports/reduce_ops.py                                         3      0   100%
src/lava/magma/core/process/process.py                                                253     20    92%   23, 186, 294, 315-318, 449, 468, 476, 554, 559, 623-625, 628-637
src/lava/magma/core/process/variable.py                                                90      7    92%   74, 78, 95, 137, 188, 218, 223
src/lava/magma/core/resources.py                                                       35      0   100%
src/lava/magma/core/run_conditions.py                                                  11      0   100%
src/lava/magma/core/run_configs.py                                                    106     32    70%   17, 30-31, 90, 192, 245, 264-292, 297, 321, 338, 389-391, 399-409, 413, 451-453, 460-471, 475
src/lava/magma/core/sync/domain.py                                                      2      0   100%
src/lava/magma/core/sync/protocol.py                                                    5      0   100%
src/lava/magma/core/sync/protocols/async_protocol.py                                   11      0   100%
src/lava/magma/core/sync/protocols/loihi_protocol.py                                   25      0   100%
src/lava/magma/runtime/message_infrastructure/factory.py                                8      1    88%   20
src/lava/magma/runtime/message_infrastructure/message_infrastructure_interface.py      18      3    83%   7-9
src/lava/magma/runtime/message_infrastructure/multiprocessing.py                       85     15    82%   7-9, 25-26, 88, 137-146
src/lava/magma/runtime/message_infrastructure/shared_memory_manager.py                 20      0   100%
src/lava/magma/runtime/mgmt_token_enums.py                                             23      0   100%
src/lava/magma/runtime/runtime.py                                                     301     49    84%   27, 143, 177, 224-237, 283-285, 345, 362, 384-387, 402-420, 434-437, 459-461, 465-466, 477, 494, 508-511, 516-518, 522-523, 563, 567
src/lava/magma/runtime/runtime_services/enums.py                                       22      0   100%
src/lava/magma/runtime/runtime_services/interfaces.py                                  23      7    70%   27, 32-34, 38, 41-42
src/lava/magma/runtime/runtime_services/runtime_service.py                            315     28    91%   146, 197-198, 238, 260-264, 274, 286, 320, 383, 388, 440-441, 451-452, 481, 500, 504, 508, 511, 515, 517, 519, 522-524
src/lava/proc/bit_check/models.py                                                      52     10    81%   52, 60-77, 95, 98, 101
src/lava/proc/bit_check/process.py                                                     29      0   100%
src/lava/proc/clp/novelty_detector/models.py                                           41      0   100%
src/lava/proc/clp/novelty_detector/process.py                                          10      0   100%
src/lava/proc/clp/nsm/models.py                                                        65      0   100%
src/lava/proc/clp/nsm/process.py                                                       24      0   100%
src/lava/proc/clp/prototype_lif/models.py                                              49      0   100%
src/lava/proc/clp/prototype_lif/process.py                                             10      0   100%
src/lava/proc/conv/process.py                                                          47      4    91%   119, 127, 137, 142
src/lava/proc/conv/utils.py                                                           152     64    58%   57, 61, 83-84, 136-143, 148-155, 194-218, 275, 280, 385-433, 449, 452
src/lava/proc/dense/models.py                                                         152      3    98%   284, 332, 341
src/lava/proc/dense/process.py                                                         45      1    98%   263
src/lava/proc/graded/models.py                                                         98      3    97%   108-111
src/lava/proc/graded/process.py                                                        41      1    98%   18
src/lava/proc/io/__init__.py                                                            2      0   100%
src/lava/proc/io/dataloader.py                                                        105      2    98%   126, 201
src/lava/proc/io/encoder.py                                                           167    116    31%   80-99, 103, 120-126, 137-141, 146-149, 166-175, 182-187, 191-198, 202-257, 262-282, 285-295
src/lava/proc/io/extractor.py                                                         106      6    94%   113-116, 240-243
src/lava/proc/io/injector.py                                                           58      3    95%   98-101
src/lava/proc/io/reset.py                                                              42     11    74%   39-44, 47-49, 60, 63-65
src/lava/proc/io/sink.py                                                               79      0   100%
src/lava/proc/io/source.py                                                             33      0   100%
src/lava/proc/io/utils.py                                                              68      0   100%
src/lava/proc/learning_rules/r_stdp_learning_rule.py                                   17      0   100%
src/lava/proc/learning_rules/stdp_learning_rule.py                                     15      0   100%
src/lava/proc/lif/models.py                                                           244      0   100%
src/lava/proc/lif/process.py                                                           49      3    94%   341, 343, 415
src/lava/proc/monitor/models.py                                                        27      0   100%
src/lava/proc/monitor/process.py                                                       87      5    94%   261-268
src/lava/proc/prodneuron/models.py                                                     26      0   100%
src/lava/proc/prodneuron/process.py                                                    17      0   100%
src/lava/proc/resfire/models.py                                                        29      0   100%
src/lava/proc/resfire/process.py                                                       24      0   100%
src/lava/proc/rf/models.py                                                             70      0   100%
src/lava/proc/rf/process.py                                                            28      0   100%
src/lava/proc/rf_iz/models.py                                                          32      0   100%
src/lava/proc/rf_iz/process.py                                                          3      0   100%
src/lava/proc/sdn/models.py                                                           159      9    94%   199-202, 225-231
src/lava/proc/sdn/process.py                                                           54      0   100%
src/lava/proc/sparse/models.py                                                        163      4    98%   252, 304, 352, 361
src/lava/proc/sparse/process.py                                                        55      2    96%   172, 283
src/lava/proc/spiker/models.py                                                         19      0   100%
src/lava/proc/spiker/process.py                                                        12      0   100%
src/lava/utils/loihi.py                                                                33      0   100%
src/lava/utils/plots.py                                                                29      0   100%
src/lava/utils/serialization.py                                                        31      2    94%   117, 125
src/lava/utils/slurm.py                                                                89      0   100%
src/lava/utils/sparse.py                                                               11      0   100%
src/lava/utils/weightutils.py                                                         108      2    98%   81-82
-----------------------------------------------------------------------------------------------------------------
TOTAL                                                                                9129   1043    89%

Required test coverage of 65.0% reached. Total coverage: 88.57%
============================================================================================ 639 passed, 6 skipped, 6 warnings in 823.67s (0:13:43) ============================================================================================
```

## 2. Install lava-dl from source

### Clone repo
```bash
(lava_env2) ~/lava_git_repos$ git clone git@github.com:lava-nc/lava-dl.git
(lava_env2) ~/lava_git_repos$ cd lava-dl/
(lava_env2) ~/lava_git_repos/lava-dl$ git checkout main
Already on 'main'
Your branch is up to date with 'origin/main'.
```

### Install
```bash
(lava_env2) ~/lava_git_repos/lava$ poetry install

Installing dependencies from lock file

Package operations: 28 installs, 47 updates, 0 removals

  • Updating zipp (3.16.2 -> 3.17.0)
  • Updating cffi (1.15.1 -> 1.16.0)
  • Updating importlib-resources (6.0.0 -> 6.1.0)
  • Updating platformdirs (3.9.1 -> 3.11.0)
  • Installing pyrsistent (0.19.3)
  • Updating smmap (5.0.0 -> 5.0.1)
  • Updating traitlets (5.9.0 -> 5.11.2)
  • Updating charset-normalizer (3.2.0 -> 3.3.0): Installing...
  • Installing cmake (3.27.7): Installing...
  • Updating fastjsonschema (2.18.0 -> 2.18.1)
  • Updating filelock (3.12.2 -> 3.12.4): Installing...
  • Downgrading jsonschema (4.18.4 -> 4.17.3): Installing...
  • Updating jupyter-core (5.3.1 -> 5.4.0): Installing...
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0): Installing...
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Downgrading jsonschema (4.18.4 -> 4.17.3): Installing...
  • Updating jupyter-core (5.3.1 -> 5.4.0): Installing...
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating filelock (3.12.2 -> 3.12.4)
  • Downgrading jsonschema (4.18.4 -> 4.17.3): Installing...
  • Updating jupyter-core (5.3.1 -> 5.4.0): Installing...
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Installing cmake (3.27.7): Installing...
  • Updating fastjsonschema (2.18.0 -> 2.18.1)
  • Updating filelock (3.12.2 -> 3.12.4)
  • Downgrading jsonschema (4.18.4 -> 4.17.3): Installing...
  • Updating jupyter-core (5.3.1 -> 5.4.0): Installing...
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating charset-normalizer (3.2.0 -> 3.3.0)
  • Installing cmake (3.27.7): Installing...
  • Updating fastjsonschema (2.18.0 -> 2.18.1)
  • Updating filelock (3.12.2 -> 3.12.4)
  • Downgrading jsonschema (4.18.4 -> 4.17.3): Installing...
  • Updating jupyter-core (5.3.1 -> 5.4.0): Installing...
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating jupyter-core (5.3.1 -> 5.4.0): Installing...
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Downgrading jsonschema (4.18.4 -> 4.17.3)
  • Updating jupyter-core (5.3.1 -> 5.4.0): Installing...
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1): Installing...
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
       519│ 
       520│ 
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating pyzmq (25.1.0 -> 25.1.1)
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
       519│ 
       520│ 

The following error occurred when trying to handle this error:


  EnvCommandError

  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1)
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating jupyter-core (5.3.1 -> 5.4.0)
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1)
  • Updating setuptools (44.0.0 -> 68.2.2): Installing...
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating setuptools (44.0.0 -> 68.2.2)
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Updating fastjsonschema (2.18.0 -> 2.18.1)
  • Updating filelock (3.12.2 -> 3.12.4)
  • Downgrading jsonschema (4.18.4 -> 4.17.3)
  • Updating jupyter-core (5.3.1 -> 5.4.0)
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1)
  • Updating setuptools (44.0.0 -> 68.2.2)
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
  • Installing cmake (3.27.7)
  • Updating fastjsonschema (2.18.0 -> 2.18.1)
  • Updating filelock (3.12.2 -> 3.12.4)
  • Downgrading jsonschema (4.18.4 -> 4.17.3)
  • Updating jupyter-core (5.3.1 -> 5.4.0)
  • Installing lit (17.0.2)
  • Updating more-itertools (10.0.0 -> 10.1.0)
  • Installing mpmath (1.3.0)
  • Updating pyzmq (25.1.0 -> 25.1.1)
  • Updating setuptools (44.0.0 -> 68.2.2)
  • Updating urllib3 (1.26.18 -> 2.0.7): Failed

  CalledProcessError

  Command '['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y']' returned non-zero exit status 2.

  at /usr/lib/python3.8/subprocess.py:516 in run
       512│             # We don't call process.wait() as .__exit__ does that for us.
       513│             raise
       514│         retcode = process.poll()
       515│         if check and retcode:
    →  516│             raise CalledProcessError(retcode, process.args,
       517│                                      output=stdout, stderr=stderr)
       518│     return CompletedProcess(process.args, retcode, stdout, stderr)
       519│ 
       520│ 

The following error occurred when trying to handle this error:


  EnvCommandError

  Command ['/homes/glue/lava_env2/bin/python', '-m', 'pip', 'uninstall', 'urllib3', '-y'] errored with the following return code 2
  
  Output:
  ERROR: Exception:
  Traceback (most recent call last):
    File "/homes/glue/lava_env2/lib/python3.8/site-packages/pip/_internal/cli/base_command.py", line 186, in _main
      status = self.run(options, args)
    File "/homes/glue/lava_env2/lib/python3.8/site-packages/pip/_internal/commands/uninstall.py", line 51, in run
      session = self.get_default_session(options)
    File "/homes/glue/lava_env2/lib/python3.8/site-packages/pip/_internal/cli/req_command.py", line 74, in get_default_session
      self._session = self.enter_context(self._build_session(options))
    File "/homes/glue/lava_env2/lib/python3.8/site-packages/pip/_internal/cli/req_command.py", line 84, in _build_session
      session = PipSession(
    File "/homes/glue/lava_env2/lib/python3.8/site-packages/pip/_internal/network/session.py", line 241, in __init__
      self.headers["User-Agent"] = user_agent()
    File "/homes/glue/lava_env2/lib/python3.8/site-packages/pip/_internal/network/session.py", line 159, in user_agent
      setuptools_version = get_installed_version("setuptools")
    File "/homes/glue/lava_env2/lib/python3.8/site-packages/pip/_internal/utils/misc.py", line 637, in get_installed_version
      working_set = pkg_resources.WorkingSet()
    File "/usr/share/python-wheels/pkg_resources-0.0.0-py2.py3-none-any.whl/pkg_resources/__init__.py", line 567, in __init__
      self.add_entry(entry)
    File "/usr/share/python-wheels/pkg_resources-0.0.0-py2.py3-none-any.whl/pkg_resources/__init__.py", line 623, in add_entry
      for dist in find_distributions(entry, True):
    File "/usr/share/python-wheels/pkg_resources-0.0.0-py2.py3-none-any.whl/pkg_resources/__init__.py", line 2065, in find_on_path
      for dist in factory(fullpath):
    File "/usr/share/python-wheels/pkg_resources-0.0.0-py2.py3-none-any.whl/pkg_resources/__init__.py", line 2127, in distributions_from_metadata
      if len(os.listdir(path)) == 0:
  FileNotFoundError: [Errno 2] No such file or directory: '/homes/glue/lava_env2/lib/python3.8/site-packages/~sonschema-4.18.4.dist-info'
  

  at ~/.local/share/pypoetry/venv/lib/python3.8/site-packages/poetry/utils/env/base_env.py:354 in _run
      350│                 output = subprocess.check_output(
      351│                     cmd, stderr=stderr, env=env, text=True, **kwargs
      352│                 )
      353│         except CalledProcessError as e:
    → 354│             raise EnvCommandError(e)
      355│ 
      356│         return output
      357│ 
      358│     def execute(self, bin: str, *args: str, **kwargs: Any) -> int:

Cannot install urllib3.

  • Installing wheel (0.41.2)
```

### Change urllib version ... seems to interfere with poetry
```bash
(lava_env2) ~/lava_git_repos/lava$ pip uninstall urllib3
Found existing installation: urllib3 1.26.18
Uninstalling urllib3-1.26.18:
  Would remove:
    /homes/glue/lava_env2/lib/python3.8/site-packages/urllib3-1.26.18.dist-info/*
    /homes/glue/lava_env2/lib/python3.8/site-packages/urllib3/*
Proceed (y/n)? y
  Successfully uninstalled urllib3-1.26.18

~/lava_git_repos/lava$ pip install urllib3==2.0.7
WARNING: Keyring is skipped due to an exception: Failed to create the collection: Prompt dismissed..
WARNING: Keyring is skipped due to an exception: Failed to create the collection: Prompt dismissed..
Collecting urllib3==2.0.7
  WARNING: Keyring is skipped due to an exception: Failed to create the collection: Prompt dismissed..
  Downloading urllib3-2.0.7-py3-none-any.whl (124 kB)
     |████████████████████████████████| 124 kB 10.9 MB/s 
ERROR: poetry 1.5.1 has requirement urllib3<2.0.0,>=1.26.0, but you'll have urllib3 2.0.7 which is incompatible.
Installing collected packages: urllib3
```

### Retry lava-dl installation
```bash
(lava_env2) glue@ncl-edu:~/lava_git_repos/lava-dl$ poetry install
Installing dependencies from lock file

Package operations: 23 installs, 32 updates, 0 removals

  • Updating exceptiongroup (1.1.2 -> 1.1.3): Installing...
  • Updating jinja2 (3.0.3 -> 3.1.2): Installing...
  • Updating exceptiongroup (1.1.2 -> 1.1.3)
  • Updating jinja2 (3.0.3 -> 3.1.2)
  • Updating jupyter-client (8.3.0 -> 8.4.0)
  • Updating msgpack (1.0.5 -> 1.0.7)
  • Updating nbformat (5.9.1 -> 5.9.2)
  • Installing nvidia-cublas-cu11 (11.10.3.66): Installing...
  • Installing nvidia-cublas-cu11 (11.10.3.66)
  • Installing nvidia-cuda-cupti-cu11 (11.7.101)
  • Installing nvidia-cuda-nvrtc-cu11 (11.7.99)
  • Installing nvidia-cuda-runtime-cu11 (11.7.99)
  • Installing nvidia-cudnn-cu11 (8.5.0.96)
  • Installing nvidia-cufft-cu11 (10.9.0.58)
  • Installing nvidia-curand-cu11 (10.2.10.91)
  • Installing nvidia-cusolver-cu11 (11.4.0.1)
  • Installing nvidia-cusparse-cu11 (11.7.4.91)
  • Installing nvidia-nccl-cu11 (2.14.3)
  • Installing nvidia-nvtx-cu11 (11.7.91)
  • Updating packaging (23.1 -> 23.2)
  • Updating pluggy (1.2.0 -> 1.3.0)
  • Updating poetry-core (1.6.1 -> 1.7.0)
  • Updating rapidfuzz (2.15.1 -> 2.15.2)
  • Updating soupsieve (2.4.1 -> 2.5)
  • Installing sympy (1.12)
  • Installing triton (2.0.0)
  • Updating typing-extensions (4.7.1 -> 4.8.0)
  • Updating bleach (6.0.0 -> 6.1.0)
  • Updating cachecontrol (0.12.14 -> 0.13.1)
  • Installing click (8.1.7)
  • Updating contourpy (1.1.0 -> 1.1.1)
  • Updating cycler (0.11.0 -> 0.12.1)
  • Updating dulwich (0.21.5 -> 0.21.6)
  • Updating fonttools (4.41.1 -> 4.43.1)
  • Updating keyring (23.13.1 -> 24.2.0)
  • Updating kiwisolver (1.4.4 -> 1.4.5)
  • Installing mypy-extensions (1.0.0)
  • Installing pathspec (0.11.2)
  • Updating pillow (10.0.1 -> 10.1.0)
  • Updating poetry-plugin-export (1.4.0 -> 1.5.0)
  • Updating pygments (2.15.1 -> 2.16.1)
  • Updating pyparsing (3.0.9 -> 3.1.1)
  • Updating pytest (7.4.0 -> 7.4.2)
  • Installing setuptools-scm (8.0.4)
  • Updating shellingham (1.5.0.post1 -> 1.5.3)
  • Updating tomlkit (0.11.8 -> 0.12.1)
  • Installing torch (2.0.0)
  • Updating trove-classifiers (2023.7.6 -> 2023.9.19)
  • Updating virtualenv (20.24.2 -> 20.24.5)
  • Installing black (22.12.0)
  • Installing h5py (3.10.0)
  • Updating matplotlib (3.7.2 -> 3.7.3)
  • Installing ninja (1.11.1.1)
  • Installing opencv-python-headless (4.8.1.78)
  • Downgrading pep8-naming (0.12.1 -> 0.11.1)
  • Updating poetry (1.5.1 -> 1.6.1)
  • Installing torchvision (0.15.1)
  • Updating lava-nc (0.8.0.dev0 /homes/glue/lava_git_repos/lava -> 0.8.0.dev0 97b8db3)

Installing the current project: lava-dl (0.4.0.dev0)
```

```bash
(lava_env2) ~/lava_git_repos/lava-dl$ pytest

...

======================================================================================================================================================================== warnings summary ========================================================================================================================================================================
../../lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:256
  /homes/glue/lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:256: PytestDeprecationWarning: The hookimpl CovPlugin.pytest_configure_node uses old-style configuration options (marks or attributes).
  Please use the pytest.hookimpl(optionalhook=True) decorator instead
   to configure the hooks.
   See https://docs.pytest.org/en/latest/deprecations.html#configuring-hook-specs-impls-using-markers
    def pytest_configure_node(self, node):

../../lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:265
  /homes/glue/lava_env2/lib/python3.8/site-packages/pytest_cov/plugin.py:265: PytestDeprecationWarning: The hookimpl CovPlugin.pytest_testnodedown uses old-style configuration options (marks or attributes).
  Please use the pytest.hookimpl(optionalhook=True) decorator instead
   to configure the hooks.
   See https://docs.pytest.org/en/latest/deprecations.html#configuring-hook-specs-impls-using-markers
    def pytest_testnodedown(self, node, error):

../../lava_env2/lib/python3.8/site-packages/torch/utils/cpp_extension.py:25
  /homes/glue/lava_env2/lib/python3.8/site-packages/torch/utils/cpp_extension.py:25: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
    from pkg_resources import packaging  # type: ignore[attr-defined]

../../lava_env2/lib/python3.8/site-packages/pkg_resources/__init__.py:2871
  /homes/glue/lava_env2/lib/python3.8/site-packages/pkg_resources/__init__.py:2871: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('mpl_toolkits')`.
  Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
    declare_namespace(pkg)

../../lava_env2/lib/python3.8/site-packages/pkg_resources/__init__.py:2871
../../lava_env2/lib/python3.8/site-packages/pkg_resources/__init__.py:2871
../../lava_env2/lib/python3.8/site-packages/pkg_resources/__init__.py:2871
../../lava_env2/lib/python3.8/site-packages/pkg_resources/__init__.py:2871
  /homes/glue/lava_env2/lib/python3.8/site-packages/pkg_resources/__init__.py:2871: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('sphinxcontrib')`.
  Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
    declare_namespace(pkg)

tests/lava/lib/dl/netx/test_blocks.py:36
  /homes/glue/lava_git_repos/lava-dl/tests/lava/lib/dl/netx/test_blocks.py:36: PytestCollectionWarning: cannot collect test class 'TestRunConfig' because it has a __init__ constructor (from: tests/lava/lib/dl/netx/test_blocks.py)
    class TestRunConfig(RunConfig):

tests/lava/lib/dl/netx/test_hdf5.py:31
  /homes/glue/lava_git_repos/lava-dl/tests/lava/lib/dl/netx/test_hdf5.py:31: PytestCollectionWarning: cannot collect test class 'TestRunConfig' because it has a __init__ constructor (from: tests/lava/lib/dl/netx/test_hdf5.py)
    class TestRunConfig(RunConfig):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

---------- coverage: platform linux, python 3.8.10-final-0 -----------
Name                                                                Stmts   Miss  Cover   Missing
-------------------------------------------------------------------------------------------------
src/lava/lib/dl/netx/__init__.py                                        5      0   100%
src/lava/lib/dl/netx/blocks/__init__.py                                 0      0   100%
src/lava/lib/dl/netx/blocks/models.py                                  39      0   100%
src/lava/lib/dl/netx/blocks/process.py                                143      4    97%   66, 182, 265, 354
src/lava/lib/dl/netx/hdf5.py                                          272     47    83%   104, 142, 149, 160-161, 165, 189-202, 301, 361-388, 393, 411-413, 416, 422-426, 478, 574, 585, 600, 639-657
src/lava/lib/dl/netx/sequential_modules.py                             62     38    39%   57-61, 64, 67-68, 71-79, 92-93, 96, 99-101, 121-125, 129, 132-136, 148-151, 154, 157-158
src/lava/lib/dl/netx/utils.py                                          83     21    75%   103-106, 108-111, 114-115, 117-118, 127-130, 173-183
src/lava/lib/dl/slayer/__init__.py                                     13      0   100%
src/lava/lib/dl/slayer/auto.py                                        111     96    14%   28-31, 50-54, 88-98, 105-132, 137-141, 146-152, 157-208, 213-227, 233-242
src/lava/lib/dl/slayer/axon/__init__.py                                 3      0   100%
src/lava/lib/dl/slayer/axon/delay.py                                   71     41    42%   66-77, 82-86, 91-93, 109-142, 174-175, 184-198, 204-220
src/lava/lib/dl/slayer/axon/delta.py                                   91     16    82%   30-41, 114, 251, 259, 265-267, 272, 300, 316
src/lava/lib/dl/slayer/block/__init__.py                                2      0   100%
src/lava/lib/dl/slayer/block/adrf.py                                   92     49    47%   18-23, 40-42, 50, 58, 68, 76-80, 88-92, 100-104, 112-116, 124-128, 136-140, 148-154
src/lava/lib/dl/slayer/block/adrf_iz.py                                92     49    47%   18-23, 40-42, 50, 58, 69, 77-81, 89-93, 101-105, 113-117, 125-129, 137-141, 149-155
src/lava/lib/dl/slayer/block/alif.py                                   92     49    47%   18-23, 40-42, 50, 58, 68, 76-80, 88-92, 100-104, 112-116, 124-128, 136-140, 148-154
src/lava/lib/dl/slayer/block/base.py                                  536    349    35%   31-32, 34-35, 38, 66-103, 109-129, 135-138, 148-165, 179-182, 188-194, 204-207, 223-226, 233-239, 245, 255-258, 314, 328-347, 353, 369, 379-383, 391-392, 415-426, 432-450, 456, 466-469, 524, 540-544, 551, 554, 562, 578, 589, 592-596, 607, 609, 616-617, 623-624, 707, 710, 718, 734, 751, 753-757, 768, 770, 777-778, 784-785, 828-850, 855-865, 871, 881-921, 968-991, 996-1006, 1012, 1022-1079, 1122-1144, 1149-1159, 1165, 1175-1215, 1272-1307, 1310, 1324-1364, 1370, 1380, 1423-1458, 1466-1488, 1494, 1504
src/lava/lib/dl/slayer/block/cuba.py                                  105     33    69%   42-44, 52, 60, 88, 120-124, 132-136, 144-148, 156-160, 168-174
src/lava/lib/dl/slayer/block/rf.py                                    101     55    46%   18-23, 39-41, 49, 57, 65-71, 79, 87-91, 99-103, 111-115, 123-127, 135-139, 147-151, 159-165
src/lava/lib/dl/slayer/block/rf_iz.py                                 101     55    46%   18-23, 40-42, 50, 58, 66-72, 80, 88-92, 100-104, 112-116, 124-128, 136-140, 148-152, 160-166
src/lava/lib/dl/slayer/block/sigma_delta.py                           120     60    50%   26, 45-47, 53-66, 74, 82, 114-118, 126-130, 138-142, 192-201, 206-211, 217, 227-260
src/lava/lib/dl/slayer/classifier.py                                   48     24    50%   39, 44, 68, 98-104, 128-129, 216-231, 251, 256
src/lava/lib/dl/slayer/dendrite/__init__.py                             2      0   100%
src/lava/lib/dl/slayer/dendrite/sigma.py                               25      3    88%   44, 51, 77
src/lava/lib/dl/slayer/io.py                                          286    206    28%   47, 77, 80, 82, 86, 114-120, 163, 184-209, 227, 243, 255-296, 304-356, 364-441, 466-482, 516-545, 577, 583-587, 604, 610, 642-654, 679-698, 728-741, 768-788, 816-838, 862-881
src/lava/lib/dl/slayer/jitconfig.py                                     8      3    62%   16-18
src/lava/lib/dl/slayer/loss.py                                        100     58    42%   38-45, 50, 97-113, 118-141, 177-183, 188-208, 303-305, 310, 316, 326-327
src/lava/lib/dl/slayer/neuron/__init__.py                               5      0   100%
src/lava/lib/dl/slayer/neuron/adrf.py                                 208     83    60%   44-48, 169, 174, 179, 214-294, 383, 392, 402, 412, 417, 428, 466, 473-516, 519, 524, 549-558, 566, 569, 571, 655, 676-677
src/lava/lib/dl/slayer/neuron/adrf_iz.py                              208     83    60%   44-48, 169, 174, 179, 215-295, 384, 393, 403, 413, 418, 429, 467, 474-518, 521, 526, 551-560, 568, 571, 573, 648, 669-670
src/lava/lib/dl/slayer/neuron/alif.py                                 172     69    60%   45, 153, 158, 191-262, 319, 329, 339, 349, 354, 360, 371, 408, 415-448, 451, 477-486, 494, 497, 589, 609-610
src/lava/lib/dl/slayer/neuron/base.py                                  56     13    77%   136, 145-155, 213, 217
src/lava/lib/dl/slayer/neuron/cuba.py                                 119     33    72%   45, 138, 143, 162-193, 238, 248, 295, 302-320, 323, 365, 419
src/lava/lib/dl/slayer/neuron/dropout.py                                7      2    71%   31-32
src/lava/lib/dl/slayer/neuron/dynamics/__init__.py                      4      0   100%
src/lava/lib/dl/slayer/neuron/dynamics/adaptive_phase_th.py            78     20    74%   27-48, 123, 198-223
src/lava/lib/dl/slayer/neuron/dynamics/adaptive_resonator.py          193     92    52%   25-46, 148, 223-262, 280-372, 465-479, 482, 484, 487, 489, 497, 499, 513-520
src/lava/lib/dl/slayer/neuron/dynamics/adaptive_threshold.py           79     22    72%   27-48, 118, 195-226
src/lava/lib/dl/slayer/neuron/dynamics/leaky_integrator.py            103     42    59%   26-47, 95, 135-155, 169-205, 252-254
src/lava/lib/dl/slayer/neuron/dynamics/resonator.py                   187     95    49%   25-46, 125, 196-235, 252-327, 407-421, 424, 426, 429, 431, 439, 441, 458-462
src/lava/lib/dl/slayer/neuron/norm.py                                  87     70    20%   38-51, 55, 60, 65-93, 140-159, 163-164, 169-170, 177, 182, 187-230
src/lava/lib/dl/slayer/neuron/rf.py                                   153     32    79%   42-46, 149, 154, 159, 181-187, 190, 195, 203-204, 207, 299, 308, 319, 350, 358, 387, 392, 411-414, 422, 425, 427, 478, 499-500
src/lava/lib/dl/slayer/neuron/rf_iz.py                                156     54    65%   44-48, 153, 158, 163, 184-225, 303, 312, 323, 354, 361-388, 391, 396, 415-418, 426, 429, 431, 490, 511-512
src/lava/lib/dl/slayer/neuron/sigma_delta.py                           47     11    77%   31, 125, 145, 165-169, 191, 195, 209, 213
src/lava/lib/dl/slayer/object_detection/__init__.py                     4      0   100%
src/lava/lib/dl/slayer/object_detection/boundingbox/__init__.py         3      0   100%
src/lava/lib/dl/slayer/object_detection/boundingbox/metrics.py        158    143     9%   31-52, 71-80, 101-141, 161-166, 209-293, 316-318, 323-324, 338-344, 354-356, 376-385
src/lava/lib/dl/slayer/object_detection/boundingbox/utils.py          237    208    12%   60-106, 139-173, 205-238, 261-270, 287-299, 316-326, 347-350, 371-374, 393-409, 444-467, 486-507, 523-535, 551-563, 595-646, 682-691
src/lava/lib/dl/slayer/object_detection/dataset/__init__.py             2      0   100%
src/lava/lib/dl/slayer/object_detection/dataset/bdd100k.py            105     85    19%   36-38, 47-77, 80-107, 110, 145-161, 176-213, 218
src/lava/lib/dl/slayer/object_detection/models/__init__.py              2      0   100%
src/lava/lib/dl/slayer/object_detection/models/tiny_yolov3_str.py     159    148     7%   48-109, 145-202, 219-235, 248-329
src/lava/lib/dl/slayer/object_detection/models/yolo_kp.py             118    103    13%   50-107, 142-171, 183-206, 217-226, 242-254, 267-296
src/lava/lib/dl/slayer/object_detection/yolo_base.py                  173    151    13%   20-55, 60-82, 104-117, 132-167, 187-201, 242-263, 282-294, 316-358, 383-389, 406-407, 423-424, 436-440, 445-453
src/lava/lib/dl/slayer/spike/__init__.py                                3      0   100%
src/lava/lib/dl/slayer/spike/complex.py                                33      3    91%   68, 75, 145
src/lava/lib/dl/slayer/spike/spike.py                                  31      7    77%   21, 80-86
src/lava/lib/dl/slayer/synapse/__init__.py                              3      0   100%
src/lava/lib/dl/slayer/synapse/complex.py                              61     34    44%   15-18, 23, 27-28, 32-33, 38, 44-49, 55-56, 61, 98-110, 164-170, 222-227, 286-292, 345-350
src/lava/lib/dl/slayer/synapse/layer.py                               206    140    32%   28-29, 33-34, 39-50, 70, 112-119, 127, 137, 142, 160, 172, 240-243, 251-254, 262-265, 273-276, 290, 295, 313, 376-436, 453-502, 567-628, 645-653, 709-768, 817-834
src/lava/lib/dl/slayer/utils/__init__.py                                7      0   100%
src/lava/lib/dl/slayer/utils/assistant.py                              99     91     8%   59-66, 80-82, 102-145, 165-203, 223-260
src/lava/lib/dl/slayer/utils/filter/__init__.py                         4      0   100%
src/lava/lib/dl/slayer/utils/filter/conv.py                            68     27    60%   25-45, 80, 93, 124-131, 158-165, 192, 221
src/lava/lib/dl/slayer/utils/filter/fir.py                             37     15    59%   45, 78-93, 98-100, 110, 115, 120
src/lava/lib/dl/slayer/utils/int_utils.py                              15      4    73%   26, 43-44, 50
src/lava/lib/dl/slayer/utils/quantize.py                               33      6    82%   81, 109-114
src/lava/lib/dl/slayer/utils/recurrent.py                              70      0   100%
src/lava/lib/dl/slayer/utils/stats.py                                 197    172    13%   37-49, 53-55, 60-63, 68-71, 76, 81, 85-97, 102-118, 143-158, 162-167, 171, 176-183, 209-237, 254-305, 316-360, 365
src/lava/lib/dl/slayer/utils/time/__init__.py                           4      0   100%
src/lava/lib/dl/slayer/utils/time/replicate.py                          6      0   100%
src/lava/lib/dl/slayer/utils/time/shift.py                             59     16    73%   22-43, 50, 55, 75, 121, 128, 135, 139
src/lava/lib/dl/slayer/utils/utils.py                                  21     11    48%   22, 43-53, 71-74
-------------------------------------------------------------------------------------------------
TOTAL                                                                6383   3341    48%

Required test coverage of 45.0% reached. Total coverage: 47.66%
==================================================================================================================================================== 112 passed, 5 skipped, 10 warnings in 245.01s (0:04:05) =====================================================================================================================================================
```

```bash
(lava_env2) ~/lava_git_repos/lava-dl$ pip list | grep lava
lava-dl                       0.4.0.dev0
lava-nc                       0.8.0.dev0
```

## Run lava process ...
```bash
(lava_env2) glue@ncl-edu:~/oasis_test/hand_detection_SDNN_inference/dataloader_monitor_encoder_test$ python dataloader_monitor_encoder.py 
Loihi2 compiler is not available in this system. This tutorial will execute on CPU backend.
Dataset loaded: 424 samples found
run_spk
got frame data
got frame encoded data
got output data
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0]
(604,)
Figure(1000x500)
```
