# Install lava form lastest branch (CPU only)
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
(lava_env2) ~/lava_git_repos/lava$ pip install -e .
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
Comment out the lava-nc dependency in `[tool.poetry.dependencies]`
```bash
(lava_env2) ~/lava_git_repos/lava-dl$ nano pyproject.toml
```
```bash
[tool.poetry.dependencies]
python = ">=3.8, <3.11"
#lava-nc = { git = "https://github.com/lava-nc/lava.git", branch = "main", develop = true }
```
### Install
```bash
pip install -e .
pip install jupyter
```
