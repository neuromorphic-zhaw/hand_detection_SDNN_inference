# Minimal lava process tests
## Raw DHP19 Data
Download the data from <https://drive.switch.ch/index.php/s/uudDYT0gU1j8o3I> and extract it in the `data/dhp19/` folder.

```
cd data/dhp19
wget https://drive.switch.ch/index.php/s/uudDYT0gU1j8o3I/download
unzip download
mv data/*.pt .
rm -rf download data/ __MACOSX/
ls data
```

The content of `data/dhp19/` should look like this.
```
S1_session1_mov1_sample0.pt  S1_session1_mov2_sample2.pt  S1_session1_mov3_sample4.pt  S1_session1_mov5_sample1.pt  S1_session1_mov7_sample0.pt  S1_session1_mov8_sample0.pt
S1_session1_mov1_sample1.pt  S1_session1_mov2_sample3.pt  S1_session1_mov3_sample5.pt  S1_session1_mov5_sample2.pt  S1_session1_mov7_sample1.pt  S1_session1_mov8_sample1.pt
S1_session1_mov1_sample2.pt  S1_session1_mov2_sample4.pt  S1_session1_mov4_sample0.pt  S1_session1_mov5_sample3.pt  S1_session1_mov7_sample2.pt  S1_session1_mov8_sample2.pt
S1_session1_mov1_sample3.pt  S1_session1_mov2_sample5.pt  S1_session1_mov4_sample1.pt  S1_session1_mov5_sample4.pt  S1_session1_mov7_sample3.pt  S1_session1_mov8_sample3.pt
S1_session1_mov1_sample4.pt  S1_session1_mov2_sample6.pt  S1_session1_mov4_sample2.pt  S1_session1_mov6_sample0.pt  S1_session1_mov7_sample4.pt  S1_session1_mov8_sample4.pt
S1_session1_mov1_sample5.pt  S1_session1_mov3_sample0.pt  S1_session1_mov4_sample3.pt  S1_session1_mov6_sample1.pt  S1_session1_mov7_sample5.pt  S1_session1_mov8_sample5.pt
S1_session1_mov1_sample6.pt  S1_session1_mov3_sample1.pt  S1_session1_mov4_sample4.pt  S1_session1_mov6_sample2.pt  S1_session1_mov7_sample6.pt  S1_session1_mov8_sample6.pt
S1_session1_mov2_sample0.pt  S1_session1_mov3_sample2.pt  S1_session1_mov4_sample5.pt  S1_session1_mov6_sample3.pt  S1_session1_mov7_sample7.pt  S1_session1_mov8_sample7.pt
S1_session1_mov2_sample1.pt  S1_session1_mov3_sample3.pt  S1_session1_mov5_sample0.pt  S1_session1_mov6_sample4.pt  S1_session1_mov7_sample8.pt
```

#



## Requirements
`lava`  and `lava-dl` installed from their repos (c.f. `Install_lava_CPUonly.md`)
```bash
pip list | grep lava
lava-dl                   0.4.0.dev0   /homes/glue/lava_git_repos/lava-dl
lava-nc                   0.8.0.dev0   /homes/glue/lava_git_repos/lava
```
- on `ncl-edu.research.intel-research.net`
