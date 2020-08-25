# TEST
Please make sure you follow the instructions in [INSTALL.md](INSTALL.md) to build necessary dependencies and download our pretrained checkpoints.

To run the test code, you also need to download the mesh_release.zip in our [NBA2K dataset](https://github.com/luyangzhu/NBA2K-dataset) and unzip it on your local machine.

You might need to change the following options in `img_to_mesh/src/experiments/mesh/mesh_test.yaml`:
- `data_root_dir`: Set it to the mesh_release dataset path on your machine. 
- `vis_gpu`: Set it as your CUDA_VISIBLE_DEVICES.
- `base_log_dir`: The root dir of the mesh log folder. If you follow the instruction of downloading pretrained checkpoints, you don't need to change it.
- `log_name`: Name of the mesh log folder. If you follow the instruction of downloading pretrained checkpoints, you don't need to change it.
- `pose_base_log_dir`: The root dir of the pose log folder. If you follow the instruction of downloading pretrained checkpoints, you don't need to change it.
- `pose_log_name`: Name of the pose log folder. If you follow the instruction of downloading pretrained checkpoints, you don't need to change it.

You also need to change the following lines in `img_to_mesh/src/experiments/mesh/mesh_run.sh`:
```
cfg_path=experiments/mesh/mesh_test.yaml
```

After you finish above steps, run the testing code:
```
cd img_to_mesh/src
bash experiments/mesh/mesh_run.sh
```

# Performance
We report the performance on the released NBA2K Dataset. The results are slightly different from what we provide in the paper as the dataset is different.
| MPVPE   |  MPVPE-PA  |  EMD  |   CD  |
| :-------: | :----------: | :-----: | :-----: |
| 65.991  |   44.622   | 0.073 | 2.870 |

**MPVPE** is mean per vertex position error in mm. **MPVPE-PA** is mean per vertex position error in mm after procrustes alignment. **EMD** is earth-mover distance. **CD** is chamfer distance scaled by 1000. EMD and CD are computed after Iterative Closest Point alignment.