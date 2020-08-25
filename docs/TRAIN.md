# Train
Please make sure you follow the instructions in [INSTALL.md](INSTALL.md) to build necessary dependencies.

To run the training code, you also need to download the mesh_release.zip in our [NBA2K dataset](https://github.com/luyangzhu/NBA2K-dataset) and unzip it on your local machine.

You might need to change the following options in `img_to_mesh/src/experiments/mesh/mesh_train.yaml`:
- `data_root_dir`: Set it to the mesh_release dataset path on your machine. 
- `vis_gpu`: Set it as your CUDA_VISIBLE_DEVICES.
- `base_log_dir`: The root dir of the mesh log folder. 
- `log_name`: Name of the mesh log folder. You can left it as None and `img_to_mesh/src/trainers/train_mesh.py` can handle it automatically. You can also manually identify it.
- `custom_postfix`: Postfix added to the end of log_name, default is ''. If you set it to a non-empty string, the final name of the mesh log folder should be ${log_name}-${custom_postfix}.


You also need to change the following lines in `img_to_mesh/src/experiments/mesh/mesh_run.sh`:
```
cfg_path=experiments/mesh/mesh_train.yaml
```

After you finish above steps, run the training code:
```
cd img_to_mesh/src
bash experiments/mesh/mesh_run.sh
```

# Play with hyperparameters
If you use the default setting in `img_to_mesh/src/experiments/mesh/mesh_train.yaml` and `img_to_mesh/src/constants.py`. You should get similar results as our pretrained checkpoints. You can also play with the hyperparameters if you want. 

You can change the training settings in the `training` subsection of `img_to_mesh/src/experiments/mesh/mesh_train.yaml`.

You can change the loss weights in the `loss` subsection of `img_to_mesh/src/experiments/mesh/mesh_train.yaml`.

For the Spiral Convolution network architecture, you should change the relevant parts in `img_to_mesh/src/constants.py`. Please see comments in `constants.py` for more details. You can also check the [original implementation](https://github.com/gbouritsas/Neural3DMM) of Spiral Conv.

**Note**: If you want to change `DS_FACTORS` in `constants.py`, you need to make sure you have installed opendr and MPI-MESH following [INSTALL.md](INSTALL.md). You also need to change `dataset.meshpackage` in `mesh_train.yaml` to 'mpi-mesh'. For the first run, it will compute some meta data for the Spiral Conv and save it to a local pkl file. For the following runs, it will directly load the saved meta data and you can use 'trimesh' as the `dataset.meshpackage`. For the default `DS_FACTORS`, we already provided the pkl files in `img_to_mesh/data/mesh/`.